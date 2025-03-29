import os
import mne
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.spatial import distance
from scipy.interpolate import griddata

class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = eeg_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.eeg_data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
    
# Load in first 8 trials of eeg data for each subject and extract 2D electrode coordinates
def load_kuleuven_aad_data(data_dir):
    """
    Load EEG data and labels from the KULeuven AAD dataset.

    Args:
        data_dir (str): Path to the dataset directory containing .mat files.

    Returns:
        subjects_data (list): A list of subjects, where each subject is a dictionary with:
            - 'trials': A list of dictionaries, each containing:
                - 'eeg': EEG signals of shape (min_length, features).
                - 'label': Corresponding label (0 for 'L', 1 for 'R').
    """
    subjects_data = []
    channel_names = []
    min_length = (6 * 60) * 128  # 6 minutes truncate length (128 Hz)

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".mat"):
            file_path = os.path.join(data_dir, file_name)
            mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)

            if 'trials' not in mat_data:
                print(f"Skipping {file_name}: 'trials' key not found.")
                continue

            trials = mat_data['trials']
            if not isinstance(trials, (list, np.ndarray)):
                trials = [trials]  # Ensure trials is iterable

            # get channel position names
            if len(channel_names) == 0:
                trial = trials[0]
                file_header = trial.FileHeader
                channels = file_header.Channels
                for channel in channels:
                    channel_names.append(channel.Label)

            subject_trials = []
            for trial in trials:
                # Skip the last 12 trials (experiment 3, repeated segments)
                if trial.TrialID == 9:
                    break

                label_char = getattr(trial, 'attended_ear', None)
                raw_data = getattr(trial, 'RawData', None)

                if label_char not in ('L', 'R') or raw_data is None:
                    print(f"Skipping trial in {file_name}: Missing 'attended_ear' or 'RawData'.")
                    continue

                eeg_data = getattr(raw_data, 'EegData', None)
                if eeg_data is None or not isinstance(eeg_data, np.ndarray) or eeg_data.ndim != 2:
                    print(f"Skipping trial in {file_name}: Invalid 'EegData' format.")
                    continue

                # Truncate EEG data
                eeg_data_truncated = eeg_data[:min_length]

                subject_trials.append({
                    'eeg': eeg_data_truncated.astype(np.float32),
                    'label': 1 if label_char == 'R' else 0
                })

            if subject_trials:
                subjects_data.append({'name': file_name, 'trials': subject_trials})

    # get electrode positions from the channel names
    montage = mne.channels.make_standard_montage("standard_1020")
    electrode_positions = {}
    for ch in channel_names:
        if ch in montage.ch_names:
            electrode_positions[ch] = montage.get_positions()["ch_pos"][ch]

    # Transform the 3D coordinates to the 2D plane
    electrode_positions = azimuthal_equidistant_proj(electrode_positions)

    if not subjects_data:
        raise ValueError("No valid EEG data found.")

    return subjects_data, electrode_positions, channel_names

def segment_eeg_data(trial, window_size=128, overlap=0.5):
    """
    Segments EEG data with overlap per trial.
    
    Args:
        eeg_data (numpy array): Shape (num_trials, time_steps, num_channels)
        labels (numpy array): Shape (num_trials,)
        window_size (int): Number of time steps in each segment
        overlap (float): Overlap percentage (0.5 = 50%)
    
    Returns:
        segmented_data (numpy array): Segmented EEG data
        segmented_labels (numpy array): Corresponding labels
    """
    step_size = int(window_size * (1 - overlap))  # Calculate step size for sliding window
    segmented_data = []
    segmented_labels = []
    trial_data = trial['eeg']

    # Create sliding windows
    for start in range(0, trial_data.shape[0] - window_size + 1, step_size):
        end = start + window_size
        segment = trial_data[start:end, :]
        segmented_data.append(segment)

    return np.array(segmented_data)

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the EEG data.
    :param data: EEG data (2D or 3D array).
    :param lowcut: Lower frequency cutoff (Hz).
    :param highcut: Upper frequency cutoff (Hz).
    :param fs: Sampling frequency (Hz).
    :param order: Order of the filter.
    :return: Filtered data.
    """
    # Design the filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter to each channel in the EEG data
    if len(data.shape) == 3:  # (samples, time_steps, channels)
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):  # Iterate over samples
            for j in range(data.shape[2]):  # Iterate over channels
                filtered_data[i, :, j] = filtfilt(b, a, data[i, :, j])
    else:  # 2D data (time_steps, channels)
        filtered_data = np.zeros_like(data)
        for j in range(data.shape[1]):  # Iterate over channels
            filtered_data[:, j] = filtfilt(b, a, data[:, j])
    
    return filtered_data

# Compute the alpha power band (8-13 hz) by taking the sum of average absolute squared value for each channel
def compute_alpha_power(eeg_data, sfreq=128):
    """
    Compute alpha band (8-13 Hz) power for EEG data
    Args:
        eeg_data: (n_samples, n_timesteps, n_channels)
        sfreq: Sampling frequency (Hz)
    Returns:
        alpha_power: (n_samples, n_channels)
    """
    n_samples, n_timesteps, n_channels = eeg_data.shape
    
    # Compute FFT and power spectrum
    fft_vals = np.abs(fft(eeg_data, axis=1)) ** 2
    freqs = fftfreq(n_timesteps, 1/sfreq)
    
    # Use only positive frequencies
    positive_freqs = freqs[:n_timesteps//2]
    fft_vals = fft_vals[:, :n_timesteps//2, :]
    
    # Sum power in alpha band (8-13 Hz)
    alpha_mask = (positive_freqs >= 8) & (positive_freqs <= 13)
    alpha_power = np.sum(fft_vals[:, alpha_mask, :], axis=1)
    
    return alpha_power

# z-score channel-wise normalization
def normalize_data(eeg_data):
    mean = np.mean(eeg_data, axis=0)
    std = np.std(eeg_data, axis=0)
    return (eeg_data - mean) / std

# Maps 3D EEG electrode positions to a 2D plane using Azimuthal Equidistant Projection.
def azimuthal_equidistant_proj(electrode_positions, center="Cz"):
    # Get center electrode position (e.g., Cz as reference)
    center_pos = np.array(electrode_positions[center])
    
    projected_2d = {}
    
    for elec, pos in electrode_positions.items():
        pos = np.array(pos)
        
        # Compute geodesic distance from center electrode
        d = distance.euclidean(pos, center_pos)
        
        # Compute azimuthal angle
        theta = np.arctan2(pos[1], pos[0])  # atan2(y, x) for angle
        
        # Apply projection formulas
        X = d * np.cos(theta)
        Y = d * np.sin(theta)
        
        projected_2d[elec] = (X, Y)
    
    return projected_2d

# Create a 2D topological map (grid_resolution, grid_resolution) from channel values and positions
def create_topo_map(channel_values, electrode_positions, channel_names, grid_resolution=64):
    """
    Create 2D topological maps using azimuthal equidistant projection
    
    Args:
        channel_values: 1D array of alpha power values (64,)
        electrode_positions_3d: Dictionary of 3D electrode positions {name: [x,y,z]}
        channel_names: List of channel names in order of channel_values
        center_ch: Name of center channel for projection
        grid_resolution: Size of output 2D map
        
    Returns:
        2D topological map (grid_resolution, grid_resolution)
    """
    
    # Prepare coordinates and values for interpolation
    x, y, values = [], [], []
    for ch_name, val in zip(channel_names, channel_values):
        if ch_name in electrode_positions:
            x.append(electrode_positions[ch_name][0])
            y.append(electrode_positions[ch_name][1])
            values.append(val)
    
    x = np.array(x) # x position
    y = np.array(y) # y position
    values = np.array(values)
    
    # Normalize coordinates to [0,1] range
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    
    # Create grid
    xi = yi = np.linspace(0, 1, grid_resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate using cubic interpolation
    zi = griddata((x, y), values, (xi, yi), method='cubic')
    
    # Fill NaN values with nearest neighbor
    if np.isnan(zi).any():
        zi_nearest = griddata((x, y), values, (xi, yi), method='nearest')
        zi[np.isnan(zi)] = zi_nearest[np.isnan(zi)]
    
    return zi

# Bandpass filter the data and segment the data into windows
# extract the alpha power for each channel in each window and apply channel-wise normalization
def preprocess(subjects_data, electrode_positions, channel_names):
    print("preprocess begin")
    for subject_data in subjects_data:
        for trial in subject_data['trials']:
            eeg_data = trial['eeg']  # EEG data (NumPy array)
            trial['eeg'] = bandpass_filter(eeg_data, lowcut=1.0, highcut=45.0, fs=128)
            trial['eeg'] = segment_eeg_data(trial)
            trial['eeg'] = compute_alpha_power(trial['eeg']) # feature extraction
            trial['eeg'] = normalize_data(trial['eeg'])
            print("create topopmaps")
            # create topo map for each window
            topomaps = []
            for window in trial['eeg']:
                window = create_topo_map(window, electrode_positions, channel_names)
                topomaps.append(window)
            
            trial['eeg'] = np.array(topomaps)
            print("done with topopmaps")
    print("preprocess complete")
    # create 2d topo maps from electrode positions
    return subjects_data

# -----------------------------
# Neural Network Models
# -----------------------------
# 2 convolutional layer
class EEGNet(nn.Module):
    def __init__(self, num_channels, num_timesteps, num_classes):
        super(EEGNet, self).__init__()
        
        # # Convolutional Layer to extract spatial and temporal features
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        # self.batchnorm1 = nn.BatchNorm2d(16)

        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        # self.batchnorm2 = nn.BatchNorm2d(32)
        
        # Convolutional Layer with Weight Normalization
        self.conv1 = weight_norm(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1))
        self.batchnorm1 = nn.BatchNorm2d(16)

        self.conv2 = weight_norm(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1))
        self.batchnorm2 = nn.BatchNorm2d(32)

        # Activation
        self.relu = nn.ReLU()
        
        # Pooling layer to reduce dimensions
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # # Fully connected layer for classification
        # self.fc1 = nn.Linear(32 * (num_timesteps // 4) * (num_channels // 4), 64)  # Adjust dimensions after pooling
        # self.fc2 = nn.Linear(64, num_classes)
        
        # Fully connected layer with Weight Normalization
        self.fc1 = weight_norm(nn.Linear(32 * (num_timesteps // 4) * (num_channels // 4), 64))
        self.fc2 = weight_norm(nn.Linear(64, num_classes))

        # Dropout for regularization
        self.dropout_conv1 = nn.Dropout(0.1)  # Dropout after first convolutional layers
        self.dropout_conv2 = nn.Dropout(0.1) # Droptou after second convolutional layers
        self.dropout_fc = nn.Dropout(0.1)    # Dropout after fully connected layers

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension (batch_size, 1, time_steps, channels)

        # Convolutional layers
        x = self.pool(self.relu(self.batchnorm1(self.conv1(x))))
        # x = self.dropout_conv1(x)
        x = self.pool(self.relu(self.batchnorm2(self.conv2(x))))
        # x = self.dropout_conv2(x)
        
        # fully connected layers
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

# 3 convolutional layer
class EEGNet2(nn.Module):
    def __init__(self, num_classes=2, num_channels=64, input_size=128):
        """
        EEGNet-inspired CNN for EEG classification.

        Args:
        - num_classes (int): Number of output classes.
        - num_channels (int): Number of EEG channels (default: 64).
        - input_size (int): Number of time steps per segment.
        """
        super(EEGNet2, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=1
        )
        self.batch_norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(64)
        

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1
        )
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Flatten and Fully Connected Layer
        # after 3 pooling layers: 128 feature maps with size  128/8 x 64/8
        self.fc = nn.Linear(128 * (input_size // 8) * (num_channels // 8), 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.3)  # Dropout after first convolutional layers

    def forward(self, x):
        x = x.unsqueeze(1) # change shape -> (batch_size, 1, num_timesteps, num_channels)

        # convolutional layers
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.dropout(x)
        
        # fully connected layers
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -----------------------------
# Training and Evaluation
# -----------------------------
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

if __name__ == "__main__":
    data_dir = "./KUL"  # Update with actual dataset path
    
    print("Loading KULeuven AAD data...")
    try:
        subjects_data, electrode_positions, channel_names = load_kuleuven_aad_data(data_dir) 
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

    print(f"Loaded {len(subjects_data)} subjects.")

    subjects_data = preprocess(subjects_data, electrode_positions, channel_names)

    # Prepare data for cross-validation
    X, y, groups = [], [], []

    for subject in subjects_data:
        trial_counter = 0 
        for trial in subject['trials']:
            eeg_windows = trial['eeg']  # Shape: (windows, resolution, resolution)
            num_windows = eeg_windows.shape[0]  # Get number of windows

            # Reshape each window to (1, resolution, resolution) and add to X
            for window in eeg_windows:
                print("window shapa: ", window.shape)
                X.append(window)  # Add a new axis for channel dim
                y.append(trial['label'])
                groups.append(trial_counter)
            
            trial_counter += 1

    # Convert to NumPy arrays
    X = np.array(X)  # Shape: (total_windows, 1, time_steps, channels)
    y = np.array(y)  # labels for each window
    groups = np.array(groups) # subject index for each window
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    print("groups len: ", len(groups))

    # leave one subject out cross-validation
    logo = LeaveOneGroupOut()
    epochs = 10
    results = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"Test subject(fold): {set(groups[i] for i in test_idx)}")
        # print(X_train.shape, X_test.shape)
        # print(np.bincount(y_train))

        train_dataset = EEGDataset(X_train, y_train)
        test_dataset = EEGDataset(X_test, y_test)
        num_channels = X_train.shape[2]
        num_timesteps = X_train.shape[1]

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_timesteps, num_channels = X_train.shape[1], X_train.shape[2]
        model = EEGNet2(num_classes=2, num_channels=num_channels, input_size=num_timesteps).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0003)

        # Training loop
        for epoch in range(epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Evaluation
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        results.append(test_acc)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    print(f"\nMean Test Accuracy: {np.mean(results):.4f} ± {np.std(results):.4f}")



