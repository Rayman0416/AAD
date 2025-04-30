import os
import mne
import math
import scipy.io as sio
import numpy as np
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
from scipy.interpolate import griddata
from sklearn.preprocessing import scale

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
            # self.eeg_data[idx].clone().detach(),  # Recommended copy method
            # self.labels[idx].clone().detach()
        )
    
# Load in first 8 trials of eeg data for each subject and extract 2D electrode coordinates
def load_kul(data_dir):
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

    if not subjects_data:
        raise ValueError("No valid EEG data found.")

    subjects_data.sort(key=lambda x: x['name'])

    return subjects_data, channel_names

def load_DTU(data_dir):
    subjects_data = []
    channel_names = []
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith("preproc128.mat"):
            subject_trials = []
            file_path = os.path.join(data_dir, file_name)
            mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
            data = mat_data['data']

            if len(channel_names) == 0:
                channels = data.dim.chan.eeg[0]
                for channel in channels:
                    channel_names.append(channel)
            
            trial_labels = data.event.eeg
            trial_eeg = data.eeg
            
            for eeg, label in zip(trial_eeg, trial_labels):
                # re-reference eeg channels to the average of the mastoid channels
                mastoid = eeg[:, -2:]
                mastoid_avg = np.mean(mastoid, axis=1, keepdims=True)
                eeg = eeg[:, :-2] # drop the last 2 (mastoid) channels
                eeg = eeg - mastoid_avg

                subject_trials.append({
                    'eeg': eeg,
                    'label': 1 if label.value == 2 else 0
                })
        
            if subject_trials:
                subjects_data.append({'name': file_name, 'trials': subject_trials})
    
    if not subjects_data:
        raise ValueError("No valid EEG data found.")
    
    subjects_data.sort(key=lambda x: x['name'])

    return subjects_data, channel_names

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
    window_length = eeg_data.shape[1]
    
    alpha_low = 1
    alpha_high = 13
    frequency_resolution = sfreq / window_length
    point_low = math.ceil(alpha_low / frequency_resolution)
    point_high = math.ceil(alpha_high / frequency_resolution) + 1

    alpha_data = []
    for window in eeg_data:
        window_data = np.fft.fft(window, n=window_length, axis=0)
        window_data = np.abs(window_data) / window_length
        window_data = np.sum(np.power(window_data[point_low:point_high, :], 2), axis=0)
        alpha_data.append(window_data)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

# z-score channel-wise normalization
def normalize_data(eeg_data):
    mean = np.mean(eeg_data, axis=0)
    std = np.std(eeg_data, axis=0)
    return (eeg_data - mean) / std

def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x**2 + y**2
    r = math.sqrt(x2_y2 + z**2)
    elev = math.atan2(z, math.sqrt(x2_y2))
    az = math.atan2(y, x)
    return r, elev, az

def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * math.cos(theta), rho * math.sin(theta)

# Maps 3D EEG electrode positions to a 2D plane using Azimuthal Equidistant Projection.
def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, math.pi / 2 - elev)

def create_topo_map(channel_values, channel_names, grid_resolution=32, reduced_channels=None):
    """
    Create 2D (reduced) topological maps using cubic interpolation.
    
    Args:
        channel_values: array of alpha power values (windows, channel values).
        channel_names: List of channel names matching channel_values.
        grid_resolution: Size of output 2D map (grid_resolution x grid_resolution).
        reduced_channels: List of channels to include in the map. If None, all channels are used.
        
    Returns:
        2D topological maps (window, grid_resolution, grid_resolution).
    """
    channel_values = np.array(channel_values)  # Convert to NumPy array

    # get electrode positions from the channel names
    montage = mne.channels.make_standard_montage("standard_1020")
    locs_2d = []

    # If reduced_channels is provided, create 2D locations for only those channels
    if reduced_channels is None:
        for ch in channel_names:
            if ch in montage.ch_names:
                coords = montage.get_positions()["ch_pos"][ch]
                locs_2d.append(azim_proj(coords))
    else:
        for ch in reduced_channels:
            if ch in montage.ch_names:
                coords = montage.get_positions()["ch_pos"][ch]
                locs_2d.append(azim_proj(coords))
    
    locs_2d_final = np.array(locs_2d)

    # Create grid for interpolation
    grid_x, grid_y = np.mgrid[
        min(locs_2d_final[:, 0]):max(locs_2d_final[:, 0]):grid_resolution * 1j,
        min(locs_2d_final[:, 1]):max(locs_2d_final[:, 1]):grid_resolution * 1j
    ]
    
    # Prepare the output images
    images = []
    for i in range(channel_values.shape[0]):  # For each window in the batch
        # Interpolate using cubic interpolation (like griddata with 'cubic' method)
        interpolated_image = griddata(locs_2d_final, channel_values[i, :], (grid_x, grid_y), method='cubic', fill_value=np.nan)
        images.append(interpolated_image)

    images = np.stack(images, axis=0)

    # Scale non-NaN values
    images[~np.isnan(images)] = scale(images[~np.isnan(images)])

    # Replace NaNs with zero or another fill value
    images = np.nan_to_num(images, nan=0)

    return images

# Bandpass filter the data and segment the data into windows
# extract the alpha power for each channel in each window and apply channel-wise normalization
# create a topographical map for each window
def window_split(subject):
    print("window split begin")
    for trial in subject['trials']:
        eeg_data = trial['eeg']  # EEG data (NumPy array)
        trial['eeg'] = bandpass_filter(eeg_data, lowcut=1.0, highcut=45.0, fs=128)
        trial['eeg'] = segment_eeg_data(trial)
    
    print("window split complete")
    return subject

# save image of a topographical map of the alpha power range of signals
def save_topo_map_image(topo_map: np.ndarray,
                        output_filepath: str,
                        title: str = None,
                        cmap: str = 'viridis',
                        show_axis: bool = False,
                        colorbar: bool = True,
                        dpi: int = 150):
    """
    Generates an image from a 2D topographic map array and saves it to a file.

    Args:
        topo_map (np.ndarray): The 2D NumPy array (grid_resolution, grid_resolution)
                               representing the topographic map.
        output_filepath (str): The full path and filename for the output image
                               (e.g., 'plots/topo_map_01.png'). The file format
                               is inferred from the extension (.png, .jpg, .pdf, etc.).
        title (str, optional): A title to add to the plot. Defaults to None.
        cmap (str, optional): The matplotlib colormap to use ('viridis', 'gray',
                              'jet', 'coolwarm', etc.). Defaults to 'viridis'.
        show_axis (bool, optional): Whether to display the coordinate axes and ticks.
                                    Defaults to False (axes hidden).
        colorbar (bool, optional): Whether to add a colorbar legend to the image.
                                   Defaults to True.
        dpi (int, optional): Dots Per Inch - the resolution of the saved image.
                             Defaults to 150.
    """
    if not isinstance(topo_map, np.ndarray) or topo_map.ndim != 2:
        raise ValueError("topo_map must be a 2D NumPy array.")

    fig, ax = plt.subplots(figsize=(6, 6)) # Adjust figsize as needed

    # Display the 2D map array as an image
    # origin='upper' places the [0,0] index at the top-left corner
    im = ax.imshow(topo_map, cmap=cmap, origin='upper', interpolation='hanning') # Use interpolation for smoother look

    # Add a colorbar if requested
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # Adjust fraction/pad as needed

    # Set the title if provided
    if title:
        ax.set_title(title)

    # Hide axes and ticks if requested
    if not show_axis:
        ax.axis('off')

    # Ensure the output directory exists (optional, good practice)
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Save the figure
    try:
        # bbox_inches='tight' and pad_inches=0 help remove extra whitespace,
        # especially when axes are off.
        plt.savefig(output_filepath, dpi=dpi, bbox_inches='tight', pad_inches=0)
        # print(f"Saved topographic map to: {output_filepath}") # Optional confirmation
    except Exception as e:
        print(f"Error saving file {output_filepath}: {e}")

    # Close the plot figure to free up memory
    plt.close(fig)

# -----------------------------
# Neural Network Models
# -----------------------------
# 1 convolutional layer

class Rayanet(nn.Module):
    def __init__(self, in_channels=1):  # Change in_channels if your images have a different number of channels
        super(Rayanet, self).__init__()
        self.features = nn.Sequential(
            # Input: (in_channels) x 32 x 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  
            nn.Dropout(0.1)
        )
        
        # Classifier: flatten the features and pass through fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(16*16*32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x  # raw logits

# 3 convolutional layer
class EEGNet2(nn.Module):
    def __init__(self, in_channels=1):  
        super(EEGNet2, self).__init__()

        self.features = nn.Sequential(
            # Block 1: Output size = 32 x 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 16 x 16
            nn.Dropout(0.2),

            # Block 2: Output size = 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 8 x 8
            nn.Dropout(0.3),

            # Block 3: Output size = 8 x 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=2, stride=2),  # Output: 4 x 4
            nn.Dropout(0.4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension if needed (for grayscale input)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x  # Raw logits

# -----------------------------
# Training and Evaluation
# -----------------------------
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        predictions = (torch.sigmoid(outputs) > 0.5).float()  # Convert logits to class 0/1
        correct += (predictions == labels).sum().item()

    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()  # Convert logits to class 0/1
            correct += (predictions == labels).sum().item()

    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

# Project EEG channels into 2D space using azim_proj
def get_channel_pixel_mapping(channel_names, grid_size=32):
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_pos = montage.get_positions()["ch_pos"]
    xy_coords = {}
    for ch in channel_names:
        if ch in ch_pos:
            x, y = azim_proj(ch_pos[ch])
            xy_coords[ch] = (x, y)

    # Normalize coords to grid indices (0 to grid_size-1)
    xs = np.array([coord[0] for coord in xy_coords.values()])
    ys = np.array([coord[1] for coord in xy_coords.values()])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    pixel_map = {}
    for ch, (x, y) in xy_coords.items():
        norm_x = int((x - x_min) / (x_max - x_min) * (grid_size - 1))
        norm_y = int((y - y_min) / (y_max - y_min) * (grid_size - 1))
        pixel_map[ch] = (norm_y, norm_x)  # Note: imshow is row-major (y, x) (row, column)

    return pixel_map

# Group trials into subsets of 5 trials each
def group_DTU_trials(subject):
    X, y, groups = [], [], []

    subset_size = 5
    trial_subsets = [
        subject["trials"][i:i + subset_size]
        for i in range(0, len(subject["trials"]), subset_size)
    ]

    # Process subsets to extract X, y, and groups
    subset_id = 0  # Unique identifier for each subset
    for subset in trial_subsets:
        for trial in subset:
            eeg_windows = trial['eeg']  # Shape: (windows, resolution, resolution)
            trial_label = trial['label']
            
            for window in eeg_windows:
                X.append(window)       # Add EEG window (shape: resolution, resolution)
                y.append(trial_label)  # Assign label
                groups.append(subset_id)  # Assign all windows to the subset group

        subset_id += 1  # Increment subset ID for next group

    return X, y, groups

# Group window data into trials
def group_KUL_trials(subject):
    X, y, groups = [], [], []
    trial_counter = 0 
    for trial in subject['trials']:
        eeg_windows = trial['eeg']  # Shape: (windows, resolution, resolution)
        
        # add each window to the dataset and assign corresponding label
        for window in eeg_windows:
            X.append(window)
            y.append(trial['label'])
            groups.append(trial_counter)
        
        trial_counter += 1
    
    return X, y, groups

def preprocess(data, channel_names, reduced_channels=None):
    data = compute_alpha_power(data)
    data = normalize_data(data)
    if reduced_channels is not None:
        data = create_topo_map(data, channel_names, reduced_channels=reduced_channels)
    else:
        data = create_topo_map(data, channel_names)

    return data
    
def reduce_channels(original_values, shap_values, channel_names, reduction=48):
    """
    Reduce the number of channels in the SHAP values to a smaller set.
    
    Args:
        shap_values: SHAP values for each channel.
        channel_names: List of channel names.
        original_values: Original channel values. (trials, windows, channel alpha values)
    """

    # Get the pixel mapping for the channels
    pixel_map = get_channel_pixel_mapping(channel_names)
    channel_shap_importance = {}

    # Rank channels based on their SHAP values
    for channel, (y, x) in pixel_map.items():
        # Get the SHAP value for the channel's pixel
        channel_shap = shap_values[y, x]

        channel_shap_importance[channel] = channel_shap
    
    sorted_channel_shap = sorted(channel_shap_importance.items(), key=lambda x: x[1], reverse=True)
    
    for i, (channel, score) in enumerate(sorted_channel_shap):
        print(f"Rank {i+1}: Channel {channel} - SHAP Value: {score:.4f}")
    
    # Select the top channels based on SHAP values top 50
    top_channels = [channel for channel, _ in sorted_channel_shap[:reduction]]
    top_channel_indices = [i for i, name in enumerate(channel_names) if name in top_channels]

    original_values = np.array(original_values)  # Convert to NumPy array
    original_values = original_values[:, :, top_channel_indices]  # Select only the top channels

    return original_values, top_channels

if __name__ == "__main__":
    data_KUL = "./KUL"  # KUL data dir
    data_DTU = "./DTU"  # DTU data dir

    # choose dataset and subject number to test
    dataset = input("Enter dataset (KUL or DTU): ").strip().upper()
    if dataset == "DTU":
        subject_nr = input("Enter subject number (1-18): ")
    elif dataset == "KUL":
        subject_nr = input("Enter subject number (1-16): ")

    # Load the dataset
    if dataset == "DTU":
        print("Loading DTU AAD data...")
        try:
            subjects_data, channel_names = load_DTU(data_DTU) 
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")
    elif dataset == "KUL":
        print("Loading KUL AAD data...")
        try:
            subjects_data, channel_names = load_kul(data_KUL)
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    print(f"Loaded {len(subjects_data)} subjects from {dataset} dataset.")

    subject = subjects_data[int(subject_nr) - 1]  # Select subject by number
    print(f"Processing subject {subject['name']}...")
    
    subject = window_split(subject)
    print("subject shape: ", subject['trials'][0]['eeg'].shape)

    # Prepare data for cross-validation
    X, y, groups = [], [], []

    if dataset == "DTU":
        X, y, groups = group_DTU_trials(subject)
    elif dataset == "KUL":
        X, y, groups = group_KUL_trials(subject)

    X = np.array(X)  # Convert to NumPy array
    y = np.array(y)
    groups = np.array(groups)
    print(f"X shape: {X.shape}, y shape: {y.shape}, groups shape: {groups.shape}")
    original_values = X.copy()  # Keep original values for SHAP calculation

    folds = 10
    epochs = 20
    results = []
    all_shap_values = []
    for fold in range(folds):
        best_val_loss = float('inf')
        best_model = None
        print(f"Fold: {fold + 1}")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42 + fold
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42 + fold
        )

        # Preprocess the data
        X_train = preprocess(X_train, channel_names)
        X_val = preprocess(X_val, channel_names)
        X_test = preprocess(X_test, channel_names)

        # Create DataLoader
        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        test_dataset = EEGDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EEGNet2(in_channels=1).to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0003)

        # Training loop
        for epoch in range(epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
        
        model.load_state_dict(best_model)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        results.append(test_acc)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # calculate SHAP values
        # SHAP on TRAINING SET ONLY to avoid leakage
        model.eval()

        # Use a small subset of training data as background
        background = torch.from_numpy(X_train[:100]).float().to(device)
        explainer = shap.DeepExplainer(model, background)

        # Compute SHAP on the rest of the training set (or all of it)
        X_train_tensor = torch.from_numpy(X_train).float().to(device)
        shap_values_fold = explainer.shap_values(X_train_tensor)  # shape: (batch, time, channels)
        shap_values_fold = shap_values_fold.squeeze()  # remove singleton dims if needed

        all_shap_values.append(shap_values_fold)

    print(f"\nMean Test Accuracy: {np.mean(results):.4f} ± {np.std(results):.4f}")

    # Concatenate across folds
    final_shap_values = np.concatenate(all_shap_values, axis=0)
    print(f"concat SHAP shape: {final_shap_values.shape}")

    # Average absolute SHAP across all samples
    mean_shap_per_pixel = np.mean(np.abs(final_shap_values), axis=0)
    print(f"mean SHAP shape: {mean_shap_per_pixel.shape}")

    
    X, reduced_channels = reduce_channels(original_values, mean_shap_per_pixel, channel_names, reduction=32)
    print(f"X shape after reduction: {X.shape}")

    shap_results = []
    for fold in range(folds):
        best_val_loss = float('inf')
        best_model = None
        print(f"Fold: {fold + 1}")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42 + fold
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42 + fold
        )

        # Preprocess the data
        X_train = preprocess(X_train, channel_names, reduced_channels=reduced_channels)
        X_val = preprocess(X_val, channel_names, reduced_channels=reduced_channels)
        X_test = preprocess(X_test, channel_names, reduced_channels=reduced_channels)

        # Create DataLoader
        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        test_dataset = EEGDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EEGNet2(in_channels=1).to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0003)

        # Training loop
        for epoch in range(epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
        
        model.load_state_dict(best_model)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        shap_results.append(test_acc)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    print(f"\nMean Test Accuracy: {np.mean(results):.4f} ± {np.std(results):.4f}")
    print(f"\nMean Test Accuracy reduced: {np.mean(shap_results):.4f} ± {np.std(shap_results):.4f}")


    plt.imshow(mean_shap_per_pixel, cmap='hot')
    plt.colorbar()
    plt.title("Mean SHAP Values per Pixel")
    plt.savefig("shap_plot.png")
    plt.close()


