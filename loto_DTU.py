import os
import torch
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from collections import defaultdict

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

def load_DTU(data_dir):
    subjects_data = []
    
    # read out each subject file (18 files)
    for file_name in os.listdir(data_dir):
        if file_name.endswith("preproc128.mat"):
            subject_trials = []
            file_path = os.path.join(data_dir, file_name)
            mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)

            data = mat_data['data']
            trials_labels = data.event.eeg # 60 labels
            trials_eeg = data.eeg # 60 trials 
            
            # for each trial append the eeg data and corresponding label
            # also drop the last 2 (mastoid) channels to get 64 channels
            for eeg, label in zip(trials_eeg, trials_labels):
                eeg = eeg[:, :-2] # drop the last 2 (mastoid) channels
                subject_trials.append({
                    'eeg': eeg,
                    'label': 1 if label.value == 2 else 0
                })
        
            # append the trials data and labels per subject
            if subject_trials:
                subjects_data.append({'name': file_name, 'trials': subject_trials})
                
    return subjects_data

# segments trial data into windows of window_size and overlap ratio
def segment_eeg_data(trial, window_size=128, overlap=0.5):
    step_size = int(window_size * (1 - overlap))  # Calculate step size for sliding window
    segmented_data = []
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

# z-score channel-wise normalization
def normalize_data(eeg_data):
    mean = np.mean(eeg_data, axis=0)
    std = np.std(eeg_data, axis=0)
    return (eeg_data - mean) / std

# Preprocess the data
def preprocess(subjects_data):
    for subject_data in subjects_data:
        for trial in subject_data['trials']:
            eeg_data = trial['eeg']  # EEG data (NumPy array)
            trial['eeg'] = bandpass_filter(eeg_data, lowcut=1.0, highcut=45.0, fs=128)
            trial['eeg'] = segment_eeg_data(trial)
            trial['eeg'] = normalize_data(trial['eeg'])
    
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
            in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1
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

        self.dropout = nn.Dropout(0.3)  

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
    data_dir = "./DTU"

    print("Loading DTU AAD data...")
    try:
        subjects_data = load_DTU(data_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

    print(f"Loaded {len(subjects_data)} subjects.")

    subjects_data = preprocess(subjects_data)
    
    # Prepare data for cross-validation
    X, y, groups = [], [], []

    # Group trials by trial_counter
    trial_groups = defaultdict(list)

    for subject in subjects_data:
        trial_counter = 0
        for trial in subject['trials']:
            trial_groups[trial_counter].append(trial)
            trial_counter += 1

    # Create subsets of 5 trials each
    subset_size = 5
    trial_counters = sorted(trial_groups.keys())  # Ensure trials are sorted
    trial_subsets = []

    for i in range(0, len(trial_counters), subset_size):
        subset = []
        for j in range(i, min(i + subset_size, len(trial_counters))):
            subset.extend(trial_groups[trial_counters[j]])
        trial_subsets.append(subset)

    # Process subsets to extract X, y, and groups
    subset_id = 0  # Unique identifier for each subset
    for subset in trial_subsets:
        for trial in subset:
            eeg_windows = trial['eeg']  # Shape: (windows, time_steps, channels)
            trial_label = trial['label']
            
            for window in eeg_windows:
                X.append(window)       # Add EEG window (shape: time_steps, channels)
                y.append(trial_label)  # Assign label
                groups.append(subset_id)  # Assign all windows to the subset group

        subset_id += 1  # Increment subset ID for next group

    # Convert to NumPy arrays
    X = np.array(X)  # Shape: (windows, time_steps, channels)
    y = np.array(y)  # Shape: (windows,)
    groups = np.array(groups)  # Shape: (windows,)
    print(X.shape, y.shape, len(groups))


    # leave one trial out cross validation
    logo = LeaveOneGroupOut()
    epochs = 20

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"Test subject(fold): {set(groups[i] for i in test_idx)}")

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
        )

        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        test_dataset = EEGDataset(X_test, y_test)
        num_channels = X_train.shape[2]
        num_timesteps = X_train.shape[1]

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()  # Add before model instantiation if using GPU
        num_timesteps, num_channels = X_train.shape[1], X_train.shape[2]
        model = EEGNet2(num_classes=2, num_channels=num_channels, input_size=num_timesteps).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0003)

        # Training loop
        for epoch in range(epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Evaluation
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    
    