import os
import torch
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

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
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith("preproc128.mat"):
            subject_trials = []
            file_path = os.path.join(data_dir, file_name)
            mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)

            data = mat_data['data']
            trial_labels = data.event.eeg
            trial_eeg = data.eeg
            
            for eeg, label in zip(trial_eeg, trial_labels):
                eeg = eeg[:, :-2] # drop the last 2 (mastoid) channels
                subject_trials.append({
                    'eeg': eeg,
                    'label': 1 if label.value == 2 else 0
                })
        
            if subject_trials:
                subjects_data.append({'name': file_name, 'trials': subject_trials})
                
    return subjects_data

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
def preprocess(subjects):
    for subject in subjects:
        # Iterate over each trial in the subject
        for trial in subject['trials']:
            eeg_data = trial['eeg']  # EEG data (NumPy array)
            trial['eeg'] = bandpass_filter(eeg_data, lowcut=1.0, highcut=45.0, fs=128)
            trial['eeg'] = segment_eeg_data(trial)
            trial['eeg'] = normalize_data(trial['eeg'])

    print(subjects_data[0]['trials'][0]['eeg'][0])
    return subjects
            
# -----------------------------
# Neural Network Models
# -----------------------------
# 2 convolutional layer
class EEGNet(nn.Module):
    def __init__(self, num_classes=2):
        super(EEGNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 20, 128),  # Adjust input dim based on conv output
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes))
    
    def forward(self, x):
        x = x.unsqueeze(1) # change shape -> (batch_size, 1, num_timesteps, num_channels)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
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

    for i, subject in enumerate(subjects_data):
        for trial in subject['trials']:
            eeg_windows = trial['eeg']  # Shape: (windows, time_steps, channels)
            num_windows = eeg_windows.shape[0]  # Get number of windows

            # Reshape each window to (1, time_steps, channels) and add to X
            for window in eeg_windows:
                X.append(window)  # Add a new axis for channel dim

            # Repeat the label for each window
            y.extend([trial['label']] * num_windows)

            # Repeat the subject index for each window
            groups.extend([i] * num_windows)

    # Convert to NumPy arrays
    X = np.array(X)  # Shape: (total_windows, 1, time_steps, channels)
    y = np.array(y)  # labels for each window
    groups = np.array(groups) # subject index for each window
    print(X.shape, y.shape, len(groups))

    logo = LeaveOneGroupOut()
    epochs = 10
    results = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"Test subject(fold): {set(groups[i] for i in test_idx)}")
        print(X_train.shape, X_test.shape)
        print(np.bincount(y_train))

        train_dataset = EEGDataset(X_train, y_train)
        test_dataset = EEGDataset(X_test, y_test)
        num_channels = X_train.shape[2]
        num_timesteps = X_train.shape[1]

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = EEGNet(num_channels=num_channels, num_timesteps=num_timesteps, num_classes=2).to(device)
        model = EEGNet(num_classes=2).to(device)

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

    
    