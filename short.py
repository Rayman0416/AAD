import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------
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

def load_kuleuven_aad_data(data_dir):
    """
    Load EEG data and labels from the KULeuven AAD dataset.

    Args:
        data_dir (str): Path to the dataset directory containing .mat files.

    Returns:
        eeg_signals (np.ndarray): EEG signals of shape (samples, features).
        labels (np.ndarray): Corresponding labels (0 for 'L', 1 for 'R').
    """
    eeg_signals, labels = [], []

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

            for trial in trials:
                if trial.TrialID == 9:
                    break

                label_char = getattr(trial, 'attended_ear', None)
                raw_data = getattr(trial, 'RawData', None)

                if label_char not in ('L', 'R') or raw_data is None:
                    print(f"Skipping trial: Missing 'attended_ear' or 'RawData'.")
                    continue

                eeg_data = getattr(raw_data, 'EegData', None)
                if eeg_data is None or not isinstance(eeg_data, np.ndarray) or eeg_data.ndim != 2:
                    print(f"Skipping trial: Invalid 'EegData' format.")
                    continue

                eeg_signals.append(eeg_data)
                labels.append(1 if label_char == 'R' else 0)

    if not eeg_signals:
        raise ValueError("No valid EEG data found.")

    # Find the minimum sequence length across all trials
    # min_length = min(signal.shape[0] for signal in eeg_signals)
    min_length = (6*60)*128 # 6 minutes truncate length

    # Truncate all EEG signals to the minimum length
    eeg_signals_truncated = np.array([signal[:min_length] for signal in eeg_signals], dtype=np.float32)

    # Convert labels to a NumPy array
    labels = np.array(labels, dtype=np.int64)

    return eeg_signals_truncated, labels


def segment_eeg_data(eeg_data, labels, window_size=128, overlap=0.5):
    """
    Segments EEG data with overlap.
    
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

    for trial_idx in range(eeg_data.shape[0]):
        trial_data = eeg_data[trial_idx]  # Extract one trial
        trial_label = labels[trial_idx]   # Extract label for this trial

        # Create sliding windows
        for start in range(0, trial_data.shape[0] - window_size + 1, step_size):
            end = start + window_size
            segment = trial_data[start:end, :]
            
            segmented_data.append(segment)
            segmented_labels.append(trial_label)  # Assign label (update this logic if labels vary)

    return np.array(segmented_data), np.array(segmented_labels)

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

def preprocess_data(eeg_signals, labels):
    """Preprocess EEG signals: apply bandpass filter, normalize globally, segment, and split."""

    # Apply bandpass filter (1-50 Hz)
    eeg_filtered = bandpass_filter(eeg_signals, lowcut=1.0, highcut=50.0, fs=128)

    # Reshape for fitting StandardScaler (combine trials and time)
    num_trials, num_timesteps, num_channels = eeg_filtered.shape
    eeg_reshaped = eeg_filtered.reshape(-1, num_channels)  # Shape: (num_trials * num_timesteps, num_channels)

    # Fit scaler on the entire training set
    scaler = StandardScaler()
    eeg_scaled = scaler.fit_transform(eeg_reshaped)

    # Reshape back to (num_trials, num_timesteps, num_channels)
    eeg_scaled = eeg_scaled.reshape(num_trials, num_timesteps, num_channels)
    print(eeg_scaled.shape)

    # Segment EEG data into smaller windows
    segmented_data, segmented_labels = segment_eeg_data(eeg_scaled, labels)
    print(segmented_data[0].shape)
    print(segmented_data[0])
    print(segmented_labels.shape)
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        segmented_data, segmented_labels, test_size=0.2, stratify=segmented_labels
    )

    # Convert to PyTorch datasets
    return EEGDataset(X_train, y_train), EEGDataset(X_test, y_test)

class EEGNet(nn.Module):
    def __init__(self, num_classes=2, num_channels=64, input_size=128):
        """
        EEGNet-inspired CNN for EEG classification.

        Args:
        - num_classes (int): Number of output classes.
        - num_channels (int): Number of EEG channels (default: 64).
        - input_size (int): Number of time steps per segment.
        """
        super(EEGNet, self).__init__()

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

    def forward(self, x):
        x = x.unsqueeze(1) # change shape -> (batch_size, 1, num_timesteps, num_channels)

        # convolutional layers
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))

        # fully connected layers
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.fc(x))
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

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    data_dir = "./KUL"  # Update with actual dataset path

    print("Loading KULeuven AAD data...")
    try:
        eeg_signals, labels = load_kuleuven_aad_data(data_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

    print(f"Loaded {len(labels)} trials with shape {eeg_signals.shape}")

    
    train_dataset, test_dataset = preprocess_data(eeg_signals, labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_channels = train_dataset[0][0].shape[1]  # EEG Channels
    num_timesteps = train_dataset[0][0].shape[0]  # EEG Time Steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNet(num_classes=2, num_channels=64, input_size=128).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )
