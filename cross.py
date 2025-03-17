import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
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
                # skip the last 12 trials (experiment 3, repeated segments)
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

# -----------------------------
# Neural Network Model
# -----------------------------
class EEGNet(nn.Module):
    def __init__(self, num_channels, num_timesteps, num_classes):
        super(EEGNet, self).__init__()
        
        # Convolutional Layer to extract spatial and temporal features
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        
        # Activation
        self.relu = nn.ReLU()
        
        # Pooling layer to reduce dimensions
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # Fully connected layer for classification
        self.fc1 = nn.Linear(32 * (num_timesteps // 4) * (num_channels // 4), 64)  # Adjust dimensions after pooling
        self.fc2 = nn.Linear(64, num_classes)
        
        # Dropout for regularization
        # self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(self.relu(self.batchnorm2(self.conv2(x))))
        
        # fully connected layers
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        x = self.relu(self.fc1(x))
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

    # Preprocess the data
    eeg_filtered = bandpass_filter(eeg_signals, lowcut=1.0, highcut=50.0, fs=128)
    num_trials, num_timesteps, num_channels = eeg_filtered.shape
    eeg_reshaped = eeg_filtered.reshape(-1, num_channels)
    scaler = StandardScaler()
    eeg_scaled = scaler.fit_transform(eeg_reshaped)
    eeg_scaled = eeg_scaled.reshape(num_trials, num_timesteps, num_channels)
    segmented_data, segmented_labels = segment_eeg_data(eeg_scaled, labels)
    print(segmented_data.shape, segmented_labels.shape)
    num_channels = segmented_data.shape[2]  # EEG Channels
    num_timesteps = segmented_data.shape[1]  # EEG Time Steps
    print("num_channels", num_channels)
    print("num_timesteps", num_timesteps)

    # Convert to PyTorch dataset
    full_dataset = EEGDataset(segmented_data, segmented_labels)

    # Set up Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_ids, test_ids) in enumerate(loo.split(full_dataset)):
        print(f"Fold {fold + 1}/{len(full_dataset)}")
        print("--------------------------------")

        # Create data loaders for training and testing in this fold
        train_loader = DataLoader(
            full_dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(train_ids)
        )
        test_loader = DataLoader(
            full_dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(test_ids)
        )

        # Initialize the model, loss function, and optimizer
        model = EEGNet(num_channels=num_channels, num_timesteps=num_timesteps, num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        epochs = 20
        for epoch in range(epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Evaluate the model on the left-out sample
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Store the results for this fold
        results.append({
            'fold': fold + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })

    # Print LOOCV results
    print("LEAVE-ONE-OUT CROSS VALIDATION RESULTS")
    print("--------------------------------")
    for result in results:
        print(
            f"Fold {result['fold']}: Train Loss: {result['train_loss']:.4f}, Train Acc: {result['train_acc']:.4f}, "
            f"Test Loss: {result['test_loss']:.4f}, Test Acc: {result['test_acc']:.4f}"
        )

    # Calculate and print average performance across all folds
    avg_train_loss = np.mean([result['train_loss'] for result in results])
    avg_train_acc = np.mean([result['train_acc'] for result in results])
    avg_test_loss = np.mean([result['test_loss'] for result in results])
    avg_test_acc = np.mean([result['test_acc'] for result in results])

    print("\nAverage Performance Across All Folds:")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
    print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")