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
    n_samples, window_length, n_channels = eeg_data.shape
    
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

    # # Compute FFT and get magnitude spectrum
    # fft_vals = np.fft.fft(eeg_data, n=window_length, axis=1)
    # fft_vals = np.abs(fft_vals) / window_length
    
    # freqs = np.fft.fftfreq(window_length, d=1/sfreq)  # Get frequency bins
    # positive_freqs = freqs[:window_length//2]  # Keep only positive frequencies

    # point_low = np.argmax(positive_freqs >= 8)   # First index where freq ≥ 8 Hz
    # point_high = np.argmax(positive_freqs > 13)  # First index where freq > 13 Hz

    # # Compute power in alpha band
    # alpha_power = np.sum(np.power(fft_vals[:, point_low:point_high, :], 2), axis=1)
    
    # print("alpha power shape: ", alpha_power.shape)
    # print(alpha_power)
    # return alpha_power

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

def create_topo_map(channel_values, channel_names, grid_resolution=32):
    """
    Create 2D topological maps using Clough-Tocher interpolation.
    
    Args:
        channel_values: 1D array of alpha power values (n_channels,).
        channel_names: List of channel names matching channel_values.
        grid_resolution: Size of output 2D map (grid_resolution x grid_resolution).
        
    Returns:
        2D topological map (grid_resolution, grid_resolution).
    """
    
    # get electrode positions from the channel names
    montage = mne.channels.make_standard_montage("standard_1020")
    locs_2d = []
    for ch in channel_names:
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
    for i in range(channel_values.shape[0]):  # For each sample in the batch
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
def preprocess(subjects_data, channel_names):
    print("preprocess begin")
    for subject_data in subjects_data:
        for trial in subject_data['trials']:
            eeg_data = trial['eeg']  # EEG data (NumPy array)
            trial['eeg'] = bandpass_filter(eeg_data, lowcut=1.0, highcut=45.0, fs=128)
            trial['eeg'] = segment_eeg_data(trial)
            trial['eeg'] = compute_alpha_power(trial['eeg']) # feature extraction
            trial['eeg'] = normalize_data(trial['eeg'])

            # create topo map for each window
            trial['eeg'] = create_topo_map(trial['eeg'], channel_names)
    
    print("preprocess complete")
    
    window = subjects_data[0]['trials'][0]['eeg'][0]
    save_topo_map_image(window, "./topomap.png")
    window = subjects_data[6]['trials'][6]['eeg'][6]
    save_topo_map_image(window, "./topomap6.png")
    # create 2d topo maps from electrode positions
    return subjects_data

def save_topo_pixel_map(channel_values, channel_names, resolution=32, cmap='viridis', filename='topo_pixel_map.png'):
    montage = mne.channels.make_standard_montage("standard_1020")
    
    # Get 2D projected electrode positions
    locs_2d = []
    for ch in channel_names:
        if ch in montage.ch_names:
            coords = montage.get_positions()["ch_pos"][ch]
            locs_2d.append(azim_proj(coords))
    
    locs_2d = np.array(locs_2d)

    # Normalize coordinates to image grid
    min_x, max_x = locs_2d[:, 0].min(), locs_2d[:, 0].max()
    min_y, max_y = locs_2d[:, 1].min(), locs_2d[:, 1].max()
    
    norm_x = ((locs_2d[:, 0] - min_x) / (max_x - min_x) * (resolution - 1)).astype(int)
    norm_y = ((locs_2d[:, 1] - min_y) / (max_y - min_y) * (resolution - 1)).astype(int)

    # Create a blank image
    image = np.zeros((resolution, resolution))

    channel_values = np.array(channel_values).flatten()

    # Place values in exact pixel positions
    for val, x, y in zip(channel_values, norm_x, norm_y):
        image[y, x] = val  # Note: y is row, x is column

    # Plot and save image
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap=cmap, origin='lower')
    plt.colorbar(label='Channel Value')
    plt.title('2D Channel positions')
    plt.xticks([]); plt.yticks([])
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    return image  # Optional: return for inspection

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

# Compute average SHAP value for each channel's pixel location
def compute_channel_shap_importance(shap_maps, pixel_map):
    channel_scores = defaultdict(list)
    for sample_map in shap_maps: # shape: (1, 32, 32)
        # take absolute value because both positive and negative values are important
        sample_map = np.abs(sample_map[0])  # shape: (32, 32)
        for ch, (i, j) in pixel_map.items():
            channel_scores[ch].append(sample_map[i, j])

    # Compute mean SHAP value for each channel
    channel_mean_shap = {ch: np.mean(vals) for ch, vals in channel_scores.items()}
    return sorted(channel_mean_shap.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    data_KUL = "./KUL"  # KUL data dir
    data_DTU = "./DTU"  # DTU data dir
    
    print("Loading AAD data...")
    try:
        subjects_data, channel_names = load_DTU(data_DTU) 
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

    print(f"Loaded {len(subjects_data)} subjects.")

    subjects_data = preprocess(subjects_data, channel_names)

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
    X = np.array(X)  # Shape: (total_windows, 1, time_steps, channels)
    y = np.array(y)  # labels for each window
    groups = np.array(groups) # subject index for each window
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    print("groups len: ", len(groups))

    # X_train, X_temp, y_train, y_temp = train_test_split(
    #     X, y, test_size=0.2, stratify=y
    # )
    
    # # Second split: 10% validation, 10% test (from the 20% temp)
    # X_val, X_test, y_val, y_test = train_test_split(
    #     X_temp, y_temp, test_size=0.5, stratify=y_temp
    # )
    
    # print(f"Train: {X_train.shape[0]} samples")
    # print(f"Validation: {X_val.shape[0]} samples")
    # print(f"Test: {X_test.shape[0]} samples")
    
    # # Convert to PyTorch datasets
    # train_dataset = EEGDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    # val_dataset = EEGDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    # test_dataset = EEGDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    # # Create dataloaders
    # batch_size = 32
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # # Initialize model, loss, optimizer
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = EEGNet2().to(device)
    # criterion = nn.BCEWithLogitsLoss().to(device)  # This loss expects raw logits
    # optimizer = optim.RMSprop(model.parameters(), lr=0.0003, weight_decay=3e-4)
    
    # # Training loop for 10 epochs
    # num_epochs = 50
    # best_val_acc = 0
    # best_model = None
    
    # for epoch in range(num_epochs):
    #     train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    #     val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    #     print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
    #     # Test (only for best model)
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         best_model = model.state_dict()

    # # Calculate test accuracy
    # model.load_state_dict(best_model)
    # test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    # print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # background, _ = next(iter(train_loader))
    # background = background[:50].to(device)

    # test_samples, _ = next(iter(test_loader))
    # test_samples = test_samples[:10].to(device)  # Use a small batch for explanation

    # explainer = shap.DeepExplainer(model, background)
    # shap_values = explainer.shap_values(test_samples)
    # shap.image_plot(shap_values, test_samples.cpu().numpy())

    # plt.savefig("shap_plot.png")
    # plt.close()

    # leave one subject out cross-validation
    logo = LeaveOneGroupOut()
    epochs = 20
    best_val_acc = 0
    best_model = None
    results = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, stratify=y_test
        )

        print(f"Test subject(fold): {set(groups[i] for i in test_idx)}")

        train_dataset = EEGDataset(X_train, y_train)
        test_dataset = EEGDataset(X_test, y_test)
        val_dataset = EEGDataset(X_val, y_val)
        num_channels = X_train.shape[2]
        num_timesteps = X_train.shape[1]

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_timesteps, num_channels = X_train.shape[1], X_train.shape[2]
        model = Rayanet().to(device)

        # assign weights to classes to counter class imbalance
        class_counts = np.bincount(y_train)
        class_weights = 1. / class_counts
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EEGNet2().to(device)
        criterion = nn.BCEWithLogitsLoss().to(device)  # This loss expects raw logits
        optimizer = optim.RMSprop(model.parameters(), lr=0.0003, weight_decay=3e-4)

        # Training loop
        for epoch in range(epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Test (only for best model)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model.state_dict()

        # Evaluation
        model.load_state_dict(best_model)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        results.append(test_acc)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    print(f"\nMean Test Accuracy: {np.mean(results):.4f} ± {np.std(results):.4f}")


