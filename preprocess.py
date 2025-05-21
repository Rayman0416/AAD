import numpy as np
import math
import mne
from utils import *
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
from scipy.signal import butter, filtfilt

# Bandpass filter the data and segment the data into windows
# extract the alpha power for each channel in each window and apply channel-wise normalization
# create a topographical map for each window
def window_split(subject):
    print("window split begin")
    for trial in subject['trials']:
        eeg_data = trial['eeg']  # EEG data (NumPy array)
        trial['eeg'] = bandpass_filter(eeg_data, lowcut=1.0, highcut=13.0, fs=128)
        trial['eeg'] = segment_eeg_data(trial)
    
    print("window split complete")
    return subject

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

# z-score channel-wise normalization
def normalize_data(eeg_data):
    mean = np.mean(eeg_data, axis=0)
    std = np.std(eeg_data, axis=0)
    return (eeg_data - mean) / std

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
    
    alpha_low = 8
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
    for i in range(images.shape[0]):
        valid = ~np.isnan(images[i])
        images[i][valid] = scale(images[i][valid])


    # Replace NaNs with zero or another fill value
    images = np.nan_to_num(images, nan=0)

    # Apply circular mask to the images
    # images = apply_circular_mask(images)

    return images