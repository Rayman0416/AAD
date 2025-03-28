import numpy as np
import mne
from scipy.fft import fft, fftfreq
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mne.channels import make_standard_montage

def compute_alpha_power(eeg_data, sfreq):
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
    
    print(alpha_power)
    return alpha_power

def azimuthal_equidistant_projection(channel_positions):
    """
    Convert 3D electrode positions to 2D using azimuthal equidistant projection
    Args:
        channel_positions: Dictionary {channel_name: (x, y, z)}
    Returns:
        projected_2d: Dictionary {channel_name: (x, y)}
    """
    projected_2d = {}
    for ch_name, pos in channel_positions.items():
        x, y, z = pos
        # Convert to spherical coordinates
        theta = np.arctan2(y, x)  # Azimuthal angle
        r = np.sqrt(x**2 + y**2)  # Projected distance from center
        
        # Convert to 2D plane coordinates
        x_2d = r * np.cos(theta)
        y_2d = r * np.sin(theta)
        
        projected_2d[ch_name] = (x_2d, y_2d)
    
    return projected_2d

def create_2d_headmap(alpha_power, channel_names, resolution=100):
    """
    Create 2D headmap from alpha power values
    Args:
        alpha_power: (n_samples, n_channels)
        channel_names: List of channel names
        resolution: Output image resolution
    Returns:
        headmaps: (n_samples, resolution, resolution)
    """
    # Get standard electrode positions
    montage = make_standard_montage('standard_1020')
    ch_pos_3d = montage.get_positions()['ch_pos']
    
    # Project to 2D
    ch_pos_2d = azimuthal_equidistant_projection(ch_pos_3d)
    
    # Prepare interpolation points
    x, y = [], []
    power = []
    for ch_name, p in zip(channel_names, alpha_power.T):
        if ch_name in ch_pos_2d:
            x.append(ch_pos_2d[ch_name][0])
            y.append(ch_pos_2d[ch_name][1])
            power.append(p)
    
    # Create grid
    xi = np.linspace(-1, 1, resolution)
    yi = np.linspace(-1, 1, resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate
    headmap = griddata(
        (x, y),
        power,
        (xi, yi),
        method='cubic',
        fill_value=0
    )
    
    return headmap

def eeg_to_2d_sequence(eeg_data, sfreq, channel_names, window_size=1.0, overlap=0.5):
    """
    Convert EEG data to sequence of 2D headmaps
    Args:
        eeg_data: (n_samples, n_timesteps, n_channels)
        sfreq: Sampling frequency (Hz)
        channel_names: List of channel names
        window_size: Window length in seconds
        overlap: Overlap between windows (0-1)
    Returns:
        headmaps: (n_windows, resolution, resolution)
    """
    n_samples, n_timesteps, n_channels = eeg_data.shape
    window_samples = int(window_size * sfreq)
    step_samples = int(window_samples * (1 - overlap))
    
    headmaps = []
    for start in range(0, n_timesteps - window_samples + 1, step_samples):
        # Extract window
        window = eeg_data[:, start:start + window_samples, :]
        
        # Compute alpha power
        alpha_power = compute_alpha_power(window, sfreq)
        
        # Create 2D headmap for each sample
        for sample_idx in range(n_samples):
            headmap = create_2d_headmap(
                alpha_power[sample_idx:sample_idx+1], 
                channel_names
            )
            headmaps.append(headmap[0])  # Remove batch dimension
    
    return np.array(headmaps)

# Example Usage
if __name__ == "__main__":
    # Parameters
    sfreq = 250  # Hz
    n_samples = 10  # Number of trials
    n_timesteps = 5 * sfreq  # 5 seconds of data
    channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                    'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3',
                    'Pz', 'P4', 'P8', 'O1', 'O2']
    
    # Generate synthetic EEG data
    eeg_data = np.random.randn(n_samples, n_timesteps, len(channel_names)) * 1e-6
    
    # Convert to 2D headmap sequence
    headmaps = eeg_to_2d_sequence(
        eeg_data,
        sfreq,
        channel_names,
        window_size=1.0,  # 1-second windows
        overlap=0.5       # 50% overlap
    )
    
    print(f"Generated headmaps shape: {headmaps.shape}")
    
    plt.imshow(headmaps[0], cmap='jet', origin='lower')
    plt.colorbar(label='Alpha Power (μV²)')
    plt.title("Alpha Power Headmap (8-13 Hz)")

    # Save the current figure
    plt.savefig('alpha_headmap.png')  # Saves as PNG
    plt.close()  # Close the figure to free memory