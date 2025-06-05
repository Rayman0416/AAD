import os
import mne
import math
import scipy.io as sio
import numpy as np
import shap
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from preprocess import *
from model import *
    
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
    for subject in subjects_data:
        print(f"Subject {subject['name']}")

    return subjects_data, channel_names

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

def preprocess(data, labels, channel_names, reduced_channels=None):
    data = compute_alpha_power(data)
    data = normalize_data(data)
    if reduced_channels is not None:
        data = create_topo_map(data, channel_names, reduced_channels=reduced_channels)
    else:
        data = create_topo_map(data, channel_names)

    # data, labels = create_stacked_input(data, labels, stack_size=3)
    
    return data, labels
    
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
    top_channel_indices = [channel_names.index(name) for name in top_channels]
    print(f"Top {reduction} channels: {[channel_names[i] for i in top_channel_indices]}")


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
    epochs = 100
    results = []
    all_shap_values = []
    for fold in range(folds):
        print(f"Fold: {fold + 1}")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=1 + fold, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=1 + fold, stratify=y_train
        )

        # Preprocess the data
        X_train, y_train = preprocess(X_train, y_train, channel_names)
        X_val, y_val = preprocess(X_val, y_val, channel_names)
        X_test, y_test = preprocess(X_test, y_test, channel_names)
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # Create DataLoader
        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        test_dataset = EEGDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Rayanet(in_channels=1).to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=3e-4)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        # Training loop
        best_val_loss = float('inf')
        best_model = None
        patience = 20
        counter = 0
        for epoch in range(epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping...")
                break

            # scheduler.step(val_loss)
        
        model.load_state_dict(best_model)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        results.append(test_acc)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # calculate SHAP values
        # SHAP on TRAINING SET ONLY to avoid leakage
        model.eval()

        # Use a small subset of training data as background
        background = torch.from_numpy(X_train[:200]).float().to(device)
        explainer = shap.DeepExplainer(model, background)

        # Compute SHAP on the rest of the training set (or all of it)
        X_test_tensor = torch.from_numpy(X_test[:100]).float().to(device)
        shap_values_fold = explainer.shap_values(X_test_tensor)  # shape: (batch, time, channels)
        shap_values_fold = shap_values_fold.squeeze()  # remove singleton dims if needed

        all_shap_values.append(shap_values_fold)

    print(f"\nMean Test Accuracy: {np.mean(results):.4f} ± {np.std(results):.4f}")

    # Concatenate across folds
    final_shap_values = np.concatenate(all_shap_values, axis=0)
    print(f"concat SHAP shape: {final_shap_values.shape}")

    # Average absolute SHAP across all samples
    mean_shap_per_pixel = np.mean(np.abs(final_shap_values), axis=0)
    print(f"mean SHAP shape: {mean_shap_per_pixel.shape}")

    
    
    # Run reduced-channel 32, 16 on the model
    reduced_results = {}
    reduction_list = [32, 16]
    for reduction in reduction_list:
        print(f"\nRunning reduced-channel experiment: top-{reduction} channels")

        X, reduced_channels = reduce_channels(original_values, mean_shap_per_pixel, channel_names, reduction=reduction)
        print(f"X shape after reduction ({reduction}): {X.shape}")

        shap_results = []

        for fold in range(folds):
            best_val_loss = float('inf')
            best_model = None
            print(f"Fold: {fold + 1}")

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=1 + fold, stratify=y
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=1 + fold, stratify=y_train
            )

            # Preprocess the data
            X_train, y_train = preprocess(X_train, y_train, channel_names, reduced_channels=reduced_channels)
            X_val, y_val = preprocess(X_val, y_val, channel_names, reduced_channels=reduced_channels)
            X_test, y_test = preprocess(X_test, y_test, channel_names, reduced_channels=reduced_channels)

            # Create DataLoader
            train_dataset = EEGDataset(X_train, y_train)
            val_dataset = EEGDataset(X_val, y_val)
            test_dataset = EEGDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)

            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Rayanet(in_channels=1).to(device)

            # Loss and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=3e-4)
            # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

            # Training loop
            best_val_loss = float('inf')
            best_model = None
            patience = 20
            counter = 0
            for epoch in range(epochs):
                train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict()
                    counter = 0
                else:
                    counter += 1

                if counter >= patience:
                    print("Early stopping...")
                    break

                # scheduler.step(val_loss)
            
            model.load_state_dict(best_model)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            shap_results.append(test_acc)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        reduced_results[reduction] = shap_results
    
    print(f"\nMean Test Accuracy: {np.mean(results):.4f} ± {np.std(results):.4f}")
    print(f"Mean Test Accuracy reduced 32: {np.mean(reduced_results[32]):.4f} ± {np.std(reduced_results[32]):.4f}")
    print(f"Mean Test Accuracy reduced 16: {np.mean(reduced_results[16]):.4f} ± {np.std(reduced_results[16]):.4f}")
    
    results_dict = {
        "subject": [subject_nr],
        "mean_accuracy": [np.mean(results)],
        "std_accuracy": [np.std(results)],
    }

    # Add dynamic reductions
    for reduction in reduction_list:
        results_dict[f"mean_accuracy_{reduction}"] = [np.mean(reduced_results[reduction])]
        results_dict[f"std_accuracy_{reduction}"] = [np.std(reduced_results[reduction])]

    # Create a DataFrame
    df = pd.DataFrame(results_dict)

    # Write or append to CSV
    output_file = "accuracy_results.csv"
    try:
        existing_df = pd.read_csv(output_file)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_csv(output_file, index=False, float_format="%.4f")
    except FileNotFoundError:
        df.to_csv(output_file, index=False, float_format="%.4f")

    print(f"\nResults saved to {output_file}")

    # plt.imshow(mean_shap_per_pixel, cmap='hot')
    # plt.colorbar()
    # plt.title("Mean SHAP Values per Pixel")
    # plt.savefig("shap_plot.png")
    # plt.close()


