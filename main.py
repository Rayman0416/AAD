import mne
import scipy.io as sio
import numpy as np
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from preprocess import *
from model import *
from load import *
import pandas as pd


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
    

def reduce_channels(original_values, shap_values, channel_names, reduction):
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

def run_subject_analysis(dataset: str, subject_nr: int, data_KUL="./KUL", data_DTU="./DTU", reduction_list=[32], save_plot=False):
    dataset = dataset.strip().upper()

    # Load the dataset
    if dataset == "DTU":
        print("Loading DTU AAD data...")
        try:
            subjects_data, channel_names = load_DTU(data_DTU)
        except Exception as e:
            raise RuntimeError(f"Failed to load DTU data: {e}")
    elif dataset == "KUL":
        print("Loading KUL AAD data...")
        try:
            subjects_data, channel_names = load_kul(data_KUL)
        except Exception as e:
            raise RuntimeError(f"Failed to load KUL data: {e}")
    else:
        raise ValueError("Invalid dataset. Must be 'KUL' or 'DTU'.")

    print(f"Loaded {len(subjects_data)} subjects from {dataset} dataset.")
    subject = subjects_data[int(subject_nr) - 1]
    print(f"Processing subject {subject['name']}...")

    subject = window_split(subject)
    print("subject shape: ", subject['trials'][0]['eeg'].shape)

    # Prepare data
    if dataset == "DTU":
        X, y, groups = group_DTU_trials(subject)
    elif dataset == "KUL":
        X, y, groups = group_KUL_trials(subject)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    print(f"X shape: {X.shape}, y shape: {y.shape}, groups shape: {groups.shape}")
    original_values = X.copy()

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

    
    
    # Run all reduction variations in reduction_list
    reduced_results = {}
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
    
    return {
        "subject": subject_nr,
        "dataset": dataset,
        "mean_acc": round(np.mean(results), 4),
        "std_acc": round(np.std(results), 4),
        "mean_shap_acc": round(np.mean(shap_results), 4),
        "std_shap_acc": round(np.std(shap_results), 4),
    }




