import os
import numpy as np
import scipy.io as sio


# KUL dataset Load in first 8 trials of eeg data for each subject and extract 2D electrode coordinates
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

# load in DTU dataset
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