import os
import numpy as np
import scipy.io as sio
import pandas as pd

def load_mat_file(file_path):
    """Load EEG data and labels from .mat file."""
    mat_data = sio.loadmat(file_path)
    eeg_data = mat_data['EEG']  # Adjust based on actual key
    labels = mat_data['labels']  # Adjust based on actual key
    return eeg_data, labels

def main(data_dir):
    for file in os.listdir(data_dir):
        if file.endswith(".mat"):
            file_path = os.path.join(data_dir, file)
            mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
            print(type(mat_data))
            print("mat_data  keys: ", mat_data.keys())
            data = mat_data['data']
            expinfo = mat_data.expinfo
            print(type(expinfo))
            df = pd.read_csv(mat_data.expinfo)
            eeg = data.eeg

if __name__ == "__main__":
    data_dir = "DTU/"
    
    file = "DTU/S1.mat"

    mat_data = sio.loadmat(file)
    print(mat_data.keys())
    expinfo = mat_data['expinfoStruct']
    print(type(expinfo))
    print(expinfo.shape)
    print(expinfo.dtype.names)
    print(expinfo['trigger'][0])
    print(expinfo['trigger'][0].item())
    value = expinfo['trigger'][0].item()
    print(expinfo['trigger'][0][0])
    print(value)

    names = expinfo.dtype.names
    first_name = names[0]
    print(first_name)
    
    