import os
import numpy as np
import scipy.io as sio
import pandas as pd

def load_DTU(data_dir):
    labels = []
    eegs = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith("preproc.mat"):
            file_path = os.path.join(data_dir, file_name)
            print(file_path)
            mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
            data = mat_data['data']
            trial_labels = data.event.eeg
            trial_eeg = data.eeg
            
            for label in trial_labels:
                print(label.value)
            
            for eeg in trial_eeg:
                print(eeg)
            

            
            
                
        



if __name__ == "__main__":
    data_dir = "./DTU"
    file_path = "./DTU/S1_data_preproc.mat"
    load_DTU(data_dir)
    mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    data = mat_data['data']
    