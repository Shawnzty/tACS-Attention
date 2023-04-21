import os
from scipy.io import loadmat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# the following import is required for matplotlib < 3.2:
from mpl_toolkits.mplot3d import Axes3D  # noqa
import mne

def mkraw(subject_id, before_or_after):
    # read the EEG data
    eeg_path = os.path.join('..', '..', 'data', str(subject_id), 'eeg_' + before_or_after + '.mat')
    eeg_data = loadmat(eeg_path)['eeg']
    eeg_data[1:33] *= 1e-6 # convert to microvolts

    # get placement data of standard 10-20 system
    montage_1020 = mne.channels.make_standard_montage('standard_1020')
    positions_1020 = montage_1020._get_ch_pos()
    elec_coords_1020 = {ch_name: coord for ch_name, coord in positions_1020.items() if ch_name in montage_1020.ch_names}

    # Define channel names and types
    ch_names = ['Time'] + ['Fp1', 'Fp2', 
                        'AF3', 'AF4', 
                        'F7', 'F3', 'Fz', 'F4', 'F8',
                        'FC1', 'FC2',
                        'T7', 'C3', 'Cz', 'C4', 'T8',
                        'CP5', 'CP1', 'CP2', 'CP6',
                        'P7', 'P5', 'P3', 'Pz', 'P4', 'P6', 'P8',
                        'PO3', 'PO4',
                        'O1', 'Oz', 'O2'] + ['Trigger'] + ['fixation',
                            'endo left', 'endo right', 'exo left', 'exo right',
                            'valid', 'invalid', 'ics fast', 'ics slow',
                            'stim', 'stim_left', 'stim_right', 'stim_close','stim_xmiddle','stim_far',
                            'stim_highest', 'stim_higher', 'stim_ymiddle', 'stim_lower', 'stim_lowest',
                            'response']
    ch_types = ['misc'] + ['eeg'] * 32 + ['misc'] + ['stim'] * 21

    # Create the info object
    info = mne.create_info(ch_names, sfreq=1200, ch_types=ch_types)
    # Create raw object
    raw = mne.io.RawArray(eeg_data, info)

    # manually add the placement of electrodes
    elec_coords = {
        'Fp1': elec_coords_1020['Fp1'],
        'Fp2': elec_coords_1020['Fp2'],
        'AF3': elec_coords_1020['AF3'],
        'AF4': elec_coords_1020['AF4'],
        'F7': elec_coords_1020['F7'],
        'F3': elec_coords_1020['F3'],
        'Fz': elec_coords_1020['Fz'],
        'F4': elec_coords_1020['F4'],
        'F8': elec_coords_1020['F8'],
        'FC1': elec_coords_1020['FC1'],
        'FC2': elec_coords_1020['FC2'],
        'T7': elec_coords_1020['T7'],
        'C3': elec_coords_1020['C3'],
        'Cz': elec_coords_1020['Cz'],
        'C4': elec_coords_1020['C4'],
        'T8': elec_coords_1020['T8'],
        'CP5': elec_coords_1020['CP5'],
        'CP1': elec_coords_1020['CP1'],
        'CP2': elec_coords_1020['CP2'],
        'CP6': elec_coords_1020['CP6'],
        'P7': elec_coords_1020['P7'],
        'P5': elec_coords_1020['P5'],
        'P3': elec_coords_1020['P3'],
        'Pz': elec_coords_1020['Pz'],
        'P4': elec_coords_1020['P4'],
        'P6': elec_coords_1020['P6'],
        'P8': elec_coords_1020['P8'],
        'PO3': elec_coords_1020['PO3'],
        'PO4': elec_coords_1020['PO4'],
        'O1': elec_coords_1020['O1'],
        'Oz': elec_coords_1020['Oz'],
        'O2': elec_coords_1020['O2'],
    }

    # Create the montage object
    montage = mne.channels.make_dig_montage(elec_coords, coord_frame='head')

    # add info and montage to raw object
    raw.set_montage(montage)
    raw.info['subject_info'] = {'id': subject_id}
    raw.filter(l_freq=0.1, h_freq=100)
    # print(raw.info['subject_info'])
    raw_save_path = os.path.join('..', '..', 'data', str(subject_id), 'raw_' + before_or_after + '.fif')
    raw.save(raw_save_path, overwrite=True)

# main
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
for subject_id in range (1,9):
    for before_or_after in ['before', 'after']:
        mkraw(subject_id, before_or_after)