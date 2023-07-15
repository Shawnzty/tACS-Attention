import os
from scipy.io import loadmat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# the following import is required for matplotlib < 3.2:
from mpl_toolkits.mplot3d import Axes3D  # noqa
import mne


def load_and_epoching(subject_id, before_or_after, minus_t, to_t):
    raw_path = os.path.join('../../data', str(subject_id), 'repaired_'+before_or_after+'.fif')
    raw = mne.io.read_raw_fif(raw_path, preload=True)

    # detect events
    ch_names = raw.info['ch_names']
    ch_types = ['misc'] + ['eeg'] * 32 + ['misc'] + ['stim'] * 21
    stim_channel_names = [ch_name for ch_name, ch_type in zip(ch_names, ch_types) if ch_type == 'stim']
    events = np.array([], dtype=int).reshape(0, 3)  # Create an empty events array with 3 columns

    for idx, stim_channel_name in enumerate(stim_channel_names):
        single_event = mne.find_events(raw, stim_channel=stim_channel_name, min_duration=1/raw.info['sfreq'])
        # Update the event id in single_event (the third column) to be idx + 1
        single_event[:, 2] = idx + 1
        # Concatenate single_event to the events array
        events = np.vstack([events, single_event])

    event_dict = {stim_channel_name: idx + 1 for idx, stim_channel_name in enumerate(stim_channel_names)}

    # epoching
    epochs = mne.Epochs(raw, events, event_id=10, tmin=minus_t, tmax=to_t, preload=True)
    return epochs


# Function to calculate the PSD for each subject
def calculate_subject_psd(subject_ids, before_or_after, fmin, fmax):
    psds = []
    for subject_id in subject_ids:
        epochs = load_and_epoching(subject_id, before_or_after, 0, 1)
        psd, freqs = mne.time_frequency.psd_multitaper(epochs, fmin=fmin, fmax=fmax, n_jobs=1)
        psds.append(psd)
    return psds
