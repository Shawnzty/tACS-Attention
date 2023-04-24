import os
from scipy.io import loadmat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# the following import is required for matplotlib < 3.2:
from mpl_toolkits.mplot3d import Axes3D  # noqa
import mne

def load_eeg_data(subject_id):
    raw_path_before = os.path.join('../../data', str(subject_id), 'repaired_before.fif')
    raw_before = mne.io.read_raw_fif(raw_path_before, preload=True)
    raw_path_after = os.path.join('../../data', str(subject_id), 'repaired_after.fif')
    raw_after = mne.io.read_raw_fif(raw_path_after, preload=True)
    return raw_before, raw_after

def epoching_data(raw, merge_events, intersect_events, center_event, minus_t, to_t):): 
    # merge_events = [2, 3], intersect_events = [10, 21]

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
    # Create separate events arrays for events to merge (2 and 3) and the event to intersect with (6)
    events_to_merge = events[np.isin(events[:, 2], merge_events)]
    event_to_intersect = events[events[:, 2] == 6]

    # Merge events 2 and 3 by changing their event ids to a new id (e.g., 8)
    merged_events = events_to_merge.copy()
    merged_events[:, 2] = 1

    # Find the intersection between the merged events and event 6
    intersection_samples = np.intersect1d(merged_events[:, 0], event_to_intersect[:, 0])

    # Create a new events array with the intersected events
    intersected_events = merged_events[np.isin(merged_events[:, 0], intersection_samples)]

    # Create epochs with the intersected events
    epochs = mne.Epochs(raw, intersected_events, event_id={'picked_events': 1}, tmin=minus_t, tmax=to_t, preload=True)

    return epochs

def analyze_eeg_data(preprocessed_data):
    # Perform time-frequency analysis and evoked response estimation
    # ...
    return analysis_results
