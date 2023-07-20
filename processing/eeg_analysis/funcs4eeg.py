import os
from scipy.io import loadmat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# the following import is required for matplotlib < 3.2:
from mpl_toolkits.mplot3d import Axes3D  # noqa
import mne
import re
import scipy.signal



def load_eeg(subject_id):
    eeg_path_before = os.path.join('..', '..', '..','data', str(subject_id), 'repaired_before_raw.fif')
    eeg_path_after = os.path.join('..', '..', '..','data', str(subject_id), 'repaired_after_raw.fif')
    eeg_before = mne.io.read_raw_fif(eeg_path_before, preload=True, verbose=False)
    eeg_after = mne.io.read_raw_fif(eeg_path_after, preload=True, verbose=False)
    # print(eeg.info['subject_info'],eeg.info['experimenter'])  
    return eeg_before, eeg_after


# Function to calculate the PSD for each subject
def calculate_subject_psd(subject_ids, before_or_after, fmin, fmax):
    psds = []
    for subject_id in subject_ids:
        epochs = load_and_epoching(subject_id, before_or_after, 0, 1)
        psd, freqs = mne.time_frequency.psd_multitaper(epochs, fmin=fmin, fmax=fmax, n_jobs=1)
        psds.append(psd)
    return psds


def condition(condition_str):
    # Replace logical operators with Python syntax and add necessary syntax for condition
    condition_str = re.sub(r"(\b\d+\b)", r"any(piece[:, 2] == \1)", condition_str)
    condition_str = condition_str.replace("AND", "&").replace("OR", "|")
    
    # Form the Python statement
    python_statement = f"[piece for piece in pieces if {condition_str}]"
    
    return python_statement


def make_default_events(eeg):
    # Extract channel names and types
    ch_names = eeg.info['ch_names']
    ch_types = ['misc'] + ['eeg'] * 32 + ['misc'] + ['stim'] * 22

    # Detect events
    stim_channel_names = [ch_name for ch_name, ch_type in zip(ch_names, ch_types) if ch_type == 'stim']
    events = np.array([], dtype=int).reshape(0, 3)  # Create an empty events array with 3 columns

    for idx, stim_channel_name in enumerate(stim_channel_names):
        single_event = mne.find_events(eeg, stim_channel=stim_channel_name, min_duration=1/eeg.info['sfreq'], verbose=False)

        # Update the event id in single_event (the third column) to be idx + 1
        single_event[:, 2] = idx + 1

        # Concatenate single_event to the events array
        events = np.vstack([events, single_event])

    event_dict = {str(idx+1)+" "+stim_channel_name: idx + 1 for idx, stim_channel_name in enumerate(stim_channel_names)}
    
    return events, event_dict


def make_custom_events(eeg, events, event_dict, behav_trials, case):
    # for example: case = '3 | 4'; for all trials case = '1
    # Get the indices that would sort the first column
    sort_indices = np.argsort(events[:, 0])

    # Use these indices to sort the entire array
    sorted_events = events[sort_indices]

    # Make pieces
    pieces = []
    piece = []
    for event in sorted_events:
        if event[2] == 1 and piece:  # Found a 1 in the third column and there are events in current piece
            pieces.append(np.array(piece))  # Save the current piece as a numpy array
            piece = []  # Create a new piece
        piece.append(event)  # Add the current event to the piece

    # Add the last piece if it's not empty
    if piece:
        pieces.append(np.array(piece))
    
     # Subtract 1 from behav_trials since Python uses 0-based indexing
    behav_trials = [trial - 1 for trial in behav_trials]
    # Select pieces that exist (are within the range of the pieces list)
    pieces = [pieces[i] for i in behav_trials if i < len(pieces)]

    the_pieces = eval(condition(case))
    picked_events = np.vstack(the_pieces)
    picked_events_dict = {key: value for key, value in event_dict.items() if value in picked_events[:, 2]}

    # fig = mne.viz.plot_events(picked_events, event_id=picked_events_dict, sfreq=eeg.info['sfreq'], first_samp=eeg.first_samp)

    return picked_events, picked_events_dict



def make_epochs(eeg, events, event_dict, watch, tmin, tmax):
    # for example: watch = '11 stim'
    epochs = mne.Epochs(eeg, events, event_id=event_dict[watch],
                           tmin=tmin, tmax=tmax, baseline=(0,0), preload=True, verbose=False)
    return epochs


def load_behavior(subject_id):
    behavior_before_path = os.path.join('..', '..', '..', 'data', str(subject_id), 'behavior_before.csv')
    behavior_before = pd.read_csv(behavior_before_path)
    behavior_after_path = os.path.join('..', '..', '..', 'data', str(subject_id), 'behavior_after.csv')
    behavior_after = pd.read_csv(behavior_after_path)

    return behavior_before, behavior_after


def remove_outlier(df, k=1.5):
    # Assume df is your DataFrame and 'reaction time' is the column you are interested in
    Q1 = df['reaction time'].quantile(0.25)
    Q3 = df['reaction time'].quantile(0.75)
    IQR = Q3 - Q1

    # Only keep rows in dataframe that have 'reaction time' within Q1 - 1.5 IQR and Q3 + 1.5 IQR
    filtered_df = df[~((df['reaction time'] < (Q1 - k * IQR)) |(df['reaction time'] > (Q3 + k * IQR)))]
    # print('Removed outliers: ' + str(len(df) - len(filtered_df)))
    return filtered_df


def find_trials(behavior):
    respond_trials = behavior[(behavior['response'] == 1) & (behavior['reaction time'] > 0.001)]
    respond_trials = remove_outlier(respond_trials)
    return respond_trials['trial'].tolist()


def inuse_trials(subject_id):
    behavior_before, behavior_after = load_behavior(subject_id)
    trials_before = find_trials(behavior_before)
    trials_after = find_trials(behavior_after)

    return trials_before, trials_after


def power_psd(psd, freqs, fmin, fmax):
    # mean power over epochs and channels
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    int_power = psd[:, :, freq_mask].sum(axis=2)
    median_power_epochs = np.median(int_power, axis=0)
    max_power = np.max(median_power_epochs)
    return max_power


def pipeline_band_power(subject_id, case, watch, fmin, fmax, tmin, tmax):
    eeg_before, eeg_after = load_eeg(subject_id)
    trials_before, trials_after = inuse_trials(subject_id)

    events, event_dict = make_default_events(eeg_before)
    picked_events, picked_events_dict = make_custom_events(eeg_before, events, event_dict, trials_before, case)
    epochs_before = make_epochs(eeg_before, picked_events, picked_events_dict, watch, tmin=tmin, tmax=tmax)
    # psd (n_epochs, n_channels, n_frequencies)
    psd, freqs = mne.time_frequency.psd_multitaper(epochs_before, fmin=fmin, fmax=fmax, n_jobs=1, verbose=False)
    power_before = power_psd(psd, freqs, fmin, fmax)

    events, event_dict = make_default_events(eeg_after)
    picked_events, picked_events_dict = make_custom_events(eeg_after, events, event_dict, trials_after, case)
    epochs_after = make_epochs(eeg_after, picked_events, picked_events_dict, watch, tmin=tmin, tmax=tmax)
    psd, freqs = mne.time_frequency.psd_multitaper(epochs_after, fmin=fmin, fmax=fmax, n_jobs=1, verbose=False)
    power_after = power_psd(psd, freqs, fmin, fmax)

    return power_before, power_after


def get_evoked_response(epochs):
    evoked = epochs.get_data()
    evoked = evoked[:,1:33,:]
    evoked = np.median(evoked, axis=0) # median
    evoked = np.median(evoked, axis=0) # median
    return evoked


def pipeline_evoked_response(subject_id, case, watch, tmin, tmax):
    eeg_before, eeg_after = load_eeg(subject_id)
    trials_before, trials_after = inuse_trials(subject_id)

    events, event_dict = make_default_events(eeg_before)
    picked_events, picked_events_dict = make_custom_events(eeg_before, events, event_dict, trials_before, case)
    epochs_before = make_epochs(eeg_before, picked_events, picked_events_dict, watch, tmin=tmin, tmax=tmax)
    evoked_before = get_evoked_response(epochs_before)

    events, event_dict = make_default_events(eeg_after)
    picked_events, picked_events_dict = make_custom_events(eeg_after, events, event_dict, trials_after, case)
    epochs_after = make_epochs(eeg_after, picked_events, picked_events_dict, watch, tmin=tmin, tmax=tmax)
    evoked_after = get_evoked_response(epochs_after)

    return evoked_before, evoked_after


def trimmed_mean_std(arr, axis=0):
    # Sort the array along the specified axis
    sorted_arr = np.sort(arr, axis=axis)

    # Remove the smallest and largest values
    trimmed_arr = sorted_arr[1:-1] if axis == 0 else sorted_arr[:, 1:-1]

    # Calculate the mean and standard deviation
    trimmed_mean = np.mean(trimmed_arr, axis=axis)
    trimmed_std = np.std(trimmed_arr, axis=axis)

    return trimmed_mean, trimmed_std


def low_pass_filter(data, sfreq, cutoff=50, order=5):
    nyquist = 0.5 * sfreq
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = scipy.signal.filtfilt(b, a, data)
    return y