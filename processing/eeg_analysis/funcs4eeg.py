import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy import signal
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# the following import is required for matplotlib < 3.2:
from mpl_toolkits.mplot3d import Axes3D  # noqa
import mne
import re
import scipy.signal
from scipy.signal import welch
import behavior.func4behav as fb
import imp
from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test
import fathon
from fathon import fathonUtils as fu
from scipy.signal import butter, lfilter

imp.reload(fb)

relative_path = os.path.join('..', '..', '..', '..', 'data')

def load_eeg(subject_id):
    eeg_path_before = os.path.join(relative_path, str(subject_id), 'repaired_before_raw.fif')
    eeg_path_after = os.path.join(relative_path, str(subject_id), 'repaired_after_raw.fif')
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

    the_pieces = eval(condition(case)) # if behav_trials hasn't considered case
    picked_events = np.vstack(the_pieces)
    # picked_events = np.vstack(pieces)
    picked_events_dict = {key: value for key, value in event_dict.items() if value in picked_events[:, 2]}
    return picked_events, picked_events_dict


def make_epochs(eeg, events, event_dict, watch, baseline, tmin, tmax, detrend):
    # for example: watch = '11 stim'
    epochs = mne.Epochs(eeg, events, event_id=event_dict[watch],
                           tmin=tmin, tmax=tmax, baseline=baseline, preload=True, verbose=False, detrend=detrend) ## detrend!!
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
    evoked = epochs.get_data(copy=False) # FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning
    evoked = evoked[:,1:33,:]
    # evoked = np.median(evoked, axis=0) # median
    # evoked = np.median(evoked, axis=0) # median
    return evoked


def pipeline_evoked_response_EMBC(subject_id, case, watch, tmin, tmax):
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


def trimmed_mean_std(data, axis=0, k=1, verbose=False):
    # Calculate Q1 and Q3 for each column of the sorted data
    q1 = np.percentile(data, 25, axis=axis)
    q3 = np.percentile(data, 75, axis=axis)
    iqr = q3 - q1

    # Masks for non-outlier values
    mask = (data >= (q1 - k*iqr)[None, :]) & (data <= (q3 + k*iqr)[None, :])

    # Compute trimmed mean and std
    trimmed_data = np.where(mask, data, np.nan)
    trimmed_mean = np.nanmean(trimmed_data, axis=axis)
    trimmed_SEM = np.nanstd(trimmed_data, axis=axis)/np.sqrt(np.sum(mask, axis=axis))

    # Calculate the number of all trials and the number of outliers
    n_total = data.shape[axis]
    n_outliers = n_total - np.sum(mask, axis=axis)

    if verbose:
        # Print the total number of trials and the number of outliers removed
        print(f"All trials: {n_total}, removed outliers: {round(np.sum(n_outliers)/n_total)}")

    return trimmed_mean, trimmed_SEM


def low_pass_filter(data, sfreq, cutoff=50, order=5):
    nyquist = 0.5 * sfreq
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def translate_case(case):
    case_id_dict = {'all': '1', 'endo': '3 | 4', 'exo': '5 | 6', 'valid': '7', 'endo valid': '(3 | 4) & 7', 'exo valid': '(5 | 6) & 7',
                    'invalid': '8', 'endo invalid': '(3 | 4) & 8', 'exo invalid': '(5 | 6) & 8',
                    'cue left': '3 | 5', 'endo cue left': '3', 'exo cue left': '5', 'cue right': '4 | 6', 'endo cue right': '4', 'exo cue right': '6',
                    'stim left': '12', 'endo stim left': '(3 | 4) & 12', 'exo stim left': '(5 | 6) & 12',
                    'stim right': '13', 'endo stim right': '(3 | 4) & 13', 'exo stim right': '(5 | 6) & 13',
                    'endo fast': '(3 | 4) & 9', 'exo fast': '(5 | 6) & 9', 'endo slow': '(3 | 4) & 10', 'exo slow': '(5 | 6) & 10',
                    'endo valid fast': '(3 | 4) & 7 & 9', 'exo valid fast': '(5 | 6) & 7 & 9', 'endo valid slow': '(3 | 4) & 7 & 10', 'exo valid slow': '(5 | 6) & 7 & 10',
                    'endo invalid fast': '(3 | 4) & 8 & 9', 'exo invalid fast': '(5 | 6) & 8 & 9', 'endo invalid slow': '(3 | 4) & 8 & 10', 'exo invalid slow': '(5 | 6) & 8 & 10'}
    case_by_id = case_id_dict[case]
    return case_by_id


def reaction_time_table(case, verbose=False):
    behavior_compare, experiment = fb.create_allsubs_compare()
    for subject_id in range (1,19):
        behavior_before, behavior_after = fb.load_behavior(subject_id)
        behavior_compare = fb.allsubs_compare(subject_id, behavior_before, behavior_after, behavior_compare, experiment, verbose=False)

    behavior_compare = behavior_compare.loc[(behavior_compare['response'] == 1) & 
                                            (behavior_compare['reaction time'] > 0.05) & (behavior_compare['reaction time'] < 1)]
    behavior_before, behavior_after = fb.filter_behav(case, behavior_compare.loc[behavior_compare['session'] == 'before'], 
                                                    behavior_compare.loc[behavior_compare['session'] == 'after'])

    behavior_compare = pd.concat([behavior_before, behavior_after])
    rt_sham_before = behavior_before.loc[behavior_compare['Real stimulation'] == 0]
    rt_sham_after = behavior_after.loc[behavior_compare['Real stimulation'] == 0 ]
    rt_real_before = behavior_before.loc[behavior_compare['Real stimulation'] == 1]
    rt_real_after = behavior_after.loc[behavior_compare['Real stimulation'] == 1]

    # remove outliers
    k_out = [1, 1, 1, 1]
    behav_sham_before = fb.remove_outlier(rt_sham_before, k=k_out[0], left=False, right=True, verbose=verbose)
    behav_sham_after = fb.remove_outlier(rt_sham_after, k=k_out[1], left=True, right=False, verbose=verbose)
    behav_real_before = fb.remove_outlier(rt_real_before, k=k_out[2], left=True, right=False, verbose=verbose)
    behav_real_after = fb.remove_outlier(rt_real_after, k=k_out[3], left=False, right=True, verbose=verbose)

    rt_sham_before = behav_sham_before.loc[:, 'reaction time'].tolist()
    rt_sham_after = behav_sham_after.loc[:, 'reaction time'].tolist()
    rt_real_before = behav_real_before.loc[:, 'reaction time'].tolist()
    rt_real_after = behav_real_after.loc[:, 'reaction time'].tolist()

    # Calculate means
    rt_means = [np.mean(rt_sham_before), np.mean(rt_sham_after), np.mean(rt_real_before), np.mean(rt_real_after)]

    # Calculate standard errors
    rt_std_errors = [
        np.std(rt_sham_before) / np.sqrt(len(rt_sham_before)), np.std(rt_sham_after) / np.sqrt(len(rt_sham_after)),
        np.std(rt_real_before) / np.sqrt(len(rt_real_before)), np.std(rt_real_after) / np.sqrt(len(rt_real_after))
    ]

    return behav_sham_before, behav_sham_after, behav_real_before, behav_real_after, rt_means, rt_std_errors


def onesub_evoked_response(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass=None, lopass=None, baseline_set=None, detrend=0):
    eeg_before, eeg_after = load_eeg(subject_id)
    if hipass is not None:
        eeg_before.filter(l_freq=hipass, h_freq=None)
    if lopass is not None:
        eeg_after.filter(l_freq=None, h_freq=lopass)

    events, event_dict = make_default_events(eeg_before)
    picked_events, picked_events_dict = make_custom_events(eeg_before, events, event_dict, trials_before, case_by_id)
    epochs_before = make_epochs(eeg_before, picked_events, picked_events_dict, watch, baseline_set, tmin=tmin, tmax=tmax, detrend=detrend)
    evoked_before = get_evoked_response(epochs_before)

    events, event_dict = make_default_events(eeg_after)
    picked_events, picked_events_dict = make_custom_events(eeg_after, events, event_dict, trials_after, case_by_id)
    epochs_after = make_epochs(eeg_after, picked_events, picked_events_dict, watch, baseline_set, tmin=tmin, tmax=tmax, detrend=detrend)
    evoked_after = get_evoked_response(epochs_after)

    return evoked_before, evoked_after


def get_inuse_trials(subject_id, before, after):
    trials_before = before.loc[before['subject id'] == subject_id, 'trial'].tolist()
    trials_after = after.loc[after['subject id'] == subject_id, 'trial'].tolist()
    return trials_before, trials_after


def pipeline_evoked_response_allsubs(case, watch, tmin, tmax, hipass=0.3, lopass=30):
    real_ids = [1, 3, 4, 5, 9, 12, 13, 17, 18]
    sham_ids = [2, 6, 7, 8, 10, 11, 14, 15, 16]
    sham_evoked_before = np.empty((0, 32, round((tmax-tmin)*1200+1)))
    sham_evoked_after = np.empty((0, 32, round((tmax-tmin)*1200+1)))
    real_evoked_before = np.empty((0, 32, round((tmax-tmin)*1200+1)))
    real_evoked_after = np.empty((0, 32, round((tmax-tmin)*1200+1)))

    case_by_id = translate_case(case)

    behav_sham_before, behav_sham_after, behav_real_before, behav_real_after, rt_means, rt_std_errors = reaction_time_table(case)

    for subject_id in sham_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_sham_before, behav_sham_after)
        evoked_before, evoked_after = onesub_evoked_response(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass)
        sham_evoked_before = np.concatenate((sham_evoked_before, evoked_before), axis=0)
        sham_evoked_after = np.concatenate((sham_evoked_after, evoked_after), axis=0)
    
    for subject_id in real_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_real_before, behav_real_after)
        evoked_before, evoked_after = onesub_evoked_response(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass)
        real_evoked_before = np.concatenate((real_evoked_before, evoked_before), axis=0)
        real_evoked_after = np.concatenate((real_evoked_after, evoked_after), axis=0)

    return sham_evoked_before, sham_evoked_after, real_evoked_before, real_evoked_after, rt_means, rt_std_errors


def makeup_subject(eeg_data, tmin, tmax, baseline=(0,0)):
    # get placement data of standard 10-20 system
    montage_1020 = mne.channels.make_standard_montage('standard_1020')
    positions_1020 = montage_1020._get_ch_pos()
    elec_coords_1020 = {ch_name: coord for ch_name, coord in positions_1020.items() if ch_name in montage_1020.ch_names}

    # Define channel names and types
    ch_names = ['Fp1', 'Fp2', 
                        'AF3', 'AF4', 
                        'F7', 'F3', 'Fz', 'F4', 'F8',
                        'FC1', 'FC2',
                        'T7', 'C3', 'Cz', 'C4', 'T8',
                        'CP5', 'CP1', 'CP2', 'CP6',
                        'P7', 'P5', 'P3', 'Pz', 'P4', 'P6', 'P8',
                        'PO3', 'PO4',
                        'O1', 'Oz', 'O2'] + ['stim']
    ch_types = ['eeg'] * 32 + ['stim']

    # Create the info object
    info = mne.create_info(ch_names, sfreq=1200, ch_types=ch_types, verbose=False)
    # Create raw object
    raw = mne.io.RawArray(eeg_data, info, verbose=False)

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

    # events
    events = mne.find_events(raw, stim_channel="stim", verbose=False)
    event_dict = {"stim": 1}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, verbose=False, detrend=0) # detrend!!

    return epochs# .average()


def pick_cortex(command):
    channels = {}

    if 'lf' in command:
        channels.update({'Fp1':1, 'AF3':3}) # 2
    if 'rf' in command:
        channels.update({'Fp2':2, 'AF4':4}) # 2
    if 'lc' in command:
        channels.update({'FC1':10, 'C3':13})
    if 'rc' in command:
        channels.update({'FC2':11, 'C4':15})
    if 'lp' in command:
        channels.update({'CP1':18, 'P5':22, 'P3':23})
    if 'rp' in command:
        channels.update({'CP2':19, 'P4':25, 'P6':26})
    if 'lt' in command:
        channels.update({'F7':5, 'F3':6, 'T7':12, 'CP5':17, 'P7':21})
    if 'rt' in command:
        channels.update({'F4':8, 'F8':9, 'T8':16, 'CP6':20, 'P8':27})

    if 'frontal' in command:
        channels.update({'Fp1':1, 'Fp2':2, 'AF3':3, 'AF4':4, 'Fz':7}) # 5
    if 'central' in command:
        channels.update({'FC1':10, 'FC2':11, 'C3':13, 'Cz':14, 'C4':15}) # 5
    if 'parietal' in command:
        channels.update({'CP1':18, 'CP2':19, 'P5':22, 'P3':23, 'Pz':24, 'P4':25, 'P6':26}) # 7
    if 'occipital' in command:
        channels.update({'PO3':28, 'PO4':29, 'O1':30, 'Oz':31, 'O2':32}) # 5
    if 'temporal' in command:
        channels.update({'F7':5, 'F3':6, 'F4':8, 'F8':9, 'T7':12, 'T8':16, 'CP5':17, 'CP6':20, 'P7':21, 'P8':27}) # 10
    
    if 'left' in command:
        channels.update({'Fp1':1, 'AF3':3, 'F7':5, 'F3':6, 'FC1':10, 'T7':12, 'C3':13, 'CP1':18, 'CP5':17, 'P5':22, 'P3':23, 'Pz':24, 'PO3':28, 'O1':30}) # 14
    if 'right' in command:
        channels.update({'Fp2':2, 'AF4':4, 'F4':8, 'F8':9, 'FC2':11, 'T8':16, 'C4':15, 'CP2':19, 'CP6':20, 'P4':25, 'P6':26, 'P8':27, 'PO4':29, 'O2':32}) # 14
    if 'all' in command:
        channels.update({'Fp1':1, 'Fp2':2, 'AF3':3, 'AF4':4, 'F7':5, 'F3':6, 'Fz':7, 'F4':8, 'F8':9, 'FC1':10, 'FC2':11, 'T7':12, 'C3':13, 'Cz':14, 'C4':15, 'T8':16, 'CP1':18, 'CP2':19, 'CP5':17, 'CP6':20, 'P7':21, 'P5':22, 'P3':23, 'Pz':24, 'P4':25, 'P6':26, 'P8':27, 'PO3':28, 'PO4':29, 'O1':30, 'Oz':31, 'O2':32}) # 32
    
    if 'anode' in command:
        channels.update({'P6':26})
    if 'cathode' in command:
        channels.update({'Cz':14})

    return channels


def pipeline_ERP_bysubs(case, watch, tmin, tmax, hipass=0.3, lopass=30):
    real_ids = [1, 3, 4, 5, 9, 12, 13, 17, 18]
    sham_ids = [2, 6, 7, 8, 10, 11, 14, 15, 16]
    sham_evoked_before = []
    sham_evoked_after = []
    real_evoked_before = []
    real_evoked_after = []

    case_by_id = translate_case(case)

    behav_sham_before, behav_sham_after, behav_real_before, behav_real_after, rt_means, rt_std_errors = reaction_time_table(case)

    for subject_id in sham_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_sham_before, behav_sham_after)
        evoked_before, evoked_after = onesub_ERP_mne(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass)
        sham_evoked_before.append(evoked_before)
        sham_evoked_after.append(evoked_after)
    
    for subject_id in real_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_real_before, behav_real_after)
        evoked_before, evoked_after = onesub_ERP_mne(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass)
        real_evoked_before.append(evoked_before)
        real_evoked_after.append(evoked_after)

    return sham_evoked_before, sham_evoked_after, real_evoked_before, real_evoked_after, rt_means, rt_std_errors


def onesub_ERP_mne(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass):
    # raw
    eeg_before, eeg_after = load_eeg(subject_id) # raw
    eeg_before.filter(l_freq=hipass, h_freq=lopass)
    eeg_after.filter(l_freq=hipass, h_freq=lopass)

    events, event_dict = make_default_events(eeg_before)
    picked_events, picked_events_dict = make_custom_events(eeg_before, events, event_dict, trials_before, case_by_id)
    epochs_before = make_epochs(eeg_before, picked_events, picked_events_dict, watch, tmin=tmin, tmax=tmax)
    evoked_before = epochs_before.average()

    events, event_dict = make_default_events(eeg_after)
    picked_events, picked_events_dict = make_custom_events(eeg_after, events, event_dict, trials_after, case_by_id)
    epochs_after = make_epochs(eeg_after, picked_events, picked_events_dict, watch, tmin=tmin, tmax=tmax)
    evoked_after = epochs_after.average()

    return evoked_before, evoked_after


def pipeline_EP_allsubs(case, watch, tmin, tmax, hipass=0.3, lopass=30, baseline=None, detrend=0):
    real_ids = [1, 3, 4, 5, 9, 12, 13, 17, 18]
    sham_ids = [2, 6, 7, 8, 10, 11, 14, 15, 16]
    sham_evoked_before = []
    sham_evoked_after = []
    real_evoked_before = []
    real_evoked_after = []

    case_by_id = translate_case(case)

    behav_sham_before, behav_sham_after, behav_real_before, behav_real_after, rt_means, rt_std_errors = reaction_time_table(case)

    for subject_id in sham_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_sham_before, behav_sham_after)
        evoked_before, evoked_after = onesub_evoked_response(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass, baseline, detrend=detrend)
        sham_evoked_before.append(evoked_before)
        sham_evoked_after.append(evoked_after)
    
    for subject_id in real_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_real_before, behav_real_after)
        evoked_before, evoked_after = onesub_evoked_response(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass, baseline, detrend=detrend)
        real_evoked_before.append(evoked_before)
        real_evoked_after.append(evoked_after)

    return sham_evoked_before, sham_evoked_after, real_evoked_before, real_evoked_after, rt_means, rt_std_errors


def detect_EP(signal, time_vector, windows):
    peaks = np.empty((2, 3))
    peak_order = ["N75", "P100", "N145"]  # Use this to ensure you're saving values in the right order

    for idx, peak in enumerate(peak_order):
        # Get indices for the current time window
        indices = np.where((time_vector >= windows[peak][0]) & (time_vector <= windows[peak][1]))[0]

        # Check if the current peak is negative or positive
        if 'N' in peak:  # For negative peaks
            peak_indices, _ = find_peaks(-signal[indices])         
        else:  # For positive peaks
            peak_indices, _ = find_peaks(signal[indices])
            
        # If a peak is found within the window, save its time and amplitude
        if len(peak_indices) > 0:
            if 'N' in peak:
                main_peak_idx = indices[peak_indices[np.argmin(signal[indices][peak_indices])]]
            else:
                main_peak_idx = indices[peak_indices[np.argmax(signal[indices][peak_indices])]]

            peaks[0, idx] = time_vector[main_peak_idx]
            peaks[1, idx] = signal[main_peak_idx]
        else:
            peaks[0, idx] = np.nan
            peaks[1, idx] = np.nan

    return peaks


def rm_outlier(data, lower_k=1.5, upper_k=1.5, verbose=False):
    """
    Remove outliers from a 1D array or list based on the IQR method.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - lower_k * iqr
    upper_bound = q3 + upper_k * iqr
    cleaned_data = [x for x in data if lower_bound <= x <= upper_bound]
    if verbose:
        print(f"Removed {len(data) - len(cleaned_data)} outliers from the data.")
    return cleaned_data


def pipeline_FBP_allsubs(case):
    case_by_id = translate_case(case)
    behav_sham_before, behav_sham_after, behav_real_before, behav_real_after, rt_means, rt_std_errors = reaction_time_table(case)

    real_ids = [1, 3, 4, 5, 9, 12, 13, 17, 18]
    sham_ids = [2, 6, 7, 8, 10, 11, 14, 15, 16]
    sham_before = np.empty((5, 32, behav_sham_before.shape[0]))
    sham_after = np.empty((5, 32, behav_sham_after.shape[0]))
    real_before = np.empty((5, 32, behav_real_before.shape[0]))
    real_after = np.empty((5, 32, behav_real_after.shape[0]))

    for subject_id in sham_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_sham_before, behav_sham_after)
        FBP_before, FBP_after = onesub_FBP(subject_id, case_by_id, trials_before, trials_after)
        sham_before = np.concatenate((sham_before, FBP_before), axis=2)
        sham_after = np.concatenate((sham_after, FBP_after), axis=2)
    
    for subject_id in real_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_real_before, behav_real_after)
        FBP_before, FBP_after = onesub_FBP(subject_id, case_by_id, trials_before, trials_after)
        real_before = np.concatenate((real_before, FBP_before), axis=2)
        real_after = np.concatenate((real_after, FBP_after), axis=2)

    return sham_before, sham_after, real_before, real_after


def onesub_FBP(subject_id, case_by_id, trials_before, trials_after):
    # laod eeg raw file
    # print(subject_id)
    eeg_before, eeg_after = load_eeg(subject_id)
    # print('before...')
    fbp_before = onesession_FBP(eeg_before, trials_before, case_by_id)
    # print('after...')
    fbp_after = onesession_FBP(eeg_after, trials_after, case_by_id)

    return fbp_before, fbp_after


def onesession_FBP(eeg_data, trials, case_by_id):
    fbp_session = np.empty((5, 32, 0))
    
    events, event_dict = make_default_events(eeg_data)
    picked_events, picked_events_dict = make_custom_events(eeg_data, events, event_dict, trials, case_by_id)
    trial_start_times, trial_end_times = start_end_times(picked_events)
    eeg_data = eeg_data.get_data()[1:33, :]
    for i, end_time in enumerate(trial_end_times):
        trial_eeg = eeg_data[:,trial_start_times[i]:end_time]
        fbp = compute_band_power(trial_eeg)
        fbp_session = np.concatenate((fbp_session, fbp[:, :, None]), axis=2)

    return fbp_session

def start_end_times(picked_events):
    trial_fixations = picked_events[picked_events[:, 2] == 1]
    trial_stims = picked_events[picked_events[:, 2] == 11]
    trial_start_times = trial_fixations[:,0] # time steps of fixation onsets
    trial_end_times = (trial_stims[:,0] + (1200*1.5)).astype(int) # time steps of stimulus onsets
    return trial_start_times, trial_end_times


def compute_band_power(trial_eeg, fs=1200, bands=[[4, 7], [8, 12], [12.5, 30], [30, 60], [60, 100]]): 
    num_channels = 32
    fbp = np.zeros((len(bands), num_channels))
    
    # Compute the power spectral density for each channel
    for ch in range(num_channels):
        freqs, psd = welch(trial_eeg[ch,:], fs=fs, nperseg=1024, noverlap=512, scaling='spectrum')
        
        for idx, band in enumerate(bands):
            mask = (freqs >= band[0]) & (freqs <= band[1])
            fbp[idx, ch] = np.sum(psd[mask])
    
    # Normalize by time
    duration = trial_eeg.shape[1] / fs
    # print(trial_eeg.shape, duration)
    fbp /= duration

    baseline = baseline_fbp(trial_eeg)
    fbp = fbp - baseline
    
    return fbp


def baseline_fbp(trial_eeg, fs=1200, bands=[[4, 7], [8, 12], [12.5, 30], [30, 60], [60, 100]]):
    fixation_dur = 1.5 # seconds
    fixation_eeg = trial_eeg[:, :int(fs*fixation_dur)]

    num_channels = 32
    fbp = np.zeros((len(bands), num_channels))
    
    # Compute the power spectral density for each channel
    for ch in range(num_channels):
        freqs, psd = welch(fixation_eeg[ch,:], fs=fs, nperseg=1024, noverlap=512, scaling='spectrum')
        
        for idx, band in enumerate(bands):
            mask = (freqs >= band[0]) & (freqs <= band[1])
            fbp[idx, ch] = np.sum(psd[mask])
    
    # Normalize by time
    duration = fixation_eeg.shape[1] / fs
    # print(trial_eeg.shape, duration)
    fbp /= duration
    
    return fbp


def pipeline_time_freq(data_list, tmin, tmax, lofreq=1, hifreq=100, freqdiv=1):
    know_list = [[],[],[],[]]
    # by channel, remove bad channels
    bad_channels = [
        [ # sham before
            [], [], [], [], [], [], [22,21], [5,9], []
        ],
        [ # sham after
            [], [], [], [], [], [], [], [], []
        ],
        [ # real before
            [], [], [], [], [], [], [], [], []
        ],
        [ # real after
            [], [], [], [], [], [7], [], [], []
        ]
    ]
    # loops
    for i, session_data in enumerate(data_list):
        print(f'processing session {i}...')
        for channel in range(1,33):
            one_chan_list = []
            for group_id in range(9):
                if channel not in bad_channels[i][group_id]:
                    for trial in range(session_data[group_id].shape[0]):
                        one_trial = session_data[group_id][trial, channel-1, :]
                        tfmap = one_trial_tfmap(one_trial, tmin, tmax, lofreq, hifreq, freqdiv, normalize=False)
                        one_chan_list.append(tfmap)

            one_chan = np.stack(one_chan_list, axis=0)
            know_list[i].append(one_chan)
    return know_list


def morlet_wavelet(f, t, sigma=1):
    """Generate a Morlet wavelet."""
    sine_wave = np.exp(2j * np.pi * f * t)
    gaussian_win = np.exp(-t ** 2 / (2 * sigma ** 2))
    return sine_wave * gaussian_win


def one_trial_tfmap(data, tmin, tmax, lofreq, hifreq, freqdiv, normalize=False):
    # Time and frequency vectors
    times = np.linspace(tmin, tmax, num=len(data))
    freqs = np.arange(lofreq, hifreq, freqdiv)
    
    # Initialize TF map
    tfmap = np.zeros((len(freqs), len(times)))

    # Calculate TF representation for each frequency
    for i_f, f in enumerate(freqs):
        wavelet = morlet_wavelet(f, times)
        conv_res = np.convolve(data, wavelet, mode='same')
        power = np.abs(conv_res) ** 2
        tfmap[i_f, :] = power

    if normalize:
        # Identify the baseline index for t=0
        baseline_index = np.argmin(np.abs(times))

        # Subtract the baseline value from each frequency row
        for i_f in range(tfmap.shape[0]):
            baseline_value = tfmap[i_f, baseline_index]
            tfmap[i_f, :] -= baseline_value

            # Normalize the data between (-1, 1)
            max_val = np.max(np.abs(tfmap[i_f, :]))
            if max_val != 0:  # to avoid dividing by zero
                tfmap[i_f, :] /= max_val

    return tfmap


def normalize_tfmap(tfmap, times, axis):
    if axis == "freq":
        # Identify the baseline index for t=0
        baseline_index = np.argmin(np.abs(times))
        
        # Subtract the baseline value from each frequency row
        for i_f in range(tfmap.shape[0]):
            baseline_value = tfmap[i_f, baseline_index]
            tfmap[i_f, :] -= baseline_value

            # Normalize the data between (-1, 1)
            max_val = np.max(np.abs(tfmap[i_f, :]))
            if max_val != 0:  # to avoid dividing by zero
                tfmap[i_f, :] /= max_val

    elif axis == "time":
        # Normalize each time column between (0, 1)
        for i_t in range(tfmap.shape[1]):
            min_val = tfmap[:, i_t].min()
            max_val = tfmap[:, i_t].max()
            
            if max_val != min_val:  # to avoid dividing by zero and to handle cases where max and min are equal
                tfmap[:, i_t] = (tfmap[:, i_t] - min_val) / (max_val - min_val)
            else:
                tfmap[:, i_t] = 0  # set column to 0 if max and min values are the same

    else:
        raise ValueError("Invalid axis value. Use either 'time' or 'freq'.")

    return tfmap


def pipeline_EP_RT(case, watch, tmin, tmax, hipass=None, lopass=None, baseline=None, move_baseline=True, detrend=0):
    real_ids = [1, 3, 4, 5, 9, 12, 13, 17, 18]
    sham_ids = [2, 6, 7, 8, 10, 11, 14, 15, 16]

    sham_evoked_before = []
    sham_RT_before = []
    sham_evoked_after = []
    sham_RT_after = []
    real_evoked_before = []
    real_RT_before = []
    real_evoked_after = []
    real_RT_after = []

    case_by_id = translate_case(case)
    behav_sham_before, behav_sham_after, behav_real_before, behav_real_after, _, _ = reaction_time_table(case)

    for subject_id in sham_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_sham_before, behav_sham_after)
        evoked_before, evoked_after = onesub_evoked_response(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass, baseline, detrend=detrend)
        RT_before = behav_sham_before.loc[behav_sham_before['subject id'] == subject_id, 'reaction time'].to_numpy()
        RT_after = behav_sham_after.loc[behav_sham_after['subject id'] == subject_id, 'reaction time'].to_numpy()
        sham_evoked_before.append(evoked_before)
        sham_evoked_after.append(evoked_after)

        if RT_before.shape[0]>evoked_before.shape[0]:
            RT_before = RT_before[:evoked_before.shape[0]]
        if RT_after.shape[0]>evoked_after.shape[0]:
            RT_after = RT_after[:evoked_after.shape[0]]
        sham_RT_before.append(RT_before)
        sham_RT_after.append(RT_after)
    
    for subject_id in real_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_real_before, behav_real_after)
        evoked_before, evoked_after = onesub_evoked_response(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass, baseline, detrend=detrend)
        RT_before = behav_real_before.loc[behav_real_before['subject id'] == subject_id, 'reaction time'].to_numpy()
        RT_after = behav_real_after.loc[behav_real_after['subject id'] == subject_id, 'reaction time'].to_numpy()
        real_evoked_before.append(evoked_before)
        real_evoked_after.append(evoked_after)

        if RT_before.shape[0]>evoked_before.shape[0]:
            RT_before = RT_before[:evoked_before.shape[0]]
        if RT_after.shape[0]>evoked_after.shape[0]:
            RT_after = RT_after[:evoked_after.shape[0]]
        real_RT_before.append(RT_before)
        real_RT_after.append(RT_after)

    EP_lists = [sham_evoked_before, sham_evoked_after, real_evoked_before, real_evoked_after]
    # move the baseline to mean of [tmin, 0]
    if move_baseline:
        for i, session_data in enumerate(EP_lists):
            for sub_data in session_data:
                for trial in range(sub_data.shape[0]):
                    for channel in range (sub_data.shape[1]):
                        sub_data[trial, channel, :] = sub_data[trial, channel, :] - np.mean(sub_data[trial, channel, :int(abs(tmin)*1200)])
                EP_lists[i].append(sub_data)

    # remove bad channels
    RT_lists = [sham_RT_before, sham_RT_after, real_RT_before, real_RT_after]
    EP_bychan_lists = [[],[],[],[]]
    RT_bychan_lists = [[],[],[],[]]
    # choose channels
    bad_channels = [
            [ # sham before
                [], [], [], [], [], [], [22,21], [5,9], []
            ],
            [ # sham after
                [], [], [], [], [], [], [], [], []
            ],
            [ # real before
                [], [], [], [], [], [], [], [], []
            ],
            [ # real after
                [], [], [], [], [], [7], [], [], []
            ]
    ]

    # remove bad channels
    for i, session_data in enumerate(EP_lists):
        one_group_list = []
        one_group_RT_list = []

        for group_id in range(9):
            sub_EP = session_data[group_id]

            if bad_channels[i][group_id] == []:
                # rm_indices = [x - 1 for x in bad_channels[i][group_id]]
                # sub_EP[:, rm_indices, :] = np.nan
                one_group_list.append(sub_EP)
                one_group_RT_list.append(RT_lists[i][group_id])
            
        one_group = np.concatenate(one_group_list, axis=0)
        one_group_RT = np.concatenate(one_group_RT_list, axis=0)
        EP_bychan_lists[i] = one_group
        RT_bychan_lists[i] = one_group_RT

    return EP_bychan_lists, RT_bychan_lists


def extract_channels_EP_RT(EP_bychan_lists, pick_channels):
    pick_channels = [x - 1 for x in pick_channels]
    sb, sa, rb, ra = EP_bychan_lists[0][:, pick_channels, :], EP_bychan_lists[1][:, pick_channels, :], EP_bychan_lists[2][:, pick_channels, :], EP_bychan_lists[3][:, pick_channels, :]
    extract = [[],[],[],[]]
    for i, session in enumerate([sb, sa, rb, ra]):
        session_extract = np.empty((session.shape[0], session.shape[2]))
        for trial in range (session.shape[0]):
            one_trial = session[trial,:,:]
            one_trial = one_trial[~np.isnan(one_trial).any(axis=1)] # remove nan channels
            averaged_trial = np.empty((one_trial.shape[1]))
            for time_step in range (one_trial.shape[1]):
                cleaned = rm_outlier(one_trial[:,time_step], lower_k=1.5, upper_k=1.5, verbose=False)
                averaged_trial[time_step] = np.mean(cleaned) # nanmean() ????
            session_extract[trial] = averaged_trial
        extract[i] = session_extract
    return extract


def pipeline_csp_onecond(case, watch, tmin, tmax, hipass, lopass, baseline=None, mean_baseline=True):
    sham_before, sham_after, real_before, real_after, _, _ = pipeline_EP_allsubs(case, watch, tmin, tmax, hipass, lopass, baseline=None)

    # move the baseline to mean of [tmin, 0]
    if mean_baseline:
        data_list = [[],[],[],[]]
        for i, session_data in enumerate([sham_before, sham_after, real_before, real_after]):
            for sub_data in session_data:
                for trial in range(sub_data.shape[0]):
                    for channel in range (sub_data.shape[1]):
                        sub_data[trial, channel, :] = sub_data[trial, channel, :] - np.mean(sub_data[trial, channel, :int(abs(tmin)*1200)])
                data_list[i].append(sub_data[:,:,int(abs(tmin)*1200):]) # only use t>0 for CSP
    else:
        data_list = [sham_before, sham_after, real_before, real_after]

    # remove bad channels
    bad_channels = [
            [ # sham before
                [], [], [], [], [], [], [22,21], [5,9], []
            ],
            [ # sham after
                [], [], [], [], [], [], [], [], []
            ],
            [ # real before
                [], [], [], [], [], [], [], [], []
            ],
            [ # real after
                [], [], [], [], [], [7], [], [], []
            ]
    ]
    cleanEP_lists = [[],[],[],[]]
    for i, session in enumerate (data_list):
        one_session = []
        for group_id in range(9):
            if bad_channels[i][group_id] == []:
                one_session.append(session[group_id])
        cleanEP_lists[i] = np.concatenate(one_session, axis=0)

    return cleanEP_lists


def find_closest_index(array, value):
    """
    Find the index of the closest value in an array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def band_power(psds, freqs, band):
    freq_start = find_closest_index(freqs, band[0])
    freq_end = find_closest_index(freqs, band[1])
    power = np.sum(psds[:,freq_start:freq_end], axis=1)*np.diff(freqs)[0]
    return power


def channel_pos():
    eeg_data = np.zeros((32, 2))
     # get placement data of standard 10-20 system
    montage_1020 = mne.channels.make_standard_montage('standard_1020')
    positions_1020 = montage_1020._get_ch_pos()
    elec_coords_1020 = {ch_name: coord for ch_name, coord in positions_1020.items() if ch_name in montage_1020.ch_names}

    # Define channel names and types
    ch_names = ['Fp1', 'Fp2', 
                        'AF3', 'AF4', 
                        'F7', 'F3', 'Fz', 'F4', 'F8',
                        'FC1', 'FC2',
                        'T7', 'C3', 'Cz', 'C4', 'T8',
                        'CP5', 'CP1', 'CP2', 'CP6',
                        'P7', 'P5', 'P3', 'Pz', 'P4', 'P6', 'P8',
                        'PO3', 'PO4',
                        'O1', 'Oz', 'O2']
    ch_types = ['eeg'] * 32

    # Create the info object
    info = mne.create_info(ch_names, sfreq=1200, ch_types=ch_types, verbose=False)
    # Create raw object
    raw = mne.io.RawArray(eeg_data, info, verbose=False)

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

    return raw


def pipeline_session_RT(case, watch, tmin, tmax, hipass=None, lopass=None, baseline=None, move_baseline=True, detrend=0):
    real_ids = [1, 3, 4, 5, 9, 12, 13, 17, 18]
    sham_ids = [2, 6, 7, 8, 10, 11, 14, 15, 16]

    sham_evoked_before = []
    sham_RT_before = []
    sham_evoked_after = []
    sham_RT_after = []
    real_evoked_before = []
    real_RT_before = []
    real_evoked_after = []
    real_RT_after = []

    case_by_id = translate_case(case)
    behav_sham_before, behav_sham_after, behav_real_before, behav_real_after, _, _ = reaction_time_table(case)

    for subject_id in sham_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_sham_before, behav_sham_after)
        evoked_before, evoked_after = onesub_evoked_response(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass, baseline, detrend=detrend)
        RT_before = behav_sham_before.loc[behav_sham_before['subject id'] == subject_id, 'reaction time'].to_numpy()
        RT_after = behav_sham_after.loc[behav_sham_after['subject id'] == subject_id, 'reaction time'].to_numpy()
        sham_evoked_before.append(evoked_before)
        sham_evoked_after.append(evoked_after)

        if RT_before.shape[0]>evoked_before.shape[0]:
            RT_before = RT_before[:evoked_before.shape[0]]
        if RT_after.shape[0]>evoked_after.shape[0]:
            RT_after = RT_after[:evoked_after.shape[0]]
        sham_RT_before.append(RT_before)
        sham_RT_after.append(RT_after)
    
    for subject_id in real_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_real_before, behav_real_after)
        evoked_before, evoked_after = onesub_evoked_response(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass, baseline, detrend=detrend)
        RT_before = behav_real_before.loc[behav_real_before['subject id'] == subject_id, 'reaction time'].to_numpy()
        RT_after = behav_real_after.loc[behav_real_after['subject id'] == subject_id, 'reaction time'].to_numpy()
        real_evoked_before.append(evoked_before)
        real_evoked_after.append(evoked_after)

        if RT_before.shape[0]>evoked_before.shape[0]:
            RT_before = RT_before[:evoked_before.shape[0]]
        if RT_after.shape[0]>evoked_after.shape[0]:
            RT_after = RT_after[:evoked_after.shape[0]]
        real_RT_before.append(RT_before)
        real_RT_after.append(RT_after)

    EP_lists = [sham_evoked_before, sham_evoked_after, real_evoked_before, real_evoked_after]
    # move the baseline to mean of [tmin, 0]
    if move_baseline:
        for i, session_data in enumerate(EP_lists):
            for sub_data in session_data:
                for trial in range(sub_data.shape[0]):
                    for channel in range (sub_data.shape[1]):
                        sub_data[trial, channel, :] = sub_data[trial, channel, :] - np.mean(sub_data[trial, channel, :int(abs(tmin)*1200)])
                EP_lists[i].append(sub_data)

    # remove bad channels
    RT_lists = [sham_RT_before, sham_RT_after, real_RT_before, real_RT_after]
    # choose channels
    bad_channels = [
            [ # sham before
                [], [], [], [], [], [], [22,21], [5,9], []
            ],
            [ # sham after
                [], [], [], [], [], [], [], [], []
            ],
            [ # real before
                [], [], [], [], [], [], [], [], []
            ],
            [ # real after
                [], [], [], [], [], [7], [], [], []
            ]
    ]

    # remove bad channels
    for i, session_data in enumerate(EP_lists):
        for group_id in range(9):
            if bad_channels[i][group_id] != []:
                bad_channel = [x - 1 for x in bad_channels[i][group_id]]
                EP_lists[i][group_id][:, bad_channel, :] = np.nan

    return EP_lists, RT_lists


def pipeline_session_channel(case, watch, tmin, tmax, hipass=None, lopass=None, baseline=None, detrend=0):
    real_ids = [1, 3, 4, 5, 9, 12, 13, 17, 18]
    sham_ids = [2, 6, 7, 8, 10, 11, 14, 15, 16]

    sham_evoked_before = []
    sham_evoked_after = []
    real_evoked_before = []
    real_evoked_after = []

    case_by_id = translate_case(case)
    behav_sham_before, behav_sham_after, behav_real_before, behav_real_after, _, _ = reaction_time_table(case)

    for subject_id in sham_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_sham_before, behav_sham_after)
        evoked_before, evoked_after = onesub_evoked_response(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass, baseline, detrend=detrend)
        sham_evoked_before.append(evoked_before)
        sham_evoked_after.append(evoked_after)
    
    for subject_id in real_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_real_before, behav_real_after)
        evoked_before, evoked_after = onesub_evoked_response(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass, baseline, detrend=detrend)
        real_evoked_before.append(evoked_before)
        real_evoked_after.append(evoked_after)

    eeg_by_subject = [sham_evoked_before, sham_evoked_after, real_evoked_before, real_evoked_after]

    # remove bad channels
    bad_channels = [
            [ # sham before
                [], [], [], [], [], [], [22,21], [5,9], []
            ],
            [ # sham after
                [], [], [], [], [], [], [], [], []
            ],
            [ # real before
                [], [], [], [], [], [], [], [], []
            ],
            [ # real after
                [], [], [], [], [], [7], [], [], []
            ]
    ]
    eeg_by_channel = []
    for i, session_by_subject in enumerate(eeg_by_subject):
        this_session = []
        for channel in range(32):
            this_channel = []
            for group_id in range(9):
                if channel+1 not in bad_channels[i][group_id]:
                    channel_session_subject = session_by_subject[group_id][:, channel, :]
                    this_channel.append(channel_session_subject)
            this_channel = np.concatenate(this_channel, axis=0)
            this_session.append(this_channel)
        eeg_by_channel.append(this_session)     

    return eeg_by_channel[0], eeg_by_channel[1], eeg_by_channel[2], eeg_by_channel[3]


def pipeline_bpdata_channel(case, watch, tmin, tmax, hipass=None, lopass=None, baseline=None, detrend=0):
    real_ids = [1, 3, 4, 5, 9, 12, 13, 17, 18]
    sham_ids = [2, 6, 7, 8, 10, 11, 14, 15, 16]

    sham_evoked_before = []
    sham_evoked_after = []
    real_evoked_before = []
    real_evoked_after = []

    case_by_id = translate_case(case)
    behav_sham_before, behav_sham_after, behav_real_before, behav_real_after, _, _ = reaction_time_table(case)

    for subject_id in sham_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_sham_before, behav_sham_after)
        evoked_before, evoked_after = onesub_evoked_response(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass, baseline, detrend=detrend)
        sham_evoked_before.append(evoked_before)
        sham_evoked_after.append(evoked_after)
    
    for subject_id in real_ids:
        trials_before, trials_after = get_inuse_trials(subject_id, behav_real_before, behav_real_after)
        evoked_before, evoked_after = onesub_evoked_response(subject_id, case_by_id, watch, tmin, tmax, trials_before, trials_after, hipass, lopass, baseline, detrend=detrend)
        real_evoked_before.append(evoked_before)
        real_evoked_after.append(evoked_after)

    eeg_by_subject = [sham_evoked_before, sham_evoked_after, real_evoked_before, real_evoked_after]

    # remove bad channels
    bad_channels = [
            [ # sham before
                [], [], [], [], [], [], [22,21], [5,9], []
            ],
            [ # sham after
                [], [], [], [], [], [], [], [], []
            ],
            [ # real before
                [], [], [], [], [], [], [], [], []
            ],
            [ # real after
                [], [], [], [], [], [7], [], [], []
            ]
    ]
    eeg_by_channel = []
    for i, session_by_subject in enumerate(eeg_by_subject):
        this_session = []
        for group_id in range(9):
            if bad_channels[i][group_id] == []:
                this_session.append(session_by_subject[group_id])
        eeg_by_channel.append(this_session)     

    return eeg_by_channel[0], eeg_by_channel[1], eeg_by_channel[2], eeg_by_channel[3]


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def sign_of_tvalue(condition1, condition2):
    # Step 1: Calculate median of each column for both conditions
    condition1_med = np.median(condition1, axis=0)
    condition2_med = np.median(condition2, axis=0)
    # Step 2: Subtract condition1_med from condition2_med
    subs = condition2_med - condition1_med
    # Step 3: Convert positive values to 1 and negative values to -1
    subs = np.where(subs > 0, 1, -1)
    
    return subs


def perm_test(condition1, condition2, adjacency):
    # T-test
    f_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [condition1, condition2], n_permutations=2000, adjacency=adjacency, tail=0, verbose=False)
    # Find significant clusters
    significant_clusters = np.nonzero(cluster_p_values < 0.05)[0]

    for idx in significant_clusters:
        cluster = clusters[idx][0]
        if cluster.shape[0] > 1:
            print(f"Cluster {idx}, p-value: {cluster_p_values[idx]}")
            print("Electrodes:", cluster)
    
    sign = sign_of_tvalue(condition1, condition2)
    t_obs = np.sqrt(f_obs) * sign
    return t_obs

def dfa_alpha(data, min_win=25, max_win=200):
    dfa_data = fu.toAggregated(data)
    pydfa = fathon.DFA(dfa_data)
    wins = fu.linRangeByStep(min_win, max_win)
    n,F = pydfa.computeFlucVec(wins, revSeg=True, polOrd=3)
    H, H_intercept = pydfa.fitFlucVec()
    return H