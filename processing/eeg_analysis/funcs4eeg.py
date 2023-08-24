import sys
import os
sys.path.insert(0, os.path.abspath('..'))
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
import behavior.func4behav as fb
import imp
imp.reload(fb)



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


def translate_case(case):
    case_id_dict = {'all': '1', 'endo': '3 | 4', 'exo': '5 | 6', 'valid': '7', 'endo valid': '(3 | 4) & 7', 'exo valid': '(5 | 6) & 7',
                    'invalid': '8', 'endo invalid': '(3 | 4) & 8', 'exo invalid': '(5 | 6) & 8',
                    'cue left': '3 | 5', 'endo cue left': '3', 'exo cue left': '5', 'cue right': '4 | 6', 'endo cue right': '4', 'exo cue right': '6',
                    'stim left': '12', 'endo stim left': '(3 | 4) & 12', 'exo stim left': '(5 | 6) & 12',
                    'stim right': '13', 'endo stim right': '(3 | 4) & 13', 'exo stim right': '(5 | 6) & 13'}
    case_by_id = case_id_dict[case]
    return case_by_id

def pipeline_evoked_response(subject_id, case, watch, tmin, tmax):
    real_ids = [1, 3, 4, 5, 9, 12, 13, 17, 18]
    sham_ids = [2, 6, 7, 8, 10, 11, 14, 15, 16]
    case_by_id = translate_case(case)

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

    # preprocessing
    k_out = [1, 0.9, 1, 0.9]
    rt_sham_before = fb.remove_outlier(rt_sham_before, k=k_out[0], left=False, right=True, verbose=True)
    rt_sham_after = fb.remove_outlier(rt_sham_after, k=k_out[1], left=True, right=False, verbose=True)
    rt_real_before = fb.remove_outlier(rt_real_before, k=k_out[2], left=True, right=False, verbose=True)
    rt_real_after = fb.remove_outlier(rt_real_after, k=k_out[3], left=False, right=True, verbose=True)

    rt_sham_before = rt_sham_before.loc[:, 'reaction time'].tolist()
    rt_sham_after = rt_sham_after.loc[:, 'reaction time'].tolist()
    rt_real_before = rt_real_before.loc[:, 'reaction time'].tolist()
    rt_real_after = rt_real_after.loc[:, 'reaction time'].tolist()
    rt_sham_before = [num * 1000 for num in rt_sham_before]
    rt_sham_after = [num * 1000 for num in rt_sham_after]
    rt_real_before = [num * 1000 for num in rt_real_before]
    rt_real_after = [num * 1000 for num in rt_real_after]

    
    # Calculate means
    means = [np.mean(rt_sham_before), np.mean(rt_sham_after), np.mean(rt_real_before), np.mean(rt_real_after)]

    # Calculate standard errors
    std_errors = [
        np.std(rt_sham_before) / np.sqrt(len(rt_sham_before)), np.std(rt_sham_after) / np.sqrt(len(rt_sham_after)),
        np.std(rt_real_before) / np.sqrt(len(rt_real_before)), np.std(rt_real_after) / np.sqrt(len(rt_real_after))
    ]

    # Calculate t-tests
    _, p_sham = mannwhitneyu(rt_sham_before, rt_sham_after)
    _, p_real = mannwhitneyu(rt_real_before, rt_real_after)
    _, p_before = mannwhitneyu(rt_sham_before, rt_real_before)
    _, p_after = mannwhitneyu(rt_sham_after, rt_real_after)


    # Calculate percentage changes
    percent_change_sham = ((np.mean(rt_sham_after) - np.mean(rt_sham_before)) / np.mean(rt_sham_before)) * 100
    percent_change_real = ((np.mean(rt_real_after) - np.mean(rt_real_before)) / np.mean(rt_real_before)) * 100

    # Bar chart
    labels = ['Sham Before', 'Sham After', 'Real Before', 'Real After']
    colors = ['lightblue', 'blue', 'lightcoral', 'red']

    fig, ax = plt.subplots()

    bars = ax.bar(labels, means, yerr=std_errors, color=colors, capsize=10)

    # Add p-values
    heights = [bar.get_height() + error for bar, error in zip(bars, std_errors)]
    fsize = 13
    ax.text(0.5, heights[0] + 2, f'p = {p_sham*4:.4f}', ha='center', va='bottom', fontsize=fsize)
    ax.text(2.5, heights[2] + 2, f'p = {p_real*4:.4f}', ha='center', va='bottom', fontsize=fsize)
    ax.text(1, heights[0] + 8, f'p = {p_before*4:.4f}', ha='center', va='bottom', fontsize=fsize)
    ax.text(2, heights[2] + 8, f'p = {p_after*4:.4f}', ha='center', va='bottom', fontsize=fsize)

    # Add percentage changes
    ax.text(0.5, heights[0] + 1, f'{percent_change_sham:.1f}%', ha='center', va='top', color='blue', fontsize=fsize)
    ax.text(2.5, heights[2] + 1, f'{percent_change_real:.1f}%', ha='center', va='top', color='red', fontsize=fsize)

    # Add some additional formatting if desired
    ax.set_ylabel('Reaction Time (ms)')
    ax.set_title(case)
    ax.set_ylim([250, 400])  # Adjust as needed

    save_path = os.path.join('..', '..', '..', 'docs', 'report', 'figs', case +'.png')
    plt.savefig(save_path, format='png')