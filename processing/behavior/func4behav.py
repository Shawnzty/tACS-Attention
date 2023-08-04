import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def create_compare():
    exp_path = os.path.join('..', '..', '..', 'data', 'experiment.csv')
    exp_info = pd.read_csv(exp_path)
    # display(exp_info)

    behavior_compare = pd.DataFrame(columns=['subject id', 'Real stimulation', 'RT mean shorten', 'RT median shorten',
                                            'RT mean shorten %', 'RT median shorten %', 'RT std decrease', 'trials_before', 'trials_after'])
    behavior_compare['subject id'] = exp_info['subject id']
    behavior_compare['Real stimulation'] = exp_info['Real stimulation']
    return behavior_compare


def create_allsubs_compare():
    behavior_compare = pd.DataFrame(columns=['subject id', 'Real stimulation', 'session', 'type', 'cue','valid','ICS','stim','response','reaction time'])
    experiment = pd.read_csv(os.path.join('..', '..', '..', 'data', 'experiment.csv'))
    return behavior_compare, experiment


def load_behavior(subject_id):
    behavior_before_path = os.path.join('..', '..', '..', 'data', str(subject_id), 'behavior_before.csv')
    behavior_before = pd.read_csv(behavior_before_path)
    behavior_after_path = os.path.join('..', '..', '..', 'data', str(subject_id), 'behavior_after.csv')
    behavior_after = pd.read_csv(behavior_after_path)

    return behavior_before, behavior_after


def make_compare(subject_id, behavior_before, behavior_after, behavior_compare, verbose=False):
    # find response=1, remove too fast and outliers
    respond_trials_before = behavior_before[(behavior_before['response'] == 1) & (behavior_before['reaction time'] > 0.05)]
    respond_trials_after = behavior_after[(behavior_after['response'] == 1) & (behavior_after['reaction time'] > 0.05)]

    if verbose:
        print(str(subject_id) + ' before-' + str(len(respond_trials_before)) + ' after-' + str(len(respond_trials_after)))
    respond_trials_before = remove_outlier(respond_trials_before)
    respond_trials_after = remove_outlier(respond_trials_after)
    if verbose:
        print('Outliers removed: before-' + str(len(respond_trials_before)) + ' after-' + str(len(respond_trials_after)) + "\n")
    
    behavior_compare.at[subject_id-1, 'trials_before'] = respond_trials_before['trial'].tolist()
    behavior_compare.at[subject_id-1, 'trials_after'] = respond_trials_after['trial'].tolist()

    respond_trials_before = respond_trials_before.copy()
    respond_trials_after = respond_trials_after.copy()
    respond_trials_before.loc[:, 'reaction time'] *= 1000
    respond_trials_after.loc[:, 'reaction time'] *= 1000

    # Extract 'reaction time' column values as lists
    data_before = respond_trials_before['reaction time'].tolist()
    data_after = respond_trials_after['reaction time'].tolist()
    
    # Calculate means of data_before and data_after and add to the dataframe
    mean_before = np.mean(data_before)
    std_before = np.std(data_before)
    mean_after = np.mean(data_after)
    std_after = np.std(data_after)
    mean_diff = mean_before - mean_after
    std_diff = std_before - std_after
    behavior_compare.at[subject_id-1, 'RT std decrease'] = std_diff
    behavior_compare.at[subject_id-1, 'RT mean shorten'] = mean_diff
    behavior_compare.at[subject_id-1, 'RT mean shorten %'] = mean_diff/mean_before*100

    median_before = np.median(data_before)
    median_after = np.median(data_after)
    median_diff = median_before - median_after
    behavior_compare.at[subject_id-1, 'RT median shorten'] = median_diff
    behavior_compare.at[subject_id-1, 'RT median shorten %'] = median_diff/median_before*100

    return behavior_compare


def allsubs_compare(subject_id, behavior_before, behavior_after, behavior_compare, experiment, verbose=False):
    for i in range(0, behavior_before.shape[0]):
        new_row = pd.DataFrame({'subject id': [subject_id],
                                'Real stimulation': [experiment.loc[experiment['subject id'] == subject_id, 'Real stimulation'].values[0]],
                                'session': ['before'],
                                'type': [behavior_before['type'].iloc[i]],
                                'cue': [behavior_before['cue'].iloc[i]],
                                'valid': [behavior_before['valid'].iloc[i]],
                                'ICS': [behavior_before['ICS'].iloc[i]],
                                'stim': [behavior_before['stimulus side'].iloc[i]],
                                'response': [behavior_before['response'].iloc[i]],
                                'reaction time': [behavior_before['reaction time'].iloc[i]]})
        behavior_compare = pd.concat([behavior_compare, new_row], ignore_index=True)


    for i in range(0, behavior_after.shape[0]):
        new_row = pd.DataFrame({'subject id': [subject_id],
                                'Real stimulation': [experiment.loc[experiment['subject id'] == subject_id, 'Real stimulation'].values[0]],
                                'session': ['before'],
                                'type': [behavior_after['type'].iloc[i]],
                                'cue': [behavior_after['cue'].iloc[i]],
                                'valid': [behavior_after['valid'].iloc[i]],
                                'ICS': [behavior_after['ICS'].iloc[i]],
                                'stim': [behavior_after['stimulus side'].iloc[i]],
                                'response': [behavior_after['response'].iloc[i]],
                                'reaction time': [behavior_after['reaction time'].iloc[i]]})
        behavior_compare = pd.concat([behavior_compare, new_row], ignore_index=True)

    return behavior_compare


def remove_outlier(df, k=1.5):
    # Assume df is your DataFrame and 'reaction time' is the column you are interested in
    Q1 = df['reaction time'].quantile(0.25)
    Q3 = df['reaction time'].quantile(0.75)
    IQR = Q3 - Q1

    # Only keep rows in dataframe that have 'reaction time' within Q1 - 1.5 IQR and Q3 + 1.5 IQR
    # filtered_df = df[~((df['reaction time'] < (Q1 - k * IQR)) |(df['reaction time'] > (Q3 + k * IQR)))]
    filtered_df = df[~(df['reaction time'] > (Q3 + k * IQR))]
    # print('Removed outliers: ' + str(len(df) - len(filtered_df)))
    return filtered_df


def add_trial_num(subject_id):
    behavior_before_path = os.path.join('..', '..', '..', 'data', str(subject_id), 'behavior_before.csv')
    behavior_before = pd.read_csv(behavior_before_path)
    behavior_after_path = os.path.join('..', '..', '..', 'data', str(subject_id), 'behavior_after.csv')
    behavior_after = pd.read_csv(behavior_after_path)
    
    # Add the trial number
    behavior_before.insert(0, 'trial', range(1, len(behavior_before) + 1))
    behavior_after.insert(0, 'trial', range(1, len(behavior_after) + 1))
    
    # Save the modified csv
    behavior_before.to_csv(behavior_before_path, index=False)
    behavior_after.to_csv(behavior_after_path, index=False)
    return True


def auto_compare(real_to_pick, sham_to_pick, watch_cases, watch_idxs):
    p_values = pd.DataFrame(index=watch_cases, columns=watch_idxs)
    for case in watch_cases:
        
        behavior_compare = create_compare()
        for subject_id in range (1,19):
            behavior_before, behavior_after = load_behavior(subject_id)
            behavior_before, behavior_after = filter_behav(case, behavior_before, behavior_after)
            behavior_compare = make_compare(subject_id, behavior_before, behavior_after, behavior_compare)
        
        behavior_compare = behavior_compare[behavior_compare['subject id'].isin(real_to_pick+sham_to_pick)]

        for idx in watch_idxs:  
            rt_diff_sham = behavior_compare.loc[behavior_compare['Real stimulation'] == 0, idx]
            rt_diff_real = behavior_compare.loc[behavior_compare['Real stimulation'] == 1, idx]
            rt_diff_sham = pd.to_numeric(rt_diff_sham)
            rt_diff_real = pd.to_numeric(rt_diff_real)

            U, p_value = stats.mannwhitneyu(rt_diff_sham, rt_diff_real)
            p_values.loc[case, idx] = p_value

    return p_values


def filter_behav(case, behavior_before, behavior_after):

    if case == 'all':
        pass
    
    elif case == 'endo':
        behavior_before = behavior_before[behavior_before['type'] == 1]
        behavior_after = behavior_after[behavior_after['type'] == 1]
    elif case == 'exo':
        behavior_before = behavior_before[behavior_before['type'] == 2]
        behavior_after = behavior_after[behavior_after['type'] == 2]

    elif case== 'valid':
        behavior_before = behavior_before[behavior_before['valid'] == 1]
        behavior_after = behavior_after[behavior_after['valid'] == 1]
    elif case == 'endo valid':
        behavior_before = behavior_before[(behavior_before['type'] == 1) & (behavior_before['valid'] == 1)]
        behavior_after = behavior_after[(behavior_after['type'] == 1) & (behavior_after['valid'] == 1)]
    elif case == 'exo valid':
        behavior_before = behavior_before[(behavior_before['type'] == 2) & (behavior_before['valid'] == 1)]
        behavior_after = behavior_after[(behavior_after['type'] == 2) & (behavior_after['valid'] == 1)]

    elif case == 'invalid':
        behavior_before = behavior_before[behavior_before['valid'] == -1]
        behavior_after = behavior_after[behavior_after['valid'] == -1]
    elif case == 'endo invalid':
        behavior_before = behavior_before[(behavior_before['type'] == 1) & (behavior_before['valid'] == -1)]
        behavior_after = behavior_after[(behavior_after['type'] == 1) & (behavior_after['valid'] == -1)]
    elif case == 'exo invalid':
        behavior_before = behavior_before[(behavior_before['type'] == 2) & (behavior_before['valid'] == -1)]
        behavior_after = behavior_after[(behavior_after['type'] == 2) & (behavior_after['valid'] == -1)]

    elif case == 'cue left':
        behavior_before = behavior_before[behavior_before['cue'] == -1]
        behavior_after = behavior_after[behavior_after['cue'] == -1]
    elif case == 'endo cue left':
        behavior_before = behavior_before[(behavior_before['type'] == 1) & (behavior_before['cue'] == -1)]
        behavior_after = behavior_after[(behavior_after['type'] == 1) & (behavior_after['cue'] == -1)]   
    elif case == 'exo cue left':
        behavior_before = behavior_before[(behavior_before['type'] == 2) & (behavior_before['cue'] == -1)]
        behavior_after = behavior_after[(behavior_after['type'] == 2) & (behavior_after['cue'] == -1)]
    
    elif case == 'cue right':
        behavior_before = behavior_before[behavior_before['cue'] == 1]
        behavior_after = behavior_after[behavior_after['cue'] == 1]  
    elif case == 'endo cue right':
        behavior_before = behavior_before[(behavior_before['type'] == 1) & (behavior_before['cue'] == 1)]
        behavior_after = behavior_after[(behavior_after['type'] == 1) & (behavior_after['cue'] == 1)]
    elif case == 'exo cue right':
        behavior_before = behavior_before[(behavior_before['type'] == 2) & (behavior_before['cue'] == 1)]
        behavior_after = behavior_after[(behavior_after['type'] == 2) & (behavior_after['cue'] == 1)]

    elif case == 'stim left':
        behavior_before = behavior_before[behavior_before['stimulus side'] == -1]
        behavior_after = behavior_after[behavior_after['stimulus side'] == -1] 
    elif case == 'endo stim left':
        behavior_before = behavior_before[(behavior_before['type'] == 1) & (behavior_before['stimulus side'] == -1)]
        behavior_after = behavior_after[(behavior_after['type'] == 1) & (behavior_after['stimulus side'] == -1)]
    elif case == 'exo stim left':
        behavior_before = behavior_before[(behavior_before['type'] == 2) & (behavior_before['stimulus side'] == -1)]
        behavior_after = behavior_after[(behavior_after['type'] == 2) & (behavior_after['stimulus side'] == -1)]

    elif case == 'stim right':
        behavior_before = behavior_before[behavior_before['stimulus side'] == 1]
        behavior_after = behavior_after[behavior_after['stimulus side'] == 1]   
    elif case == 'endo stim right':
        behavior_before = behavior_before[(behavior_before['type'] == 1) & (behavior_before['stimulus side'] == 1)]
        behavior_after = behavior_after[(behavior_after['type'] == 1) & (behavior_after['stimulus side'] == 1)]
    elif case == 'exo stim right':
        behavior_before = behavior_before[(behavior_before['type'] == 2) & (behavior_before['stimulus side'] == 1)]
        behavior_after = behavior_after[(behavior_after['type'] == 2) & (behavior_after['stimulus side'] == 1)]    

    return behavior_before, behavior_after