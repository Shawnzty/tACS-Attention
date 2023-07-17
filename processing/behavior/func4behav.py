import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def make_compare(subject_id, behavior_compare):
    behavior_before_path = os.path.join('..', '..', '..', 'data', str(subject_id), 'behavior_before.csv')
    behavior_before = pd.read_csv(behavior_before_path)
    behavior_after_path = os.path.join('..', '..', '..', 'data', str(subject_id), 'behavior_after.csv')
    behavior_after = pd.read_csv(behavior_after_path)
    # display(behavior_before)

    # find response=1, remove too fast and outliers
    respond_trials_before = behavior_before[(behavior_before['response'] == 1) & (behavior_before['reaction time'] > 0.001)]
    respond_trials_after = behavior_after[(behavior_after['response'] == 1) & (behavior_after['reaction time'] > 0.001)]

    respond_trials_before = respond_trials_before.copy()
    respond_trials_after = respond_trials_after.copy()
    respond_trials_before.loc[:, 'reaction time'] *= 1000
    respond_trials_after.loc[:, 'reaction time'] *= 1000


    # Extract 'reaction time' column values as lists
    data_before = respond_trials_before['reaction time'].tolist()
    data_after = respond_trials_after['reaction time'].tolist()
    
    # Calculate means of data_before and data_after and add to the dataframe
    mean_before = np.mean(data_before)
    mean_after = np.mean(data_after)
    mean_diff = mean_before - mean_after
    behavior_compare.loc[behavior_compare['subject id'] == subject_id, 'RT before mean'] = mean_before
    behavior_compare.loc[behavior_compare['subject id'] == subject_id, 'RT after mean'] = mean_after
    behavior_compare.loc[behavior_compare['subject id'] == subject_id, 'RT mean shorten'] = mean_diff
    behavior_compare.loc[behavior_compare['subject id'] == subject_id, 'RT mean shorten %'] = mean_diff/mean_before*100

    median_before = np.median(data_before)
    median_after = np.median(data_after)
    median_diff = median_before - median_after
    behavior_compare.loc[behavior_compare['subject id'] == subject_id, 'RT before median'] = median_before
    behavior_compare.loc[behavior_compare['subject id'] == subject_id, 'RT after median'] = median_after
    behavior_compare.loc[behavior_compare['subject id'] == subject_id, 'RT median shorten'] = median_diff
    behavior_compare.loc[behavior_compare['subject id'] == subject_id, 'RT median shorten %'] = median_diff/median_before*100

    return behavior_compare


def remove_outlier(df):
    # Assume df is your DataFrame and 'reaction time' is the column you are interested in
    Q1 = df['reaction time'].quantile(0.25)
    Q3 = df['reaction time'].quantile(0.75)
    IQR = Q3 - Q1

    # Only keep rows in dataframe that have 'reaction time' within Q1 - 1.5 IQR and Q3 + 1.5 IQR
    filtered_df = df[~((df['reaction time'] < (Q1 - 1.5 * IQR)) |(df['reaction time'] > (Q3 + 1.5 * IQR)))]
