import os
from scipy.io import loadmat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# the following import is required for matplotlib < 3.2:
from mpl_toolkits.mplot3d import Axes3D  # noqa
import mne


def icarepair(subject_id, before_or_after, df):
    # load original raw
    raw_path = os.path.join('..', '..', '..', 'data', str(subject_id), 'raw_' + before_or_after + '.fif')
    raw = mne.io.read_raw_fif(raw_path, preload=True)

    # Get the remove_ic
    selected_row = df.loc[(df['subject_id'] == subject_id) & (df['before_or_after'] == before_or_after)]
    remove_ic = selected_row['remove_ic'].iloc[0].split(',')

    # ica
    raw_filtered = raw.copy() # raw already filtered
    n_components = 32  # Number of ICA components (adjust as needed)
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, method='picard')
    ica.fit(raw_filtered)

    # # epochs (maybe not necessary)
    # stim_channel_name = 'fixation'
    # event_id = {'fixation': 1}
    # events = mne.find_events(raw, stim_channel=stim_channel_name, min_duration=1/raw.info['sfreq'])
    # tmin, tmax = -1.5, 3.58  # Define the time range of epochs 3.583s-5.05s
    # epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=(0, 0), preload=True)
    # ica_epochs = ica.apply(epochs.copy(), exclude=ica.exclude)

    # Remove the components
    ica.exclude = [int(i) for i in remove_ic]
    raw_ica_applied = raw.copy()
    ica.apply(raw_ica_applied)

    # Save the repaired raw
    raw_save_path = os.path.join('..', '..','..', 'data', str(subject_id), 'repaired_' + before_or_after + '_raw' + '.fif')
    raw_ica_applied.save(raw_save_path, overwrite=True)


# main
# Load the CSV file
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
csv_file = 'ic_to_remove.csv'
df = pd.read_csv(csv_file)
for subject_id in range (1,19):
    for before_or_after in ['before', 'after']:
        icarepair(subject_id, before_or_after, df)