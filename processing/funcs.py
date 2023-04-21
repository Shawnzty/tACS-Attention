import os
from scipy.io import loadmat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# the following import is required for matplotlib < 3.2:
from mpl_toolkits.mplot3d import Axes3D  # noqa
import mne

def mkraw(subject_id):
    # read the EEG data
    subject_id = 1
    eeg_path = os.path.join('..', '..', 'data', str(subject_id), 'eeg_before.mat')
    eeg_before = loadmat(eeg_path)['eeg']
    eeg_before[1:33] *= 1e-6 # convert to microvolts

    # read the behavior data
    behavior_before_path = os.path.join('..', '..', 'data', str(subject_id), 'behavior_before.csv')
    behavior_before = pd.read_csv(behavior_before_path)
    behavior_after_path = os.path.join('..', '..', 'data', str(subject_id), 'behavior_after.csv')
    behavior_after = pd.read_csv(behavior_after_path)