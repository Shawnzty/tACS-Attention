import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

for subject_id in range(1,2):
    behavior_before_path = os.path.join('..', '..', '..', 'data', str(subject_id), 'behavior_before.csv')
    behavior_before = pd.read_csv(behavior_before_path)
    behavior_after_path = os.path.join('..', '..', '..', 'data', str(subject_id), 'behavior_after.csv')
    behavior_after = pd.read_csv(behavior_after_path)
    
    # Add the trial number
    behavior_before.insert(0, 'trial', range(1, len(behavior_before) + 1))
    behavior_after.insert(0, 'trial', range(1, len(behavior_after) + 1))
    
    # Save the modified csv
    save_before = os.path.join('..', '..', '..', 'data', str(subject_id), 'behavior_before_addtrial.csv')
    save_after = os.path.join('..', '..', '..', 'data', str(subject_id), 'behavior_after_addtrial.csv')
    behavior_before.to_csv(behavior_before_path, index=False)
    behavior_after.to_csv(behavior_after_path, index=False)
