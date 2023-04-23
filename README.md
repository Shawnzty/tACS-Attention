# tes-attention

exp_program is the main program to run the experiment.

processing is the program to process the data.
prepare_data is the MATLAB program to prepare the data for MNE-Pyhton.

## Analysis of behavior data
behavior is the program to analyze and plot the behavior data.

## Analysis of EEG data
eeg_overview is the program to plot the overview of the EEG data. (not used any more)
1. mkRaw_ica: make raw data and apply ICA
2. manual_pick_ica and ic_to_remove: are used for manual picking of ICA components
3. icaRepair: repair the ICA components based on ic_to_remove
