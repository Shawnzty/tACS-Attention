# tes-attention

exp_program is the main program to run the experiment.

processing is the program to process the data.
prepare_data is the MATLAB program to prepare the data for MNE-Pyhton.

## Analysis of behavior data
- behavior is the program to analyze and plot the behavior data.
- Behavior data has been removed from repository.

## Analysis of EEG data
eeg_overview is the program to plot the overview of the EEG data. (not used any more)
1. mkRaw_ica: make raw data and apply ICA
2. manual_pick_ica and ic_to_remove: are used for manual picking of ICA components
3. icaRepair: repair the ICA components based on ic_to_remove

## Preprocessing of eeg data
1. 'rename_eeg': change the name of the variable from y to eeg.
2. 'check_data': check the data of raw eeg.
3.  
