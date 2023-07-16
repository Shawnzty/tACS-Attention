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

## `Preprocessing` of eeg data
1. `rename_eeg`: change the name of the variable from y to eeg.
2. `trigger_align`: align the trigger to eeg to remove delay.
3. `remove_head_tail`: remove the head and tail of raw eeg.`
4. `event_channels`: add event channels to raw eeg.
5. `check_filter`: check the data and see which filter is good.
- `check_one_chan`: check the data of one channel.

## `eeg_analysis` of eeg data
1. `mkRaw`: make raw data as MNE format.
2. `icaRepair`: repair the ICA components based on ic_to_remove
3. 

- `eeg_overview`: show the overview of eeg data
- `manual_pick_ica`: manually pick ICAs to remove or keep, and save in ic_to_remove.csv


