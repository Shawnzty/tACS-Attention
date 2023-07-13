clear;
clc;

eeg_filenames = ["eeg_before", "eeg_after"];
behavior_filenames = ["behavior_before", "behavior_after"];
save_filenames = ["eeg_before_addstim", "eeg_after_addstim"];

for subject = 1:8
    folder = "../../../data/" + num2str(subject) + "/"
    for trial = 1:2
        eeg_1 = load(folder+eeg_filenames(trial)).eeg;
        eeg_2 = load(folder+save_filenames(trial)).eeg;
        
        eeg_1 = eeg_1(35:55,:);
        eeg_2 = [eeg_2(35,:); eeg_2(37:56,:)];
        diff = eeg_1 - eeg_2;
        disp(sum(sum(eeg_1-eeg_2)));
    end
end