subject_num = 8;

advance = 58.56; % ms
sampling_freq = 1200; % Hz
advance_step = round(advance*sampling_freq/1000);

% eeg = load("../../data/1/eeg_before").y;
% eeg(end,1) = 10000;
% trigger = eeg(end,:);
% trigger = trigger(1:end-advance_step);
% trigger = [zeros(1,advance_step), trigger];
% eeg(end,:) = trigger;

filenames = ["eeg_before", "eeg_after"];
for i = 1:subject_num
    folder = "../../data/" + num2str(i) + "/";
    for j = 1:2
        eeg = load(folder+filenames(j)).y;
        trigger = eeg(end,:);
        trigger = trigger(1:end-advance_step);
        trigger = [zeros(1,advance_step), trigger];
        eeg(end,:) = trigger;
        save(folder+filenames(j), "eeg");
    end
end