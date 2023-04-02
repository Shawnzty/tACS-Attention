subject_num = 8;

advance = 58.56; % ms
sampling_freq = 1200; % Hz
advance_step = round(advance*sampling_freq/1000);

file = "../../data/7/eeg_after";
eeg = load(file).y;
trigger = eeg(end,:);
trigger = trigger(1:end-advance_step);
trigger = [zeros(1,advance_step), trigger];
eeg(end,:) = trigger;
save(file, "eeg");