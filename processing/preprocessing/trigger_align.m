session_names = ["eeg_before", "eeg_after"];
advance = 58.56; % ms
sampling_freq = 1200; % Hz
advance_step = round(advance*sampling_freq/1000);


filenames = ["eeg_before", "eeg_after"];
for i = 9:18
    folder = "../../../data/" + num2str(i) + "/"
    for j = 1:2
        eeg = load(folder+filenames(j)).eeg;
        trigger = eeg(end,:);
        trigger = trigger(1:end-advance_step);
        trigger = [zeros(1,advance_step), trigger];
        eeg(end,:) = trigger;
        save(folder+filenames(j), "eeg");
    end
end