session_names = ["eeg_before", "eeg_after"];


for sub_id = 9:18
    for session = 1:2
        filename = append("../../../data/",num2str(sub_id),"/",session_names(session))
        eeg = load(filename).y;
        save(filename, "eeg");
    end
end