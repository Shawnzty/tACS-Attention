subject_num = 8;
behavior_filenames = ["behavior_before", "behavior_after"];

for subject = 1:subject_num
    folder = "../../../data/" + num2str(subject) + "/";
    for trial = 1:2
        behavior = readmatrix(folder+behavior_filenames(trial));

        
    end
end