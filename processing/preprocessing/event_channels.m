% subject_num = 8;
eeg_filenames = ["eeg_before", "eeg_after"];
behavior_filenames = ["behavior_before", "behavior_after"];
% save_filenames = ["eeg_before_addstim", "eeg_after_addstim"];

for subject = 1:18
    folder = "../../../data/" + num2str(subject) + "/";
    for trial = 1:2
        disp(folder+eeg_filenames(trial));
        % load data
        eeg_origin = load(folder+eeg_filenames(trial)).eeg;
        behavior = readmatrix(folder+behavior_filenames(trial));


trigger = eeg_origin(34,:);
sampling_freq = 1200;
fix_least = 1500;
fix_most = 1800;
stim_least = 50;
stim_most = 80;

% unit in steps
fix = 1.5 * sampling_freq;
endo = 1 * sampling_freq;
exo = round(0.033 * 4 * sampling_freq);
ics_f = 0.5 * sampling_freq;
ics_s = 1 * sampling_freq;
stim_t = round(0.05 * sampling_freq);


% create containers
fixation = zeros(1,length(trigger));

cue = zeros(1,length(trigger));
endo_left = zeros(1,length(trigger));
endo_right = zeros(1,length(trigger));
exo_left = zeros(1,length(trigger));
exo_right = zeros(1,length(trigger));

valid = zeros(1,length(trigger));
invalid = zeros(1,length(trigger));

ics_fast = zeros(1,length(trigger));
ics_slow = zeros(1,length(trigger));

stim = zeros(1,length(trigger));
stim_left = zeros(1,length(trigger));
stim_right = zeros(1,length(trigger));

stim_close = zeros(1,length(trigger));
stim_xmiddle = zeros(1,length(trigger));
stim_far = zeros(1,length(trigger));

stim_highest = zeros(1,length(trigger));
stim_higher = zeros(1,length(trigger));
stim_ymiddle = zeros(1,length(trigger));
stim_lower = zeros(1,length(trigger));
stim_lowest = zeros(1,length(trigger));

response = zeros(1,length(trigger));


% mark start of fixation and stim
i = 1;
while i <= length(trigger)
    if trigger(i) == 8 && trigger(i+fix_least) == 8 ...
            && trigger(i+fix_most) == 0
        fixation(i) = 1;
        i = i + fix_most;
    end
    if trigger(i) == 8 && trigger(i+stim_least) == 8 ...
            && trigger(i+stim_most) == 0
        stim(i) = 1;
        i = i + stim_most;
    end
    i = i + 1;
end

% mark others
j = 1; k = 1;
while j <= length(trigger)
    event = behavior(k,:);
    % find fixation
    if fixation(j) == 1
        % disp(k);
        % GOTO cue
        j = j+fix;
        cue(j) = 1;
        % mark cue
        if event(2) == 1
            % endo
            if event(3) == -1
                endo_left(j) = 1;
            elseif event(3) == 1
                endo_right(j) = 1;
            end
        elseif event(2) == 2
            % exo
            if event(3) == -1
                exo_left(j) = 1;
            elseif event(3) == 1
                exo_right(j) = 1;
            end
        end
        
        % mark valid
        if event(4) == 1
            valid(j) = 1;
        elseif event(4) == -1
            invalid(j) = 1;
        end

        % GOTO ics
        if event(2) == 1
            j = j + endo;
        elseif event(2) == 2
            j = j + exo;
        end

        % mark ics
        if event(5) == 0.5
            ics_fast(j) = 1;
        elseif event(5) == 1
            ics_slow(j) = 1;
        end
    end

    % find stim
    if stim(j) == 1
        % disp(k);
        if event(6) == -1
            stim_left(j) = 1;
        elseif event(6) == 1
            stim_right(j) = 1;
        end
        
        % mark stim x
        if event(7) == 969
            stim_close(j) = 1;
        elseif event(7) == 1131
            stim_xmiddle(j) = 1;
        elseif event(7) == 1292
            stim_far(j) = 1;
        end

        % mark stim y
        if event(8) == -220
            stim_lowest(j) = 1;
        elseif event(8) == -73
            stim_lower(j) = 1;
        elseif event(8) == 0
            stim_ymiddle(j) = 1;
        elseif event(8) == 73
            stim_higher(j) = 1;
        elseif event(8) == 220
            stim_highest(j) = 1;
        end
        
        % GOTO wait response
        j = j + stim_t;
        % mark response
        if event(9) == 1 && event(10) > 0.01
            response(j+round(event(10)*sampling_freq)) = 1;
        end

        % next trial
        if k<size(behavior,1) k = k + 1; end
    end
    j = j + 1;
end

% concatenate
eeg = zeros(56,size(eeg_origin,2));
eeg(1:34,:) = eeg_origin(1:34,:);

eeg(35,:) = fixation;

eeg(36,:) = cue;
eeg(37,:) = endo_left;
eeg(38,:) = endo_right;
eeg(39,:) = exo_left;
eeg(40,:) = exo_right;

eeg(41,:) = valid;
eeg(42,:) = invalid;

eeg(43,:) = ics_fast;
eeg(44,:) = ics_slow;

eeg(45,:) = stim;
eeg(46,:) = stim_left;
eeg(47,:) = stim_right;

eeg(48,:) = stim_close;
eeg(49,:) = stim_xmiddle;
eeg(50,:) = stim_far;

eeg(51,:) = stim_highest;
eeg(52,:) = stim_higher;
eeg(53,:) = stim_ymiddle;
eeg(54,:) = stim_lower;
eeg(55,:) = stim_lowest;

eeg(56,:) = response;

% save eeg data
save(folder+eeg_filenames(trial), "eeg");

    end
end
% figure();
% plot(fixation);
% hold on;
% plot(endo_left); plot(endo_right); plot(exo_left); plot(exo_right);
% plot(valid); plot(invalid); plot(ics_fast); plot(ics_slow);
% plot(stim); plot(stim_left); plot(stim_right);
% plot(stim_close); plot(stim_xmiddle); plot(stim_far);
% plot(stim_highest); plot(stim_higher); plot(stim_ymiddle);
% plot(stim_lower); plot(stim_lowest);
% plot(response);
