behavior_file = "../../../data/1/behavior_before";
behavior = readmatrix(behavior_file);
eeg_file = "../../../data/1/eeg_before";
eeg = load(eeg_file).eeg;

trigger = eeg(end,:);
sampling_freq = 1200;
fix_least = 1800;
fix_most = 1900;
stim_least = 65;
stim_most = 85;

% unit in steps
fix = 1.5 * sampling_freq;
endo = 1 * sampling_freq;
exo = round(0.033 * 4 * sampling_freq);
ics_f = 0.5 * sampling_freq;
ics_s = 1 * sampling_freq;
stim_t = round(0.05 * sampling_freq);


% create containers
fixation = zeros(1,length(trigger));

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
    event = bahvaior(k,:);
    % find fixation
    if fixation(j) == 1
        % GOTO cue
        j = j+fix;
        % mark cue
        if event(1) == 1
            % endo
            if event(2) == -1
                endo_left(j) = 1;
            elseif event(2) == 1
                endo_right(j) = 1;
            end
        elseif event(1) == 2
            % exo
            if event(2) == -1
                exo_left(j) = 1;
            elseif event(2) == 1
                exo_right(j) = 1;
            end
        end
        
        % mark valid
        if event(3) == 1
            valid(j) = 1;
        elseif event(3) == -1
            invalid(j) = 1;
        end
        
        % GOTO ics
        if event(1) == 1
            j = j + endo;
        elseif event(1) == 2
            j = j + exo;
        end

        % mark ics
        if event(4) == 0.5
            ics_fast(j) = 1;
            % j = j + ics_f;
        elseif event(4) == 1
            ics_slow(j) = 1;
            % j = j + ics_s;
        end
    end

    % find stim
    if stim(j) == 1
        if event(5) == -1
            stim_left(j) = 1;
        elseif event(5) == 1
            stim_right(j) = 1;
        end
        
        % mark stim x
        if event(6) == 969
            stim_close(j) = 1;
        elseif event(6) == 1131
            stim_xmiddle(j) = 1;
        elseif event(6) == 1292
            stim_far(j) = 1;
        end

        % mark stim y
        if event(7) == -220
            stim_lowest(j) = 1;
        elseif event(7) == -73
            stim_lower(j) = 1;
        elseif event(7) == 0
            stim_ymiddle = 1;
        elseif event(7) == 73
            stim_higher = 1;
        elseif event(7) == 220
            stim_highest = 1;
        end
        
        % GOTO wait response
        j = j + stim_t;
        % mark response
        if event(8) == 1 && event(9) > 0.001
            response(j+round(event(9)*sampling_freq)) = 1;
        end
    end
end
% figure();
% plot(fixation);
% hold on;
% plot(stim);
