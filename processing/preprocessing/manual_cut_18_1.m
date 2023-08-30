clear;
clc;

% least and most steps
fix_least = 1400;
fix_most = 1800;
stim_least = 50;
stim_most = 80;

filename = "../../../data/18/eeg_before";
eeg = load(filename).eeg;
eeg = eeg(:,35000:end);

trig = eeg(34,:);

% find first fix
for head=1:length(trig)
if trig(head) == 8 && trig(head + fix_least) == 8 ...
        && trig(head + fix_most) == 0
    disp(head);
    break
else
    continue
end
end

% find last stim
for tail=length(trig):-1:1
if trig(tail) == 8 && trig(tail-stim_least) == 8 ...
        && trig(tail-stim_most) == 0
    disp(tail);
    break
else
    continue
end
end
% add some buffer
head = head - 1200;
tail = tail + 3600;
disp(tail-head);
eeg = eeg(:,head:tail);
plot(eeg(34,:))
save(filename, "eeg");