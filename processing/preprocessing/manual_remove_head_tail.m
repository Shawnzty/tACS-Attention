% least and most steps
fix_least = 1700;
fix_most = 1900;
stim_least = 60;
stim_most = 90;

eeg = load("../../../data/6/eeg_after").eeg;
trig = eeg(end,:);

for head=110000:length(trig)
    if trig(head) == 8 && trig(head + fix_least) == 8 ...
            && trig(head + fix_most) == 0
        disp(head);
        break
    else
        continue
    end
end


for tail=length(trig):-1:1
    if trig(tail) == 8 && trig(tail - stim_least) == 8 ...
            && trig(head - stim_most) == 0
        disp(tail);
        break
    else
        continue
    end
end

head = head - 1200;
tail = tail + 3600;
eeg = eeg(:,head:tail);
save("../../../data/6/eeg_after", "eeg");
figure();
plot(eeg(end,:));

eeg = load("../../../data/7/eeg_after").eeg;
trig = eeg(end,:);
for head=270000:length(trig)
    if trig(head) == 8 && trig(head + fix_least) == 8 ...
            && trig(head + fix_most) == 0
        disp(head);
        break
    else
        continue
    end
end


for tail=length(trig):-1:1
    if trig(tail) == 8 && trig(tail - stim_least) == 8 ...
            && trig(head - stim_most) == 0
        disp(tail);
        break
    else
        continue
    end
end

head = head - 1200;
tail = tail + 3600;
eeg = eeg(:,head:tail);
save("../../../data/7/eeg_after", "eeg");
figure();
plot(eeg(end,:));