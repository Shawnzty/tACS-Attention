clear;
clc;

% least and most steps
fix_least = 1400;
fix_most = 1800;
stim_least = 50;
stim_most = 80;


filenames = ["eeg_before", "eeg_after"];
for i = 9:18
    folder = "../../../data/" + num2str(i) + "/";
    for j = 1:2
        disp(folder+filenames(j));
        eeg = load(folder+filenames(j)).eeg;
        trig = eeg(end,:);
        
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

       save(folder+filenames(j), "eeg");
   end
end