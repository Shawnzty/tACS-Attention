% eeg = load("../../data/4/eeg_after").eeg;
% trig = eeg(end,:);

subject_num = 8;
% least and most steps
fix_least = 1700;
fix_most = 1900;
stim_least = 60;
stim_most = 90;

% for i=1:length(trig)
%     if trig(i) == 8 && trig(i + fix_least) == 8
%         disp(i)
%         break
%     else
%         continue
%     end
% end
% 
% 
% for j=length(trig):-1:1
%     if trig(j) == 8 && trig(j-stim_least) == 8
%         disp(j)
%         break
%     else
%         continue
%     end
% end
% 
% head = i - 1200;
% tail = j + 3600;
% trig = trig(head:tail);

filenames = ["eeg_before", "eeg_after"];
for i = 1:subject_num
    folder = "../../data/" + num2str(i) + "/";
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
        eeg = eeg(:,head:tail);

        % save(folder+filenames(j), "eeg");
    end
end