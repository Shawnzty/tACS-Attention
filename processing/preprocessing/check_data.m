clear;
clc;

sub_id = 9;
session = 'before';
filename = append("../../../data/",num2str(sub_id),"/eeg_",session);
eeg = load(filename).eeg;

time = eeg(1,:);
figure();

for channel = 2:34
    subplot(33,1,channel-1);
    subplot(size(eeg, 1) - 1, 1, channel - 1);
    plot(time, eeg(channel,:));
    ylabel([num2str(channel)]);
    axis tight; % autoscale
end
