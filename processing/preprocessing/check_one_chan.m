clear;
clc;

sub_id = 12;
session = 'before';
filename = append("../../../data/",num2str(sub_id),"/eeg_",session);
eeg = load(filename).eeg;

plot(eeg(1,:), eeg(10,:));
hold on;
plot(eeg(1,:), eeg(34,:));

