clear;
clc;

sub_id = 7;
session = 'before';
filename = append("../../../data/",num2str(sub_id),"/eeg_",session);
eeg = load(filename).eeg;

plot(eeg(1,:), eeg(34,:));
hold on;
plot(eeg(1,:), eeg(35,:));
plot(eeg(1,:), eeg(36,:));
plot(eeg(1,:), eeg(45,:));
% plot(eeg(1,:), eeg(56,:));
