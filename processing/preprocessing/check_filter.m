clear;
clc;
close all;

session_name = ["before","after"];

% get the size and position of each monitor
monitors = get(0, 'MonitorPositions');
monitor_shift = [-110 50]; % adjust this value to fit your screen
taskbar_shift = [60 50];

Fs = 1200; % Sample rate

% Design a Notch filter: quality factor
F1 = 50;
q1 = 20;
bw1 = 2; %F1/q1; % Bandwidth
[b1,a1] = iirnotch(F1/(Fs/2),bw1/(Fs/2));

% Design a Notch filter: quality factor = 10
F2 = 100;
q2 = 20;
bw2 = 4; % F2/q2; % a bandwidth factor of 5Hz
[b2,a2] = iirnotch(F2/(Fs/2),bw2/(Fs/2));

% Design a Notch filter: quality factor = 10
F3 = 150;
q3 = 6;
bw3 = 3; % F3/q3; % a bandwidth factor of 5Hz
[b3,a3] = iirnotch(F3/(Fs/2),bw3/(Fs/2));

for sub_id = 12
for session = 1:2
filename = append("../../../data/",num2str(sub_id),"/eeg_",session_name(session));
eeg = load(filename).eeg;
time = eeg(1,:);

% choose the screen size based on the session number
screen_size = monitors(session, 3:4);
fig_width = screen_size(1) / 8; % divide the screen width by the number of columns
fig_height = (screen_size(2) - taskbar_shift(session)) / 4; % divide the screen height (minus taskbar height) by the number of rows
left_offset = monitors(session, 1); % add offset depending on the monitor

for channel = 2:33
    make_title = append(num2str(sub_id),", ", session_name(session),", ",num2str(channel-1));
    
    data = eeg(channel,:);
    data_filter = lowpass(data,100,Fs);

    data_filter = filtfilt(b1,a1,data_filter); % Apply Notch filter
    data_filter = filtfilt(b2,a2,data_filter);
    data_filter = filtfilt(b3,a3,data_filter);

    f = figure('Position', [(mod(channel-2,8)*fig_width)+left_offset (floor((channel-2)/8)*fig_height)+monitor_shift(session) fig_width fig_height]); % calculate the position based on the figure index

    subplot(2,1,1);
    plot(time, data);
    hold on;
    plot(time, data_filter);
    title(make_title);
    ylabel([num2str(channel)]);
    ylim([-300 300]);
    
    subplot(2,1,2);
    [freq,db] = getPSD(data,Fs);
    plot(freq,db);
    hold on;
    [freq,db] = getPSD(data_filter,Fs);
    plot(freq,db);

    grid on
    title("Periodogram Using FFT")
    xlabel("Frequency (Hz)")
    ylabel("Power/Frequency (dB/Hz)")
    xlim([0 200]);
    ylim([-50 50]);
    
end

end
end
