% eegdata = rand(32, 256*100); % 32 channels of random activity (100 s sampled at 256 Hz).
% eegdata(33,[10:256:256*100]) = 1; % simulating a stimulus onset every second
% eegdata(33,[100:256:256*100]+round(rand*128)) = 2; % simulating reaction times about 500 ms after stimulus onsets

chanlocs = struct('labels', { 'cz' 'c3' 'c4' 'pz' 'p3' 'p4' 'fz' 'f3' 'f4'});
pop_chanedit( chanlocs );