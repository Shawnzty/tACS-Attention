function [freq,db] = getPSD(x,fs)
%GETPSD Summary of this function goes here
%   Detailed explanation goes here

N = length(x);
xdft = fft(x);
xdft = xdft(1:N/2+1);
psdx = (1/(fs*N)) * abs(xdft).^2;
psdx(2:end-1) = 2*psdx(2:end-1);
freq = 0:fs/length(x):fs/2;

db = pow2db(psdx);
plot(freq,db)
end

