import numpy as np
import mne
import matplotlib.pyplot as plt

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = (sample_data_folder / 'MEG' / 'sample' /
                        'sample_audvis_filt-0-40_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)

import matplotlib.pyplot as plt

# Select a single channel
channel_name = 'STI 014'  # Replace with the desired channel name
channel_idx = raw.ch_names.index(channel_name)

# Extract the channel data and the times
data, times = raw[channel_idx, :]
data = data.squeeze()

fig, ax = plt.subplots()

# Create line plot
ax.plot(times, data, color='blue', alpha=0.5)

# Create scatter plot
ax.scatter(times, data, s=5, color='red', picker=5)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (uV)')
ax.set_title(f'Signal from channel {channel_name}')

# Define the callback function for the click event
def on_pick(event):
    ind = event.ind[0]  # Get the index of the selected data point
    x = times[ind]
    y = data[ind]
    print(f'Value at time {x:.2f}s: {y:.2f} uV')

# Connect the callback function to the pick event
fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()
