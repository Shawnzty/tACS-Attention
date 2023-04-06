import numpy as np
import mne
import matplotlib.pyplot as plt
import os

subject_id = 4

current_script_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_script_folder, '..', '..', 'data')
subject_folder = os.path.join(data_folder, str(subject_id))
raw_load_path = os.path.join(subject_folder, 'raw.fif')
raw = mne.io.read_raw_fif(raw_load_path, preload=True)

# Select a single channel
channel_name = 'fixation'  # Replace with the desired channel name
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
