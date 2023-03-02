# All parameters for experiment is defined here
# Author: Tianyi Zheng
import math
import numpy as np

# display settings
# 1512 982 for mbp14
# 3840 2160 for dell
# 5129 1440 for Philips
screen_width = 5120
screen_width_mm = 1193.5
screen_height = 1440
distance = 480 # distance between screen and participant in unit of mm
FoV = 90 # field of view in unit of degree
beta = 0.5*FoV -  (180/math.pi) * np.arcsin(((1800-distance)/1800) * math.sin(math.radians(180-0.5*FoV))) # beta is the angle between the center of the screen and the center of the RF
l = 10 * math.pi * beta

# receptive field
rf_size = 300
rf_pos = l*screen_width/screen_width_mm

# arrow
arrow_left = ((0, 15), (-80, 15), (-80, 40), (-140, 0), (-80, -40), (-80, -15), (0, -15))
arrow_right = ((0, 15), (80, 15), (80, 40), (140, 0), (80, -40), (80, -15), (0, -15))

# trigger
trigger_sizex = 30
trigger_sizey = 60
trigger_ypos = -1*(screen_height/2 - trigger_sizey/2 - 100)

# test
test_endo_trials = 4
test_exo_trials = 4
test_val_ratio = 0.75

# experiment
fix_time = 2
val_ratio = 0.8
stimulus_FoV = 2 # in degree
dr = (math.sin(math.radians(beta)) / math.sin(math.radians(180-0.5*FoV))) * 1800
stimulus_size = math.radians(stimulus_FoV) * dr
stimulus_pos = rf_pos

endo_trials = 50
endo_cue_time = 1
endo_stim_time = 0.05
endo_ics = 1
endo_res = 2

exo_trials = 50
exo_cue_flash = 2
exo_cue_flash_ontime = 0.1
exo_cue_flash_offtime = 0.1
exo_stim_time = 0.1
exo_ics = 0.5
exo_res = 2

# text
text_size = 35

# Finish