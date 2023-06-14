# All parameters for experiment is defined here
# Author: Tianyi Zheng
import math
import numpy as np

# display settings
# 1512 982 for mbp14
# 3840 2160 for dell
# 5120 1440 for Philips
screen_width = 5120
screen_width_mm = 1193.5
mm_pix = screen_width / screen_width_mm
screen_height = 1440

# visual settings
distance = 580 # distance between screen and participant in unit of mm
FoV_degree = 60 # field of view in unit of degree
rf_FoV = 10 # field of sharp view

alpha = math.radians(FoV_degree)
gamma = math.radians(rf_FoV)
beta = 0.5*alpha -  np.arcsin(((1800-distance)/1800) * math.sin(math.pi - 0.5*alpha))
phi = 0.5*alpha - gamma - np.arcsin(((1800-distance)/1800) * math.sin(math.pi - 0.5*alpha + gamma))

l1 = phi*1800*mm_pix
l2 = beta*1800*mm_pix

# arrow
arrow_left = ((0, 15), (-80, 15), (-80, 40), (-140, 0), (-80, -40), (-80, -15), (0, -15))
arrow_right = ((0, 15), (80, 15), (80, 40), (140, 0), (80, -40), (80, -15), (0, -15))

# trigger_flash
trigger_sizex = 30
trigger_sizey = 60
trigger_ypos = -1*(screen_height/2 - trigger_sizey/2 - 100)

# test
test_endo_trials = 4
test_exo_trials = 4
test_val_ratio = 0.75

# experiment
fix_time = 1.5
val_ratio = 0.8

# receptive field
rf_size = l2 - l1
rf_pos = 0.5*(l1 + l2)

# stimulus
stim_time = 0.05
stimulus_FoV = 2 # in degree
psi = 0.5*(alpha - gamma) - np.arcsin((1800-distance)/1800 * math.sin(math.pi - 0.5*(alpha - gamma)))
dr = (math.sin(psi) / math.sin(math.pi - 0.5*(alpha - gamma))) * 1800
stimulus_size = math.radians(stimulus_FoV) * dr
stimulus_pos_a = ((l2 - l1) / 6) + l1
stimulus_pos_b = ((l2 - l1) / 2) + l1
stimulus_pos_c = ((l2 - l1) * 5 / 6) + l1
stimulus_pos_up = ((l2 - l1) / 2) - stimulus_size
stimulus_pos_down = -1 * stimulus_pos_up

# ics: interval between cue and stimulus
ics_fast = 0.5
ics_slow = 1

endo_trials = 60
endo_cue_time = 1
endo_res = 1.5

exo_trials = 60
exo_cue_flash = 2
exo_cue_flash_ontime = 0.033
exo_cue_flash_offtime = 0.033
exo_res = 1.5

# text
text_size = 35

# Finish