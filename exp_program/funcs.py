import numpy as np
from psychopy import core, visual, gui, data, event
from settings import *

def make_trials(num1, code1, num2, code2):
    trial = np.concatenate((code1*np.ones(num1, dtype=int), code2*np.ones(num2, dtype=int)))
    np.random.shuffle(trial)
    return trial

def fix(mywin, fixation, fix_time):
    fixation.draw()
    mywin.flip()
    core.wait(fix_time)

def endo(mywin, fixation, arrow, cue, valid, stim, cue_time, stim_time, res_time):
    
