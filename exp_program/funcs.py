import numpy as np
from psychopy import core, visual, gui, data, event
from settings import *
import serial
import time

def generate_all_trials(endo_trials, exo_trials, val_ratio):

    cue_type = make_trial(endo_trials, 1, exo_trials, 2)

    endo_cue = make_trial(round(endo_trials/2), 1, round(endo_trials/2), -1)
    exo_cue = make_trial(round(exo_trials/2), 1, round(exo_trials/2), -1)
    all_cue = np.concatenate((endo_cue, exo_cue))

    valid_unit = make_trial(round(endo_trials*val_ratio/2), 1, round(endo_trials*(1-val_ratio)/2), -1)
    all_valid = np.concatenate((valid_unit, valid_unit, valid_unit, valid_unit))

    all_stim = np.multiply(all_cue, all_valid)

    endo_ics_unit = np.concatenate((np.linspace(endo_ics_min, endo_ics_max, num=round(endo_trials*val_ratio/2)), 
                                np.linspace(endo_ics_min, endo_ics_max, num=round(endo_trials*(1-val_ratio)/2))))
    endo_ics = np.concatenate((endo_ics_unit, endo_ics_unit))
    exo_ics_unit = np.concatenate((np.linspace(exo_ics_min, exo_ics_max, num=round(exo_trials*val_ratio/2)), 
                                np.linspace(exo_ics_min, exo_ics_max, num=round(exo_trials*(1-val_ratio)/2))))
    exo_ics = np.concatenate((exo_ics_unit, exo_ics_unit))
    all_ics = np.concatenate((endo_ics, exo_ics))

    all_trials = np.vstack((cue_type, all_cue, all_valid, all_stim, all_ics)).T
    np.random.shuffle(all_trials)
    return all_trials

def make_trial(num1, code1, num2, code2):
    trial = np.concatenate((code1*np.ones(num1, dtype=int), code2*np.ones(num2, dtype=int)))
    return trial.tolist()

def fix(mywin, fixation, fix_time, left_rf, right_rf, trigger):
    fixation.draw()
    left_rf.draw()
    right_rf.draw()
    trigger.write(b'H')
    mywin.flip()
    core.wait(fix_time)
    trigger.write(b'L')


def endo(mywin, fixation, left_rf, right_rf, arrow, stimulus, trigger, cue, stim, ics):
    
    if cue == -1:
        arrow.setVertices(arrow_left)
    else:
        arrow.setVertices(arrow_right)
    stimulus.setPos((stim*stimulus_pos, 0))

    # draw cue and flip window
    left_rf.draw()
    right_rf.draw()
    arrow.draw()
    mywin.flip()
    core.wait(endo_cue_time)

    # wait for stimulus
    left_rf.draw()
    right_rf.draw()
    fixation.draw()
    mywin.flip()
    core.wait(ics)

    # draw stimulus and flip window
    left_rf.draw()
    right_rf.draw()
    fixation.draw()
    stimulus.draw()
    trigger.write(b'H')
    mywin.flip()
    core.wait(stim_time)
    trigger.write(b'L')

    # wait for response
    left_rf.draw()
    right_rf.draw()
    fixation.draw()
    mywin.flip()
    response = 0
    reaction_time = 0
    rt_clock = core.Clock()
    while rt_clock.getTime() < endo_res:
        keys = event.getKeys(keyList=['space'])
        if keys:
            response = 1
            reaction_time = rt_clock.getTime()

    return response, reaction_time


def exo(mywin, fixation, left_rf, right_rf, stimulus, trigger, exo_rect, cue, stim, ics):
    
    exo_rect.setPos((cue*rf_pos, 0))
    stimulus.setPos((stim*stimulus_pos, 0))

    # draw cue and flip window
    if cue == 1:
        show_exo_cue(mywin, fixation, left_rf, exo_rect)
    else:
        show_exo_cue(mywin, fixation, right_rf, exo_rect)

    # wait for stimulus
    left_rf.draw()
    right_rf.draw()
    fixation.draw()
    mywin.flip()
    core.wait(ics)

    # draw stimulus and flip window
    left_rf.draw()
    right_rf.draw()
    fixation.draw()
    stimulus.draw()
    trigger.write(b'H')
    mywin.flip()
    core.wait(stim_time)
    trigger.write(b'L')


    # wait for response
    left_rf.draw()
    right_rf.draw()
    fixation.draw()
    mywin.flip()
    response = 0
    reaction_time = 0
    rt_clock = core.Clock()
    while rt_clock.getTime() < exo_res:
        keys = event.getKeys(keyList=['space'])
        if keys:
            response = 1
            reaction_time = rt_clock.getTime()

    return response, reaction_time

def show_exo_cue(mywin, fixation, rf, exo_rect):
    for i in range(exo_cue_flash):
        rf.draw()
        fixation.draw()
        exo_rect.draw()
        mywin.flip()
        core.wait(exo_cue_flash_ontime)
        
        rf.draw()
        fixation.draw()
        mywin.flip()
        core.wait(exo_cue_flash_offtime)

def start(mywin, expInfo):
    # display instructions and wait
    message1 = visual.TextStim(mywin, pos=[0,+100],
        text="Hello, " + expInfo['Name'] + "!\n Welcome to the experiment.")
    message2 = visual.TextStim(mywin, pos=[0,-100],
                               text='Please hit the space to start.')
    message1.size = [None, 35]
    message2.setSize(text_size)
    message1.draw()
    message2.draw()
    mywin.flip()
    event.waitKeys(keyList=['space'])


def finish(mywin, expInfo):
    # display end message and quit
    message1 = visual.TextStim(mywin, pos=[0,0],
        text="Thank you for participating, " + expInfo['Name'] + "!")
    message1.setSize(text_size)
    message1.draw()
    mywin.flip()
    core.wait(2)
