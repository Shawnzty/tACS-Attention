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


    ics_unit = np.concatenate((make_trial(round((endo_trials*val_ratio)/4), ics_slow, round((endo_trials*val_ratio)/4), ics_fast), 
                               make_trial(round((endo_trials*(1-val_ratio))/4), ics_slow, round((endo_trials*(1-val_ratio))/4), ics_fast)))
    all_ics = np.concatenate((ics_unit, ics_unit, ics_unit, ics_unit))

    stim_side = np.multiply(all_cue, all_valid)

    stim_x_unit1 = np.concatenate((stimulus_pos_a*np.ones(round(endo_trials*val_ratio/12)), 
                                   stimulus_pos_b*np.ones(round(endo_trials*val_ratio/12)), 
                                   stimulus_pos_c*np.ones(round(endo_trials*val_ratio/12))))
    
    stim_x_unit2 = np.concatenate((stimulus_pos_a*np.ones(round(endo_trials*(1-val_ratio)/12)),
                                   stimulus_pos_b*np.ones(round(endo_trials*(1-val_ratio)/12)),
                                   stimulus_pos_b*np.ones(round(endo_trials*(1-val_ratio)/12))))
    stim_x_unit = np.concatenate((stim_x_unit1, stim_x_unit1, stim_x_unit2, stim_x_unit2))
    stim_x = np.concatenate((stim_x_unit, stim_x_unit, stim_x_unit, stim_x_unit))

    stim_y_unit1 = np.concatenate((np.linspace(stimulus_pos_down, stimulus_pos_up, num=round(endo_trials*val_ratio/12)), 
                                   np.linspace(stimulus_pos_down, stimulus_pos_up, num=round(endo_trials*val_ratio/12)),
                                   np.linspace(stimulus_pos_down, stimulus_pos_up, num=round(endo_trials*val_ratio/12))))
    stim_y_unit2 = np.linspace(stimulus_pos_down, stimulus_pos_up, num=round(endo_trials*(1-val_ratio)/4))
    stim_y_unit = np.concatenate((stim_y_unit1, stim_y_unit1, stim_y_unit2, stim_y_unit2))
    stim_y = np.concatenate((stim_y_unit, stim_y_unit, stim_y_unit, stim_y_unit))

    all_trials = np.vstack((cue_type, all_cue, all_valid, all_ics, stim_side, stim_x, stim_y)).T
    np.random.shuffle(all_trials)
    return all_trials

def make_trial(num1, code1, num2, code2):
    trial = np.concatenate((code1*np.ones(num1, dtype=int), code2*np.ones(num2, dtype=int)))
    return trial

def fix(mywin, fixation, fix_time, left_rf, right_rf, trigger):
    fixation.draw()
    left_rf.draw()
    right_rf.draw()
    # trigger_flash.draw()
    trigger.write(b'H')
    mywin.flip()
    core.wait(fix_time)
    trigger.write(b'L')


def endo(mywin, left_rf, right_rf, arrow, stimulus,
          trigger, cue, stim_side, ics, stim_x, stim_y):
    
    if cue == -1:
        arrow.setVertices(arrow_left)
    else:
        arrow.setVertices(arrow_right)
    stimulus.setPos((stim_side*stim_x, stim_y))

    # draw cue and flip window
    left_rf.draw()
    right_rf.draw()
    arrow.draw()
    mywin.flip()
    core.wait(endo_cue_time)

    # wait for stimulus
    left_rf.draw()
    right_rf.draw()
    mywin.flip()
    core.wait(ics)

    # draw stimulus and flip window
    left_rf.draw()
    right_rf.draw()
    stimulus.draw()
    # trigger_flash.draw()
    trigger.write(b'H')
    mywin.flip()
    core.wait(stim_time)
    trigger.write(b'L')

    # wait for response
    left_rf.draw()
    right_rf.draw()
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


def exo(mywin, fixation, left_rf, right_rf, stimulus, 
        trigger, exo_rect, cue, stim_side, ics, stim_x, stim_y):
    
    exo_rect.setPos((cue*rf_pos, 0))
    stimulus.setPos((stim_side*stim_x, stim_y))

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
    # trigger_flash.draw()
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
