import numpy as np
from psychopy import core, visual, gui, data, event
from settings import *

def generate_all_trials(endo_trials, exo_trials, val_ratio):
    cue_type = make_trial(endo_trials, 1, exo_trials, 2)
    endo_cue = make_trial(int(endo_trials/2), 1, int(endo_trials/2), -1)
    exo_cue = make_trial(int(exo_trials/2), 1, int(exo_trials/2), -1)
    endo_valid = make_trial(round(endo_trials*val_ratio), 1, round(endo_trials*(1-val_ratio)), -1)
    exo_valid = make_trial(round(exo_trials*val_ratio), 1, round(exo_trials*(1-val_ratio)), -1)
    endo_stim = np.multiply(endo_cue, endo_valid).tolist()
    exo_stim = np.multiply(exo_cue, exo_valid).tolist()
    return cue_type, endo_cue, exo_cue, endo_valid, exo_valid, endo_stim, exo_stim

def make_trial(num1, code1, num2, code2):
    trial = np.concatenate((code1*np.ones(num1, dtype=int), code2*np.ones(num2, dtype=int)))
    np.random.shuffle(trial)
    return trial.tolist()

def fix(mywin, fixation, fix_time, left_rf, right_rf, trigger):
    fixation.draw()
    left_rf.draw()
    right_rf.draw()
    trigger.draw()
    mywin.flip()
    core.wait(fix_time)


def endo(mywin, fixation, left_rf, right_rf, arrow, stimulus, trigger, cue, stim):
    
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
    core.wait(endo_ics)

    # draw stimulus and flip window
    left_rf.draw()
    right_rf.draw()
    fixation.draw()
    stimulus.draw()
    trigger.draw()
    mywin.flip()
    core.wait(endo_stim_time)

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


def exo(mywin, fixation, left_rf, right_rf, stimulus, trigger, exo_rect, cue, stim):
    
    exo_rect.setPos((cue*rf_pos, 0))
    stimulus.setPos((stim*stimulus_pos, 0))

    # draw cue and flip window
    for i in range(exo_cue_flash):
        left_rf.draw()
        right_rf.draw()
        fixation.draw()
        exo_rect.draw()
        mywin.flip()
        core.wait(exo_cue_flash_ontime)

        left_rf.draw()
        right_rf.draw()
        fixation.draw()
        mywin.flip()
        core.wait(exo_cue_flash_offtime)

    # wait for stimulus
    left_rf.draw()
    right_rf.draw()
    fixation.draw()
    mywin.flip()
    core.wait(exo_ics)

    # draw stimulus and flip window
    left_rf.draw()
    right_rf.draw()
    fixation.draw()
    stimulus.draw()
    trigger.draw()
    mywin.flip()
    core.wait(exo_stim_time)


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
