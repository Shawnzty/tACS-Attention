import numpy as np
from psychopy import core, visual, gui, data, event
from settings import *

def make_trials(num1, code1, num2, code2):
    trial = np.concatenate((code1*np.ones(num1, dtype=int), code2*np.ones(num2, dtype=int)))
    np.random.shuffle(trial)
    return trial.tolist()

def fix(mywin, fixation, fix_time, left_rf, right_rf):
    fixation.draw()
    left_rf.draw()
    right_rf.draw()
    mywin.flip()
    core.wait(fix_time)


def endo(mywin, left_rf, right_rf, arrow, stimulus, cue, stim):
    
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
    mywin.flip()
    core.wait(endo_ics)

    # draw stimulus and flip window
    left_rf.draw()
    right_rf.draw()
    stimulus.draw()
    mywin.flip()
    core.wait(endo_stim_time)

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


def exo(mywin, left_rf, right_rf, stimulus, exo_rect, cue, stim):
    
    exo_rect.setPos((cue*rf_pos, 0))
    stimulus.setPos((stim*stimulus_pos, 0))

    # draw cue and flip window
    for i in range(exo_cue_flash):
        left_rf.draw()
        right_rf.draw()
        exo_rect.draw()
        mywin.flip()
        core.wait(exo_cue_flash_ontime)

        left_rf.draw()
        right_rf.draw()
        mywin.flip()
        core.wait(exo_cue_flash_offtime)

    # wait for stimulus
    left_rf.draw()
    right_rf.draw()
    mywin.flip()
    core.wait(exo_ics)

    # draw stimulus and flip window
    left_rf.draw()
    right_rf.draw()
    stimulus.draw()
    mywin.flip()
    core.wait(exo_stim_time)


    # wait for response
    left_rf.draw()
    right_rf.draw()
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
    message2 = visual.TextStim(mywin, pos=[0,-100],text='Please hit the space when ready.')
    message1.setSize(30)
    message2.setSize(30)
    message1.draw()
    message2.draw()
    mywin.flip()
    event.waitKeys(keyList=['space'])


def finish(mywin, expInfo):
    # display instructions and wait
    message1 = visual.TextStim(mywin, pos=[0,0],
        text="Thank you for participating, " + expInfo['Name'] + "!")
    message1.setSize(30)
    message1.draw()
    mywin.flip()
    core.wait(2)
