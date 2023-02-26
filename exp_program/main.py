from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
import numpy as np
from settings import *
from funcs import *

# dilogue box
expInfo = {'Name':'HAL', 'Test': 0}
expInfo['dateStr'] = data.getDateStr()  # add the current time
# present a dialogue to change params
dlg = gui.DlgFromDict(expInfo, title='Info', fixed=['dateStr'])
if dlg.OK:
    filename = expInfo['Name'] + "_" + expInfo['dateStr']
else:
    core.quit()  # the user hit cancel so exit
dataFile = open(filename+'.csv', 'w')  # a simple text file with 'comma-separated-values'
''' type: 1 = endogenous, 2 = exogenous
    cue: -1 = left, 1 = right
    valid: -1 = invalid, 1 = valid
    stimulus: -1 = left, 1 = right
    response: 0 = no response, 1 = has response
    reaction time: in ms '''
dataFile.write('type, cue, valid, stimulus, response, reaction time\n')

#create a window
mywin = visual.Window([screen_width, screen_height], 
                      fullscr=True, screen=0, monitor="testMonitor", 
                      color=[-1,-1,-1], units="pix")
print("Window created.")

#create objects
fixation = visual.ShapeStim(mywin, pos=[0,0], vertices=((0, -20), (0, 20), (0,0), (-20,0), (20, 0)),
                            lineWidth=5, closeShape=False, lineColor='white')
left_rf = visual.Rect(mywin, pos=(-1*rf_pos, 0), size=rf_size, lineColor='white', fillColor=None, lineWidth=5)
right_rf = visual.Rect(mywin, pos=(rf_pos, 0), size=rf_size, lineColor='white', fillColor=None, lineWidth=5)
stimulus = visual.Circle(mywin, pos=(stimulus_pos, 0), size=stimulus_size, lineColor=None, fillColor='red')
arrow = visual.ShapeStim(mywin, vertices=((0, 15), (-80, 15), (-80, 40), (-140, 0), (-80, -40), (-80, -15), (0, -15)),
                         fillColor='white', lineColor=None)
arrow.setVertices(arrow_right)
exo_rect = visual.Rect(mywin, pos=(-1*rf_pos, 0), size=rf_size, lineColor=None, fillColor='white')
print("Objects created.")

# generate trials
type = make_trials(endo_trials, 1, exo_trials, 2)
endo_cue = make_trials(int(endo_trials/2), 1, int(endo_trials/2), -1)
exo_cue = make_trials(int(exo_trials/2), 1, int(exo_trials/2), -1)
endo_valid = make_trials(int(endo_trials*val_ratio), 1, int(endo_trials*(1-val_ratio)), -1)
exo_valid = make_trials(int(exo_trials*val_ratio), 1, int(exo_trials*(1-val_ratio)), -1)
endo_stim = np.multiply(endo_cue, endo_valid)
exo_stim = np.multiply(exo_cue, exo_valid)
print("Trials generated.")

#draw the stimuli and update the window
fixation.draw()
left_rf.draw()
right_rf.draw()
stimulus.draw()
arrow.draw()
exo_rect.draw()
mywin.update()
#pause, so you get a chance to see it!
core.wait(3.0)