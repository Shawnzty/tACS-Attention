from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
import numpy as np
from settings import *
from funcs import *
import serial
import time

# dilogue box
''' Session: before, during, after
    test: 1 = test, 0 = not test '''
expInfo = {'Name':'HAL', 'Session':'before', 'Test': 0}
expInfo['dateStr'] = data.getDateStr()  # add the current time
# present a dialogue to change params
dlg = gui.DlgFromDict(expInfo, title='Info', fixed=['dateStr'])
if dlg.OK:
    filename = expInfo['Name'] + "_test_" + expInfo['dateStr'] if expInfo['Test']==1 else expInfo['Name'] + "_" + expInfo['Session'] + "_" + expInfo['dateStr']
else:
    core.quit()  # the user hit cancel so exit
dataFile = open('../data/'+filename+'.csv', 'w')  # a simple text file with 'comma-separated-values'
''' type: 1 = endogenous, 2 = exogenous
    cue: -1 = left, 1 = right
    valid: -1 = invalid, 1 = valid
    stimulus: -1 = left, 1 = right
    interval between cue and stimulus: in second
    response: 0 = no response, 1 = has response
    reaction time: in second '''
dataFile.write('type,cue,valid,ICS,stimulus side,stimulus x,stimulus y,response,reaction time\n')

#create a window
mywin = visual.Window([screen_width, screen_height], 
                      fullscr=True, screen=1, monitor="testMonitor", 
                      color=[-1,-1,-1], units="pix")
print("Window created.")

#create objects
fixation = visual.ShapeStim(mywin, pos=[0,0], vertices=((0, -20), (0, 20), (0,0), (-20,0), (20, 0)),
                            lineWidth=5, closeShape=False, lineColor='white')
left_rf = visual.Rect(mywin, pos=(-1*rf_pos, 0), size=rf_size, lineColor='white', fillColor=None, lineWidth=5)
right_rf = visual.Rect(mywin, pos=(rf_pos, 0), size=rf_size, lineColor='white', fillColor=None, lineWidth=5)
stimulus = visual.Circle(mywin, pos=(0, 0), size=stimulus_size, lineColor=None, fillColor='red')
arrow = visual.ShapeStim(mywin, vertices=((0, 15), (-80, 15), (-80, 40), (-140, 0), (-80, -40), (-80, -15), (0, -15)),
                         fillColor='white', lineColor=None)
arrow.setVertices(arrow_right)
exo_rect = visual.Rect(mywin, pos=(rf_pos, 0), size=rf_size, lineColor='white', fillColor=None, lineWidth=50)
trigger_flash = visual.Rect(mywin, pos=((screen_width-trigger_sizex)/2, trigger_ypos),
                       size=(trigger_sizex,trigger_sizey), lineColor=None, fillColor='white')
print("Objects created.")

# generate trials
if expInfo['Test'] == 0:
    all_trials = generate_all_trials(endo_trials, exo_trials, val_ratio)
else:
    all_trials = generate_all_trials(test_endo_trials, test_exo_trials, test_val_ratio)
print("Trials generated.")
print(all_trials)

refresh_rate = mywin.getActualFrameRate()
print("Refresh rate: %.2f", refresh_rate)

trigger = serial.Serial('COM3', 9600) # lab 11, office 3
print("Serial port for Arduino opened.")


start(mywin, expInfo)

for row in all_trials:
    fix(mywin, fixation, fix_time, left_rf, right_rf, trigger)

    if row[0] == 1: # endogenous
        response, reaction_time = endo(mywin, left_rf, right_rf, arrow,
                                         stimulus, trigger, cue = row[1], stim_side = row[4],
                                           ics = row[3], stim_x = row[5], stim_y = row[6])
    
        # save data
        dataFile.write('%i,%i,%i,%.2f,%i,%i,%i,%i,%.5f\n' %(row[0], row[1], row[2], row[3],
                                                     row[4], row[5], row[6], response, reaction_time))

    else: # exogenous
        response, reaction_time = exo(mywin, fixation, left_rf, right_rf, stimulus,
                                      trigger, exo_rect, cue = row[1], stim_side = row[4], 
                                        ics = row[3], stim_x = row[5], stim_y = row[6])

        # save data
        dataFile.write('%i,%i,%i,%.2f,%i,%i,%i,%i,%.5f\n' %(row[0], row[1], row[2], row[3],
                                                     row[4], row[5], row[6], response, reaction_time))
    print(dataFile)


finish(mywin, expInfo)
dataFile.close()
