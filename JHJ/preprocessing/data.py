import os
import sys
import mne
from scipy import io

import sklearn
import numpy as np
import matplotlib.pyplot as plt
file = "C:/Users/oo/Desktop/Artigence/s01.mat"
file_path = os.path.join(file)
mat_file = io.loadmat(file)
eeg_file = mat_file.get('eeg')
#print(eeg_file)
"""
• rest: resting state with eyes-open condition
• noise:
    - eye blinking, 5 seconds × 2
    - eyeball movement up/down, 5 seconds × 2
    - eyeball movement left/right, 5 seconds × 2
    - jaw clenching, 5 seconds × 2
    - head movement left/right, 5 seconds × 2
• imagery_left: 100 or 120 trials of left hand MI
• imagery_right: 100 or 120 trials of right hand MI
• n_imagery_trials: 100 or 120 trials for each MI class
• imagery_event: value “1” represents onset for each MI trial
• movement_left: 20 trials of real left hand movement
• movement_right: 20 trials of real right hand movement
• n_movement_trials: 20 trials for each real hand movement class
• movement_event: value “1” represents onset for each movement trial
• frame: temporal range of a trial in milliseconds
• srate: sampling rate
• senloc: 3D sensor locations
• psenloc: sensor location projected to unit sphere
• subject: subject's two-digit ID - “s#”
• comment: comments for the subject
• bad_trial_indices
    - bad trials determined by voltage magnitude
    - bad trials correlated with EMG activity
"""
noise = eeg_file[0][0][0]
rest = eeg_file[0][0][1]
srate = eeg_file[0][0][2]
movement_left = eeg_file[0][0][3]
movement_right = eeg_file[0][0][4]
movement_event = eeg_file[0][0][5]
n_movement_trials = eeg_file[0][0][6]
imagery_left = eeg_file[0][0][7]
imagery_right = eeg_file[0][0][8]
n_imagery_trials = eeg_file[0][0][9]
frame = eeg_file[0][0][10]
imagery_event = eeg_file[0][0][11]
comment = eeg_file[0][0][12]
subject = eeg_file[0][0][13]
bad_trial_indices = eeg_file[0][0][14]
psenloc = eeg_file[0][0][15]
senloc = eeg_file[0][0][16]


plt.plot(movement_event)
"""
#np.set_printoptions(threshold=sys.maxsize)
print("rest:",rest.shape)
print("noise:",noise.shape)
print("im_left:", imagery_left)
print("im_right:",imagery_right.shape , "im_left")
print("n_im_trials:",n_imagery_trials.shape)
print("im_event:",imagery_event.shape)
print("mv_left:", movement_left)
print("mv_right:",movement_right.shape)
print("n_mv_trials:",n_movement_trials.shape)
print("mv_event:",movement_event)
print("frame:",frame)
print("srate:", srate.shape)
print("senloc:",senloc)
print("psenloc",psenloc)
print("subject:",subject)
print("comment:",comment.shape)
print("bad_trail_in:", bad_trial_indices)
"""
