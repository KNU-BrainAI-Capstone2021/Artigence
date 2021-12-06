# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 21:29:59 2021

@author: PC00
"""
import numpy as np
import tensorflow as tf
import mne
import random
import glob
import random
from tensorflow.keras.callbacks import ModelCheckpoint
# EEGNet-specific imports
import os 
os.chdir("D:/Artigence")
#sys.path
from EEGNet import Multi_DS_EEGNet, EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras import backend as K

# PyRiemann imports
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.model_selection import train_test_split, KFold

# tools for plotting confusion matrices
from matplotlib import pyplot as plt
X_train = []
Y_train =[]
X_val = []
Y_val = []
X_test = []
Y_test = []
data = []
X=[]
remove_chan = [2,14,18,29]
#%%
#x = np.loadtxt("C:/Users/PC/Desktop/BCI/data.txt")
#eeglab_raw  = mne.io.read_raw_edf("C:/Users/PC/Desktop/data/prepro/left/6.edf")
eeglab_raw  = mne.io.read_raw_edf("C:/Users/PC/Desktop/data/prepro/test.edf")
temp = eeglab_raw.get_data( ) # format is in (trials, channels, samples)
temp = mne.filter.resample(temp,down=1.28,npad='auto')
for i in range(32):
    if i not in remove_chan:
        data.append(temp[i,:])
    else:
        continue
data = np.array(data)
data = data[:,3000:] #samplingrate = 256
data = [data[:,i:i + 600] for i in range(0, len(data[-1]), 600)]
data = data[:30] #37

for i in range(len(data)):
    X.append(data[i])

#samples = 3sec * 512Hz sampling rate
kernels, chans, samples = 1, 28, 600

X = np.array(X)
X = X.reshape(len(data), chans, samples, kernels)

new_model = Multi_DS_EEGNet(nb_classes=3, Chans=chans, Samples=samples,
                       dropoutRate=0.5, kernLength=100, F1=4, D=2, F2=8,
                       dropoutType='Dropout')
# find best weights
#latest = tf.train.latest_checkpoint("C:/Users/PC/Desktop/BCI/checkpoint/1/epoch_125.ckpt")

new_model.load_weights("C:/Users/PC/Desktop/data/checkpoints/epoch_087.ckpt")
probs = new_model.predict(X)
preds = probs.argmax(axis=-1)

print(probs)
print(preds)