"""
mne에서 tutorial로 제공하는 MI Dataset으로 EEGNet돌려보기
Requirments
mne = 0.17.1
"""
import numpy as np
import tensorflow as tf
# mne imports
import mne
from mne import io
from mne.datasets import sample

# EEGNet-specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# tools for plotting confusion matrices
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

print(__doc__)

# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -1., 4.
event_id = dict(left=2, right=3)
subject = [1,2,3,4,5]
runs = [4, 8, 12]  # motor imagery: hands vs feet

raw_fnames = [eegbci.load_data(s, runs) for s in subject]
print(raw_fnames)
X=[]
y=[]
for i in range(len(subject)):
    raw_files = [read_raw_edf(f, preload=True) for f in raw_fnames[i]]
    raw = concatenate_raws(raw_files)
     # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # Apply band-pass filter
    raw.filter(7., 30., method='iir')

    events = find_events(raw, shortest_event=0, stim_channel='STI 014')

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs_train = epochs.crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2


    # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
    #(45,64,161)
    data = epochs.get_data()*1000 # format is in (trials, channels, samples)
    X.extend(data)
    y.extend(labels+1)
X = np.array(X)
y = np.array(y)
print(len(X))
kernels, chans, samples = 1, 64, 161

# take 50/25/25 percent of the data to train/validate/test
X_train,X_test,Y_train,Y_test = train_test_split(X,y, train_size=0.75, shuffle=True,random_state=1004)

############################# EEGNet portion ##################################

# convert labels to one-hot encodings.
Y_train      = np_utils.to_categorical(Y_train -1)
#Y_validate   = np_utils.to_categorical(Y_validate -1)
Y_test       = np_utils.to_categorical(Y_test -1)
# convert data to NHWC (trials, channels, samples, kernels) format. Data
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
#X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# Chans, Samples  : number of channels and time points in the EEG data
# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
# model configurations may do better, but this is a good starting point)
# kernels, chans, samples = 1, 60, 151
model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples,
               dropoutRate = 0.5, kernLength = 32, F1 = 16, D = 2, F2 = 32,
               dropoutType = 'Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])
model.summary()
# count number of parameters in the model
numParams    = model.count_params()

# set a valid path for your system to record model checkpoints

###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during
# optimization to balance it out. This data is approximately balanced so we
# don't need to do this, but is shown here for illustration/completeness.
###############################################################################

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
class_weights = {0:1, 1:1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN +
# Riemannian geometry classification (below)
################################################################################
hist = model.fit(X_train, Y_train, batch_size = 16, epochs = 300,
                        verbose = 2,validation_split=0.33,
                     class_weight = class_weights)

# load optimal weights


###############################################################################
# can alternatively used the weights provided in the repo. If so it should get
# you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
# system.
###############################################################################

# WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5
# model.load_weights(WEIGHTS_PATH)

###############################################################################
# make prediction on test set.
###############################################################################

probs = model.predict(X_test)
preds = probs.argmax(axis = -1)
acc = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
# ############################# PyRiemann Portion ##############################
#
# # code is taken from PyRiemann's ERP sample script, which is decoding in
# # the tangent space with a logistic regression
#
# n_components = 2  # pick some components
#
# # set up sklearn pipeline
# clf = make_pipeline(XdawnCovariances(n_components),
#                     TangentSpace(metric='riemann'),
#                     LogisticRegression())
#
# preds_rg     = np.zeros(len(Y_test))
#
# # reshape back to (trials, channels, samples)
# X_train      = X_train.reshape(X_train.shape[0], chans, samples)
# X_test       = X_test.reshape(X_test.shape[0], chans, samples)
#
# # train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# # labels need to be back in single-column format
# history = clf.fit(X_train, Y_train.argmax(axis = -1))
# preds_rg     = clf.predict(X_test)
#
# # Printing the results
# acc2 = np.mean(preds_rg == Y_test.argmax(axis = -1))
# print("Classification accuracy: %f " % (acc2))
#
# # plot the confusion matrices for both classifiers
# names        = ['audio left', 'audio right', 'vis left', 'vis right']
# plt.figure(0)
# plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')
#
# plt.figure(1)
# plot_confusion_matrix(preds_rg, Y_test.argmax(axis = -1), names, title = 'xDAWN + RG')
# fig, loss_ax = plt.subplots()
#
# acc_ax = loss_ax.twinx()
#
# loss_ax.plot(hist.history['loss'], 'y', label='train loss')
# loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
#
# acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
# acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
#
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# acc_ax.set_ylabel('accuray')
#
# loss_ax.legend(loc='upper left')
# acc_ax.legend(loc='lower left')
#
# plt.show()
print("end")

