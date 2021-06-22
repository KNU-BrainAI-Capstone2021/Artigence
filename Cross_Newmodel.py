"""
Name : Artigence
Data : 2021.06.22
source : https://github.com/vlawhern/arl-eegmodels
"""
import numpy as np
import tensorflow as tf
import mne
import random
# EEGNet-specific imports
from EEGNet import Multi_DS_EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras import backend as K

# PyRiemann imports
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.model_selection import train_test_split, KFold

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# tf.debugging.set_log_device_placement(True)

#saving result path
save_dir = 'C:/Users/PC/Desktop/multi_ds_cross.txt'

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

#set Kfold's k Num using n_splits
kf = KFold(n_splits=4, shuffle= True, random_state= True)

##################### Process, filter and epoch the data ######################
dpath = "C:/Users/PC/OneDrive - knu.ac.kr/Data/"
#median = 56.75 good subject acc > medain else bad subject
good_subject = [1,2,3,4,5,6,8,11,12,13,14,20,21,22,23,25,27,29,32,33,39,43,46,47,50,51]
bad_subject = [7,9,10,15,16,17,18,19,24,26,28,30,31,34,35,36,37,38,40,41,42,44,45,48,49,52]

event_id = dict(left=1, right=2)
tmin = 0
tmax = 2.9980
custom_mapping = {'left': 1, 'right': 2}

for test_num in range(1,53):
    X = []
    Y = []
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    random.shuffle(good_subject)
    random.shuffle(bad_subject)
    print(good_subject)
    print(bad_subject)
    good_count = 0
    bad_count = 0
    eeglab_raw = mne.io.read_raw_eeglab(dpath + "s" + str(test_num) + "_pir.set")
    (events_from_annot, event_dict) = mne.events_from_annotations(eeglab_raw, event_id=custom_mapping)
    epochs = mne.Epochs(eeglab_raw, events_from_annot, event_id, tmin, tmax, baseline=None)
    labels = epochs.events[:, -1]
    data = epochs.get_data() * 1000  # format is in (trials, channels, samples)
    X_test = np.array(data)
    Y_test = np.array(labels)
    for good_num in good_subject:
        if(good_num == test_num):
            continue
        eeglab_raw  = mne.io.read_raw_eeglab(dpath + "s"+str(good_num)+"_pir.set")
        (events_from_annot,event_dict) = mne.events_from_annotations(eeglab_raw, event_id=custom_mapping)
        epochs = mne.Epochs(eeglab_raw, events_from_annot, event_id, tmin, tmax,baseline = None)
        labels = epochs.events[:,-1]
        data = epochs.get_data( ) *1000 # format is in (trials, channels, samples)
        X.extend(data)
        Y.extend(labels)
        good_count=good_count+1
        if(good_count==4):
            break;
    for bad_num in bad_subject:
        if(bad_num == test_num):
            continue
        eeglab_raw  = mne.io.read_raw_eeglab(dpath + "s"+str(bad_num)+"_pir.set")
        (events_from_annot,event_dict) = mne.events_from_annotations(eeglab_raw, event_id=custom_mapping)
        epochs = mne.Epochs(eeglab_raw, events_from_annot, event_id, tmin, tmax,baseline = None)
        labels = epochs.events[:,-1]
        data = epochs.get_data( ) *1000 # format is in (trials, channels, samples)
        X.extend(data)
        Y.extend(labels)
        bad_count=bad_count+1
        if(bad_count==4):
            break
    X_train = np.array(X)
    Y_train =np.array(Y)

    kernels, chans, samples = 1, 64, 1536

    ############################# EEGNet portion ##################################
    # convert labels to one-hot encodings.
    Y_train = np_utils.to_categorical(Y_train - 1)
    Y_test = np_utils.to_categorical(Y_test - 1)

    # convert data to NHWC (trials, channels, samples, kernels) format. Data
    # contains 60 channels and 151 time-points. Set the number of kernels to 1.
    X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)
    print(X_train.shape,"X train shape")
    print(X_test.shape, "X test shape")
    # take 50/25/25 percent of the data to train/validate/test
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.9, shuffle=True,
                                                      random_state=1004)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    # Chans, Samples  : number of channels and time points in the EEG data
    # kernelLength : half of data sampling rate
    model = Multi_DS_EEGNet(nb_classes=2, Chans=chans, Samples=samples,
                   dropoutRate=0.5, kernLength=256, F1=4, D=2, F2=8,
                   dropoutType='Dropout')

    # compile the model and set the optimizers using binary-cross entropy
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    # count number of parameters in the model
    numParams = model.count_params()

    ##############################################################################
    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    class_weights = {0: 1, 1: 1}

    ################################################################################
    # fit the model
    ################################################################################'
    hist = model.fit(X_train, Y_train, batch_size=16, epochs=150,
                     verbose=2, validation_data=(X_val, Y_val), shuffle=True,
                     class_weight=class_weights, callbacks=[early_stopping])

    ###############################################################################
    # make prediction on test set.
    ###############################################################################

    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
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

    f = open(save_dir, "a")
    f.write("TEST" + str(test_num) + "\n")
    f.write("test_acc=" + str(hist.history['accuracy'][-1]) + "\n")
    f.write("val_acc=" + str(hist.history['val_accuracy'][-1]) + "\n")
    f.write("test_acc" + str(acc) + "\n")
    f.close()


    ##########################################
    # show Confusion_matrix
    ##########################################
    names = ['hand left', 'hand right']
    plt.figure()
    plot_confusion_matrix(preds, Y_test.argmax(axis=-1), names, title='EEGNet-8,2')
    # plt.show()
