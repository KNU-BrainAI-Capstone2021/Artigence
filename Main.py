import numpy as np
import tensorflow as tf
import mne

# EEGNet-specific imports
from EEGNet import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras import backend as K

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

#show device placement
#tf.debugging.set_log_device_placement(True)

#saving result path
save_dir = 'C:/Users/PC/Desktop/artigence_result.txt'

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

#set Kfold's k Num using n_splits
kf = KFold(n_splits=4, shuffle= True, random_state= True)

##################### Process Main  ######################
for i in range(1,31):
    count = 1
    #data path where preprocessed data
    dpath = "C:/Users/PC/OneDrive - knu.ac.kr/Data/s"+str(i)+"_pir.set"
    #get data and find event
    eeglab_raw  = mne.io.read_raw_eeglab(dpath)
    custom_mapping = {'left': 1, 'right': 2}
    (events_from_annot,event_dict) = mne.events_from_annotations(eeglab_raw, event_id=custom_mapping)
    event_id = dict(left=1,right=2)
    tmin = 0
    tmax = 2.9980
    epochs = mne.Epochs(eeglab_raw, events_from_annot, event_id, tmin, tmax,baseline = None)
    labels = epochs.events[:,-1]
    #data
    X = epochs.get_data( )*1000 # format is in (trials, channels, samples)
    Y = labels

    kernels, chans, samples = 1, 64, 1536
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        Y_train = Y[train_index]
        print("x_train_shape", X_train.shape)
        X_test = X[test_index]
        Y_test = Y[test_index]
        # take 50/25/25 percent of the data to train/validate/test

        ############################# EEGNet portion ##################################
        # convert labels to one-hot encodings.
        Y_train      = np_utils.to_categorical(Y_train -1)
        Y_test       = np_utils.to_categorical(Y_test -1)

        # convert data to NHWC (trials, channels, samples, kernels) format. Data
        # contains 60 channels and 151 time-points. Set the number of kernels to 1.
        X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.67, shuffle=True,
                                                          random_state=1004)
        # Chans, Samples  : number of channels and time points in the EEG data
        # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
        # model configurations may do better, but this is a good starting point)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
        model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples,
                       dropoutRate = 0.8, kernLength = 256, F1 = 4, D = 2, F2 = 8,
                       dropoutType = 'Dropout')

        # compile the model and set the optimizers
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics = ['accuracy'])
        model.summary()
        # count number of parameters in the model
        numParams    = model.count_params()

        ###############################################################################
        # if the classification task was imbalanced (significantly more trials in one
        # class versus the others) you can assign a weight to each class during
        # optimization to balance it out. This data is approximately balanced so we
        # don't need to do this, but is shown here for illustration/completeness.
        ###############################################################################

        # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
        # the weights all to be 1
        class_weights = {0 :1, 1 :1}

        ################################################################################
        # fit the model. Due to very small sample sizes this can get
        # pretty noisy run-to-run, but most runs should be comparable to xDAWN +
        # Riemannian geometry classification (below)
        ################################################################################
        hist = model.fit(X_train, Y_train, batch_size = 16, epochs = 150,
                         verbose = 2, validation_data=(X_val,Y_val), shuffle=True,
                         class_weight = class_weights, callbacks = [early_stopping])

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

        f = open(save_dir, "a")
        f.write("TEST"+ str(i) + "\n")
        f.write("k =" + str(count) + "\n")
        f.write("test_acc=" + str(hist.history['accuracy'][-1]) + "\n")
        f.write("val_acc=" + str(hist.history['val_accuracy'][-1]) + "\n")
        f.write("test_acc" + str(acc) + "\n")
        f.close()
        count = count+1

        ##########################################
        #sh
        ##########################################
        names        = ['hand left', 'hand right']
        plt.figure()
        plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')
        #plt.show()
