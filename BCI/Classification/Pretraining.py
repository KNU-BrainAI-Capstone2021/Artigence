"""
Name : Artigence
Data : 2021.06.22
source : https://github.com/vlawhern/arl-eegmodels
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

# tf.debugging.set_log_device_placement(True)

#saving result path
save_dir = 'C:/Users/PC/Desktop/resring_x_select.txt'

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

#set Kfold's k Num using n_splits
kf = KFold(n_splits=4, shuffle= True, random_state= True)

##################### Process, filter and epoch the data ######################
dpath = "D:/data/3-ClassMI/"
#median = 56.75 good subject acc > medain else bad subject
subject = [1,2,3,4,5,6,7,9,10,11,12]

#%%
event_id = dict(left=1, right=2, foot=3)
tmin = 0
tmax = 2.9970
custom_mapping = {'left': 1, 'right': 2, 'Foot':3}

"""
                    #%%
                    count = 0
                    temp = events_from_annot[0]
                    for k in range(1,len(events_from_annot)):
                        if events_from_annot[k][2] == 4:
                            if count < 30:
                                temp = np.vstack((temp,events_from_annot[k]))
                                count += 1
                            else:
                                continue
                        else: 
                            temp = np.vstack((temp,events_from_annot[k]))
                    events_from_annot = temp
"""                    


for k in range(1,2):
    #for i in range(1,13):

    for i in range(3,4):
        if i == 8:
            continue
        X_train = []
        Y_train =[]
        X_validate = []
        Y_validate = []
        X_test = []
        Y_test = []
        
        test=[]
        val=[]
        train=[]
        
        rand_list = []
        val_list= []
        train_list = []
        #train_list = [1,4,6,7,9,11]
        
        
        for x in range(1,13):
            if x == i or x ==8: continue
            else:rand_list.append(x)
            
        val_list = random.sample(rand_list,2)
        
        
        for j in subject:
            if j == i :   
                print("test subject",j)
                test.append(j)
                for filename in glob.glob("D:/data/3-ClassMI/S"+str(i)+"/*.set"):
                    #data path where preprocessed data                    
                    dpath = filename
                    #get data and find event
                    eeglab_raw  = mne.io.read_raw_eeglab(dpath)   
                    anno = mne.read_annotations(dpath)        
                    (events_from_annot,event_dict) = mne.events_from_annotations(eeglab_raw)
                    
                                        
                    #eeglab_raw._annotations
                    
                    epochs = mne.Epochs(eeglab_raw, events_from_annot, event_id, tmin, tmax,baseline = None)
                    #print(epochs)
                    #print(epochs.events)
                    labels = epochs.events[:,-1]
                    #data *1000 uV to V
                    data = epochs.get_data( )*1000000 # format is in (trials, channels, samples)
                    #print(data.shape)
                    
                    X_test.extend(data)
                    Y_test.extend(labels)
            
            elif(j in val_list) :
            #elif (j == 3):
                print("validate subject",j)
                val.append(j)
                
                for filename in glob.glob("D:/data/3-ClassMI/S"+str(j)+"/*.set"):
                    #data path where preprocessed data
                    dpath = filename
                    #get data and find event
                    eeglab_raw  = mne.io.read_raw_eeglab(dpath)   
                    anno = mne.read_annotations(dpath)        
                    (events_from_annot,event_dict) = mne.events_from_annotations(eeglab_raw)
                    
                    
                    epochs = mne.Epochs(eeglab_raw, events_from_annot, event_id, tmin, tmax,baseline = None)
                    labels = epochs.events[:,-1]
                    #data *1000 uV to V
                    data = epochs.get_data( )*1000000 # format is in (trials, channels, samples)
                    X_validate.extend(data)
                    Y_validate.extend(labels)
            
            #elif(j in train_list) :
            else:
                print("train subject",j)
                train.append(j)
                for filename in glob.glob("D:/data/3-ClassMI/S"+str(j)+"/*.set"):
                    #data path where preprocessed data
                    dpath = filename
                    #get data and find event
                    eeglab_raw  = mne.io.read_raw_eeglab(dpath)   
                    anno = mne.read_annotations(dpath)        
                    (events_from_annot,event_dict) = mne.events_from_annotations(eeglab_raw)
                    
                    
                    epochs = mne.Epochs(eeglab_raw, events_from_annot, event_id, tmin, tmax,baseline = None)
                    labels = epochs.events[:,-1]
                    #data *1000 uV to V
                    data = epochs.get_data( )*1000000 # format is in (trials, channels, samples)
                    X_train.extend(data)
                    Y_train.extend(labels)              

            #else : continue
#%%

        
        print(np.shape(X_test))
        X_test=np.array(X_test)
        print(np.shape(X_test))
        Y_test=np.array(Y_test)
        print(np.shape(X_validate))
        X_validate=np.array(X_validate)
        print(np.shape(X_validate))
        Y_validate=np.array(Y_validate)
        X_train=np.array(X_train)
        Y_train=np.array(Y_train)
    
    
    
        kernels, chans, samples = 1, 28, 600
    
        ############################# EEGNet portion ##################################
        # convert labels to one-hot encodings.
        Y_train = np_utils.to_categorical(Y_train - 1)
        Y_validate = np_utils.to_categorical(Y_validate - 1)
        #Y_test = np_utils.to_categorical(Y_test - 1)
    
        # convert data to NHWC (trials, channels, samples, kernels) format. Data
        # contains 60 channels and 151 time-points. Set the number of kernels to 1.
        X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
        X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)
        print(X_train.shape,"X train shape")
        print(X_validate.shape,"X validate shape")
        print(X_test.shape, "X test shape")
        # take 50/25/25 percent of the data to train/validate/test
        #%%
        #X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.9, shuffle=True,
                                                       #   random_state=1004)
    
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        # Chans, Samples  : number of channels and time points in the EEG data
        # kernelLength : half of data sampling rate
        model = Multi_DS_EEGNet(nb_classes=3, Chans=chans, Samples=samples,
                       dropoutRate=0.5, kernLength=100, F1=4, D=2, F2=8,
                       dropoutType='Dropout')
    
        # compile the model and set the optimizers using binary-cross entropy
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        # count number of parameters in the model
        numParams = model.count_params()
        checkpoint_path = "C:/Users/PC/Desktop/BCI/#openvibe/checkpoints/3/epoch_{epoch:03d}.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True, period=1,
                                                 verbose=1, save_best_only = True)
        
        ##############################################################################
        # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
        # the weights all to be 1
        class_weights = {0: 1, 1: 1, 2: 1}
    
        ################################################################################
        # fit the model
        ################################################################################'
        hist = model.fit(X_train, Y_train, batch_size=128, epochs=150,
                         verbose=2, validation_data=(X_validate, Y_validate), shuffle=True,
                         class_weight=class_weights, callbacks=[early_stopping, cp_callback])
    
        ###############################################################################
        # make prediction on test set.
        ###############################################################################
        model.save('C:/Users/PC/Desktop/data/prepro/pretrained_model')
        probs = model.predict(X_test)
        preds = probs.argmax(axis=-1)
        #acc = np.mean(preds == Y_test.argmax(axis=-1))
        acc = np.mean(preds == Y_test-1)
        print("Classification accuracy: %f " % (acc))
    
        print(preds)
        print(Y_test.argmax(axis=-1))
        preds_, counts_p = np.unique(preds, return_counts=True)
        dict(zip(preds_, counts_p))
        Y_test_, counts_Y = np.unique(Y_test-1, return_counts=True)
        dict(zip(Y_test_, counts_Y))
        
    
    
    
    
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
        f.write("TEST" + str(i) + "\n")
        f.write("test_acc=" + str(hist.history['accuracy'][-1]) + "\n")
        f.write("val_acc=" + str(hist.history['val_accuracy'][-1]) + "\n")
        f.write("test_acc " + str(acc) + "\n")
        f.close()
    
        ##########################################
        # show Confusion_matrix
        ##########################################
        names = ['left', 'right','foot']
        plt.figure()
        plot_confusion_matrix(Y_test-1, preds,  names, title=('test {0}'.format(i) )) # ( y_true , y_pred)
        plt.show()
