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
save_dir = 'C:/Users/PC/Desktop/J.txt'

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

#set Kfold's k Num using n_splits
kf = KFold(n_splits=4, shuffle= True, random_state= True)

##################### Process, filter and epoch the data ######################
#%% load file left
Dpath = "C:/Users/PC/Desktop/data/prepro/"
task = ['left','right','foot']

"""
file=[]
for i in range(0,3):
    filepath = dpath+str(task[i])+"/"
    for j in range(1,4):
        #file.append(glob.glob(filepath+str(j)+".set"))
        file.append(filepath+str(j)+".set")
print(file)

"""

left=[]
right=[]
foot=[]


X_train = []
Y_train =[]
X_val = []
Y_val = []
X_test = []
Y_test = []

remove_chan = [2,14,18,29]
#%% left
for i in range(1,7):    
    filepath = Dpath+"left/"
    left.append(filepath+str(i)+".edf")
print(left)

for j in range(len(left)):
    data = []
    dpath=left[j]             #left\2.fdt 38
    #get data 
    eeglab_raw  = mne.io.read_raw_edf(dpath)   
    
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
    print(len(data))
    data = data[:36] #37
    labels = [1 for i in range(len(data))]
    
    for i in range(len(data)):
        if i % 10 == 0:
            X_train.append(data[i])
            Y_train.append(labels[i])
            
        else:
            X_test.append(data[i])
            Y_test.append(labels[i])

#%% right
for i in range(1,7):    
    filepath = Dpath+"right/"
    right.append(filepath+str(i)+".edf")
print(right)

for j in range(len(right)):
    data = []
    dpath=right[j]             #left\2.fdt 38
    #get data 
    eeglab_raw  = mne.io.read_raw_edf(dpath)   
    
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
    data = data[:36] #37
    labels = [2 for i in range(len(data))]
    #print(data)
    
    
    for i in range(len(data)):
        if i % 10 == 0:
            X_train.append(data[i])
            Y_train.append(labels[i])
            
        else:
            X_test.append(data[i])
            Y_test.append(labels[i])

#%% foot
for i in range(1,8):    
    filepath = Dpath+"foot/"
    foot.append(filepath+str(i)+".edf")
print(foot)

for j in range(len(foot)):
    data = []
    dpath=foot[j]             #left\2.fdt 38
    #get data 
    eeglab_raw  = mne.io.read_raw_edf(dpath)   
    
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
    data = data[:36] #37
    labels = [3 for i in range(len(data))]
    
    for i in range(len(data)):
        if i % 10 == 0:
            X_train.append(data[i])
            Y_train.append(labels[i])
            
        else:
            X_test.append(data[i])
            Y_test.append(labels[i])            
#%%
'''
eeglab_raw  = mne.io.read_raw_eeglab("C:/Users/PC/Desktop/data/prepro/left/1.set")   
   
data = eeglab_raw.get_data( )*1000000 # format is in (trials, channels, samples)
data = data[:,0:] #samplingrate = 256
data = [data[:,i:i + 768] for i in range(0, len(data[-1]), 768)]
data = data[:40]
labels = [3 for i in range(len(data))]

for i in range(len(data)):
    X_test.append(data[i])
    Y_test.append(labels[i])            
    
'''
    
#%%


""" 
a=[]
task = ['left','right','foot'] 
for k in range(3):
    for i in range(1,4):    
        filepath = Dpath+str(task[k])+"/"
        a.append(filepath+str(i)+".set")
task[k] = a
print(a)
    
    for j in range(len(left)):
        dpath=left[j]             #left\2.fdt 38
        #get data 
        eeglab_raw  = mne.io.read_raw_eeglab(dpath)   
        
        data = eeglab_raw.get_data( )*1000000 # format is in (trials, channels, samples)
        data = data[:,2560:] #samplingrate = 256
        data = [data[:,i:i + 768] for i in range(0, len(data[-1]), 768)]
        data = data[:40]
        labels = [1 for i in range(40)]
        
        for i in range(len(data)):
            if i % 3 != 0:
                X_train.append(data[i])
                Y_train.append(labels[i])
                
            else:
                X_test.append(data[i])
                Y_test.append(labels[i])
"""


#%%
"""
event_id = dict(left=1, right=2, foot=3)
tmin = 0
tmax = 2.9970
custom_mapping = {'left': 1, 'right': 2, 'Foot':3}
"""
           
#%%
print("end")
print(np.shape(X_train))
print(np.shape(X_test))

# shuffle the test set
test = [[x,y] for x, y in zip(X_test, Y_test)]
random.shuffle(test)
X_test = [n[0] for n in test]
Y_test = [n[1] for n in test]

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

Y_train      = np_utils.to_categorical(Y_train -1)
#Y_test       = np_utils.to_categorical(Y_test -1)

#samples = 3sec * 512Hz sampling rate
kernels, chans, samples = 1, 28, 600

## convert data to NHWC (trials, channels, samples, kernels) format. Data
X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

# take 50/25/25 percent of the data to train/validate/test
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.7, shuffle=True,
                                                  random_state=1004)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)



# model training
model = tf.keras.models.load_model('C:/Users/PC/Desktop/data/prepro/pretrained_model')
# find best weights
#latest = tf.train.latest_checkpoint("C:/Users/PC/Desktop/BCI/checkpoint/1/epoch_125.ckpt")

#model.load_weights("C:/Users/PC/Desktop/data/shuffle/epoch_074.ckpt")
#model = Multi_DS_EEGNet(nb_classes=3, Chans=28, Samples=600,
#                        dropoutRate=0.5, kernLength=100, F1=8, D=2, F2=16)
# compile the model and set the optimizers using binary-cross entropy
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.summary()

# count number of parameters in the model
"""
numParams = model.count_params()
checkpoint_path = "C:/Users/PC/Desktop/data/checkpoints/epoch_{epoch:03d}.ckpt"

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
hist = model.fit(X_train, Y_train, batch_size=64, epochs=100,
                 verbose=2, validation_data=(X_val, Y_val), shuffle=True,
                 class_weight=class_weights, callbacks=[early_stopping, cp_callback])

###############################################################################
# make prediction on test set.
###############################################################################
model.save('C:/Users/PC/Desktop/data/prepro/finetunning_model')
"""
probs = model.predict(X_test)
preds = probs.argmax(axis=-1)

print(probs)
print(preds)
print(Y_test-1)


acc = np.mean(preds == (Y_test-1))
print("Classification accuracy: %f " % (acc))
