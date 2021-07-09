"""
 ARL_EEGModels - A collection of Convolutional Neural Network models for EEG
 Signal Processing and Classification, using Keras and Tensorflow
 Requirements:
    (1) tensorflow == 2.X (as of this writing, 2.0 - 2.3 have been verified
        as working)

 To run the EEG/MEG ERP classification sample script, you will also need
    (4) mne >= 0.17.1
    (5) PyRiemann >= 0.2.5
    (6) scikit-learn >= 0.20.1
    (7) matplotlib >= 2.2.3

 To use:

    (1) Place this file in the PYTHONPATH variable in your IDE (i.e.: Spyder)
    (2) Import the model as

        from EEGModels import EEGNet

        model = EEGNet(nb_classes = ..., Chans = ..., Samples = ...)

    (3) Then compile and fit the model

        model.compile(loss = ..., optimizer = ..., metrics = ...)
        fitted    = model.fit(...)
        predicted = model.predict(...)
 Portions of this project are works of the United States Government and are not
 subject to domestic copyright protection under 17 USC Sec. 105.  Those
 portions are released world-wide under the terms of the Creative Commons Zero
 1.0 (CC0) license.

 Other portions of this project are subject to domestic copyright protection
 under 17 USC Sec. 105.  Those portions are licensed under the Apache 2.0
 license.  The complete text of the license governing this material is in
 the file labeled LICENSE.TXT that is a part of this project's official
 distribution.

 source: https://github.com/vlawhern/arl-eegmodels
"""
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Concatenate, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

def Multi_DS_EEGNet(nb_classes, Chans=64, Samples=1536,
           dropoutRate=0.5, kernLength=256, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    # model for sampling 512
    input = Input(shape=(Chans, Samples, 1))

    ##################################################################
    # padding = 'valid' => no padding
    # padding = 'same' => output has the same height/width dimension as the input
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('ReLU')(block1)
    block1 = AveragePooling2D((1, 8))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 32),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('ReLU')(block2)
    block2 = AveragePooling2D((1, 4))(block2)
    block2 = dropoutType(dropoutRate)(block2)
    flatten1 = Flatten(name='flatten1')(block2)

    # model for sampling 128Hz
    ##################################################################
    # padding = 'valid' => no padding
    # padding = 'same' => output has the same height/width dimension as the input
    flatten2 = fake_inception(D,F2,64,Chans,input,dropoutRate,dropoutType)
    flatten3 = fake_inception(D,F2,128,Chans,input,dropoutRate,dropoutType)
    flatten = Concatenate()([flatten1,flatten2,flatten3])
    dense2 = Dense(nb_classes, name='dense2', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense2)

    return Model(inputs=input, outputs=softmax)

def fake_inception(D,F2,kernLength,Chans, input,dropoutRate,dropoutType):
    block3 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(input)
    block3 = BatchNormalization()(block3)
    block3 = Activation('ReLU')(block3)
    block3 = AveragePooling2D((1, 8))(block3)
    block3 = dropoutType(dropoutRate)(block3)

    block4 = SeparableConv2D(F2, (1, kernLength),
                             use_bias=False, padding='same')(block3)
    block4 = BatchNormalization()(block4)
    block4 = Activation('ReLU')(block4)
    block4 = AveragePooling2D((1, 4))(block4)
    block4 = dropoutType(dropoutRate)(block4)
    return Flatten(name='flatten'+str(kernLength))(block4)
   
   
def EEGNet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=256, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    """ Keras Implementation of EEGNet
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    # padding = 'valid' => no padding
    # padding = 'same' => output has the same height/width dimension as the input
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('ReLU')(block1)
    block1 = AveragePooling2D((1, 8))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 32),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('ReLU')(block2)
    block2 = AveragePooling2D((1, 4))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense2 = Dense(nb_classes, name='dense2',
                  kernel_constraint=max_norm(norm_rate))(flatten)


    softmax = Activation('softmax', name='softmax')(dense2)

    return Model(inputs=input1, outputs=softmax)


def EEGNet_new(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=256, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))
    ##################################################################
    # Upgrade version of EEGNet maked py artigence
    # padding = 'valid' => no padding
    # padding = 'same' => output has the same height/width dimension as the input
    block1 = Conv2D(F1, (Chans, 1), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)

    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, F1),
                    use_bias=False)(block1)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 8))(block1)
    block1 = dropoutType(dropoutRate)(block1)


    block2 = SeparableConv2D(F2, (1,32),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 4))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense2 = Dense(nb_classes, name='dense2',
                  kernel_constraint=max_norm(norm_rate))(flatten)


    softmax = Activation('softmax', name='softmax')(dense2)

    return Model(inputs=input1, outputs=softmax)

def Multi_input_EEGNet(nb_classes, Chans=64, Samples1=1536,
           dropoutRate=0.5, kernLength1=256, kernLength2 = 64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    # model for sampling 512
    input1 = Input(shape=(Chans, Samples1, 1))

    ##################################################################
    # padding = 'valid' => no padding
    # padding = 'same' => output has the same height/width dimension as the input
    block1 = Conv2D(F1, (1, kernLength1), padding='same',
                    input_shape=(Chans, Samples1, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('ReLU')(block1)
    block1 = AveragePooling2D((1, 8))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 32),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('ReLU')(block2)
    block2 = AveragePooling2D((1, 4))(block2)
    block2 = dropoutType(dropoutRate)(block2)
    flatten1 = Flatten(name='flatten1')(block2)

    # model for sampling 128Hz
    ##################################################################
    # padding = 'valid' => no padding
    # padding = 'same' => output has the same height/width dimension as the input
    block3 = DepthwiseConv2D((Chans, 1), use_bias=False, strides=4,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(input1)
    block3 = BatchNormalization()(block3)
    block3 = Conv2D(F1, (1, kernLength2), padding='same',
                    input_shape=(Chans, Samples1, 1),
                    use_bias=False)(block3)
    block3 = BatchNormalization()(block3)
    block3 = Activation('ReLU')(block3)
    block3 = AveragePooling2D((1, 8))(block3)
    block3 = dropoutType(dropoutRate)(block3)

    block4 = SeparableConv2D(F2, (1, 32),
                             use_bias=False, padding='same')(block3)
    block4 = BatchNormalization()(block4)
    block4 = Activation('ReLU')(block4)
    block4 = AveragePooling2D((1, 4))(block4)
    block4 = dropoutType(dropoutRate)(block4)

    flatten2 = Flatten(name='flatten2')(block4)

    flatten = Concatenate()([flatten1,flatten2])
    dense2 = Dense(nb_classes, name='dense2', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense2)

    return Model(inputs=input1, outputs=softmax)

