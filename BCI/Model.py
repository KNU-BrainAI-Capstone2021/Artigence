
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
    block1 = Activation('relu')(block1)
    block1 = AveragePooling2D((1, 8))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 32),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)
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
    block3 = Activation('relu')(block3)
    block3 = AveragePooling2D((1, 8))(block3)
    block3 = dropoutType(dropoutRate)(block3)

    block4 = SeparableConv2D(F2, (1, kernLength),
                             use_bias=False, padding='same')(block3)
    block4 = BatchNormalization()(block4)
    block4 = Activation('relu')(block4)
    block4 = AveragePooling2D((1, 4))(block4)
    block4 = dropoutType(dropoutRate)(block4)
    return Flatten(name='flatten'+str(kernLength))(block4)