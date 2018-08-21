from keras.layers import Dense, Conv3D, MaxPooling3D, Dropout, Flatten, concatenate, GRU, Input, BatchNormalization, \
    AveragePooling2D, Activation, Lambda, GlobalAveragePooling2D, Add, UpSampling2D, Conv3D, MaxPooling3D, GlobalAveragePooling3D, Activation, GRU
from keras.models import Model
import numpy
from keras import objectives
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.backend import squeeze
from keras.applications.inception_v3 import InceptionV3


def fire_module3D(id, squeeze, expand):
    """
    Fire module, as in paper.
    :param input_, id, squeeze, expand: input tensor (object).
    :return: as in paper.
    """

    def layer(input_):
        with tf.name_scope('fire_module3D_%i' % id):
            conv_squeezed = Conv3D(squeeze, (1, 1, 1), padding='valid', name='fm_%i_s1x1' % id, activation='relu')(input_)

            left = Conv3D(expand, (1, 1, 1), padding='valid', name='fm_%i_e1x1' % id, activation='relu')(conv_squeezed)

            right = Conv3D(expand, (3, 3, 3), padding='same', name='fm_%i_e3x3' % id, activation='relu')(conv_squeezed)

            out = concatenate([left, right], axis=-1, name='fire_module3D_%i' % id + 'concat')

            return out

    return layer


def SqueezeNetOutput3D(input_, num_classes=4, bypass=None):
    valid = [None, 'simple', 'complex']
    if bypass not in valid:
        raise UserWarning('"bypass" argument must be one of %s.' % ', '.join(map(str, valid)))

    conv_0 = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='valid', name='conv_0', activation='relu')(input_)
    mxp_0 = MaxPooling3D(pool_size=(3, 3, 2), strides=(2, 2, 1), name='pool_0')(conv_0)

    # Block 1
    fm_2 = fire_module3D(id=2, squeeze=16, expand=64)(mxp_0)
    fm_3 = fire_module3D(id=3, squeeze=16, expand=64)(fm_2)
    input_fm_4_ = fm_3
    if bypass == 'simple':
        input_fm_4_ = Add()([fm_2, fm_3])
    fm_4 = fire_module3D(id=4, squeeze=32, expand=128)(input_fm_4_)
    mxp_1 = MaxPooling3D(pool_size=(3, 3, 2), strides=(2, 2, 2), name='pool_1')(fm_4)

    # Block 2
    fm_5 = fire_module3D(id=5, squeeze=32, expand=128)(mxp_1)
    input_fm_6_ = fm_5
    if bypass == 'simple':
        input_fm_6_ = Add()([mxp_1, fm_5])
    fm_6 = fire_module3D(id=6, squeeze=48, expand=192)(input_fm_6_)
    fm_7 = fire_module3D(id=7, squeeze=48, expand=192)(fm_6)
    input_fm_8_ = fm_7
    if bypass == 'simple':
        input_fm_8_ = Add()([fm_6, fm_7])
    fm_8 = fire_module3D(id=8, squeeze=64, expand=256)(input_fm_8_)
    mxp_2 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='pool_2')(fm_8)

    # Block 3
    fm_9 = fire_module3D(id=9, squeeze=64, expand=256)(mxp_2)
    input_conv_10_ = fm_9
    if bypass == 'simple':
        input_conv_10_ = Add()([mxp_2, fm_9])
    dropped = Dropout(0.5, name='Dropout')(input_conv_10_)
    conv_10 = Conv3D(num_classes, (1, 1, 1), padding='valid', name='conv10', activation='relu')(dropped)
    normalized = BatchNormalization(name='batch_normalization')(conv_10)

    print(normalized.shape)

    # Predictions
    avgp_0 = GlobalAveragePooling3D(name='globalaveragepooling')(normalized)
    probas = Activation('softmax', name='probabilities')(avgp_0)

    return probas


def NaiveModelOutput(input_, num_classes=4):
    x = Conv3D(32, kernel_size=(3, 3, 3), padding='same', activation='relu')(input_)
    x = Conv3D(32, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(x)
    x = Dropout(0.25)(x)

    x = Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(x)
    x = Dropout(0.25)(x)

    x = Conv3D(128, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(x)
    x = Dropout(0.25)(x)
    print(x.shape)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    return x

def VGG16Output(input_, num_classes=4):
    x = Conv3D(64, (3, 3, 3), padding='same', name='block1_conv1', activation='relu')(input_)
    x = BatchNormalization()(x)
    
    x = Conv3D(64, (3, 3, 3), padding='same', name='block1_conv2', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x)

    x = Conv3D(128, (3, 3, 3), padding='same', name='block2_conv1', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv3D(128, (3, 3, 3), padding='same', name='block2_conv2', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(x)

    x = Conv3D(256, (3, 3, 3), padding='same', name='block3_conv1', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(256, (3, 3, 3), padding='same', name='block3_conv2', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(256, (3, 3, 3), padding='same', name='block3_conv3', activation='relu')(x)
    x = BatchNormalization()(x)
    

    x = Conv3D(256, (3, 3, 3), padding='same', name='block3_conv4', activation='relu')(x)
    x = BatchNormalization()(x)
    

    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block3_pool')(x)

    x = Conv3D(512, (3, 3, 3), padding='same', name='block4_conv1', activation='relu')(x)
    x = BatchNormalization()(x)
    

    x = Conv3D(512, (3, 3, 3), padding='same', name='block4_conv2', activation='relu')(x)
    x = BatchNormalization()(x)
    

    x = Conv3D(512, (3, 3, 3), padding='same', name='block4_conv3', activation='relu')(x)
    x = BatchNormalization()(x)
    

    x = Conv3D(512, (3, 3, 3), padding='same', name='block4_conv4', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool')(x)

    x = Conv3D(512, (3, 3, 3), padding='same', name='block5_conv1', activation='relu')(x)
    x = BatchNormalization()(x)
    

    x = Conv3D(512, (3, 3, 3), padding='same', name='block5_conv2', activation='relu')(x)
    x = BatchNormalization()(x)
    

    x = Conv3D(512, (3, 3, 3), padding='same', name='block5_conv3', activation='relu')(x)
    x = BatchNormalization()(x)
    

    x = Conv3D(512, (3, 3, 3), padding='same', name='block5_conv4', activation='relu')(x)
    x = BatchNormalization()(x)

    print(x.shape)
    

    x = Flatten()(x)

    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Dropout(0.5)(x)

    x = Dense(4096, name='fc2', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Dropout(0.5)(x)

    x = Dense(num_classes)(x)
    x = BatchNormalization()(x)
    x = Activation('softmax')(x)

    return x


def IRNNOutput(input_, num_classes=4):

	base = InceptionV3(input_tensor = input_, include_top=False, weights='imagenet')

	pass



