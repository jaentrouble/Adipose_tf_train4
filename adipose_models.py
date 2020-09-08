import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import custom_layers as clayers

# Get inputs and return outputs
# Don't forget to squeeze output

def conv4_b2_0(inputs):
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def conv16_b4_0(inputs):
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def conv16_b4_1(inputs):
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def res_4_2_0_noBN(inputs):
    # To make the same filter size
    xi = layers.Conv2D(32, 1, padding='same', activation='linear')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.ReLU()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def res_4_2_0_BN(inputs):
    # To make the same filter size
    xi = layers.Conv2D(32, 1, padding='same', activation='linear')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def res_12_2_0_BN(inputs):
    # To make the same filter size
    xi = layers.Conv2D(32, 1, padding='same', activation='linear')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def hr_2_2_0(inputs):
    # WRONG NAME
    # Should be hr_3_2_0 
    x = [inputs]
    x = clayers.HighResolutionModule(
        filters=[8],
        blocks=[2],
        name='HR_0'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16],
        blocks=[2,2],
        name='HR_1'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16,32],
        blocks=[2,2,2],
        name='HR_2'
    )(x)
    x = clayers.HighResolutionFusion(
        filters=[8],
        name='Fusion_0'
    )(x)
    x = layers.Conv2D(
        1,
        1,
        padding='same',
        name='Final_conv'
    )(x[0])
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def hr_5_3_0(inputs):
    x = [inputs]
    x = clayers.HighResolutionModule(
        filters=[8],
        blocks=[3],
        name='HR_0'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16],
        blocks=[3,3],
        name='HR_1'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16,32],
        blocks=[3,3,3],
        name='HR_2'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16,32,64],
        blocks=[3,3,3,3],
        name='HR_3'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16,32,64],
        blocks=[3,3,3,3],
        name='HR_4'
    )(x)
    x = clayers.HighResolutionFusion(
        filters=[8],
        name='Fusion_0'
    )(x)
    x = layers.Conv2D(
        1,
        1,
        padding='same',
        name='Final_conv'
    )(x[0])
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs
