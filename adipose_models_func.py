import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import custom_layers as clayers

# Get inputs and return outputs

def hr_5_3_0_func(inputs):
    x = [inputs]
    x = clayers.high_resolution_module(
        inputs=x,
        filters=[8],
        blocks=[3],
        name='HR_0'
    )
    x = clayers.high_resolution_module(
        inputs=x,
        filters=[8,16],
        blocks=[3,3],
        name='HR_1'
    )
    x = clayers.high_resolution_module(
        inputs=x,
        filters=[8,16,32],
        blocks=[3,3,3],
        name='HR_2'
    )
    x = clayers.high_resolution_module(
        inputs=x,
        filters=[8,16,32,64],
        blocks=[3,3,3,3],
        name='HR_3'
    )
    x = clayers.high_resolution_module(
        inputs=x,
        filters=[8,16,32,64],
        blocks=[3,3,3,3],
        name='HR_4'
    )
    x = clayers.high_resolution_fusion(
        inputs=x,
        filters=[8],
        name='FinalFusion'
    )
    x = layers.Conv2D(
        1,
        1,
        padding='same',
        name='Conv_squeeze'
    )(x[0])
    x = tf.squeeze(x,axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs
