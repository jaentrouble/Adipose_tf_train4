import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import custom_layers as clayers

# Get inputs and return outputs

def hr_5_3_0(inputs):
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
        filters=[64],
        name='encoder_Fusion'
    )
    outputs = layers.Activation('linear', dtype='float32')(x[0])
    return outputs

def hr_4_3_down2(inputs):
    x = inputs
    x = clayers.high_resolution_branch(
        inputs=x,
        filters=8,
        blocks=3,
        stride=2,
        name='bottleneck1'
    )
    x = clayers.high_resolution_branch(
        inputs=x,
        filters=16,
        blocks=3,
        stride=2,
        name='bottleneck2'
    )
    x = clayers.high_resolution_module(
        inputs=[x],
        filters=[32],
        blocks=[3],
        name='HR_0'
    )
    x = clayers.high_resolution_module(
        inputs=x,
        filters=[32,64],
        blocks=[3,3],
        name='HR_1'
    )
    x = clayers.high_resolution_module(
        inputs=x,
        filters=[32,64,128],
        blocks=[3,3,3],
        name='HR_2'
    )
    x = clayers.high_resolution_module(
        inputs=x,
        filters=[32,64,128],
        blocks=[3,3,3],
        name='HR_3'
    )
    x = clayers.high_resolution_fusion(
        inputs=x,
        filters=[64],
        name='encoder_Fusion'
    )
    outputs = layers.Activation('linear', dtype='float32')(x[0])
    return outputs
