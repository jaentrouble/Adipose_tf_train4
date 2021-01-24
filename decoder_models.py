import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import custom_layers as clayers

def gaussian_heatmap(pos, shape, sigma=10):
    """
    Returns a heatmap of a point
    Shape Format : (HEIGHT, WIDTH)
    pos Format : xy (WIDTH, HEIGHT)
    x corresponds to column,
    y corresponds to row
    I.e. image[y,x] is the point

    Arguments
    ---------
    pos : tf.Tensor
        Shape : (Batch, 2)
        Format : (WIDTH, HEIGHT)
    shape : tf.Tensor
        (HEIGHT, WIDTH)

    Returns
    -------
    heatmap : tf.Tensor
        shape : (HEIGHT, WIDTH)
    """
    # Shape : (batch,2)
    pos = tf.cast(pos,tf.float32)
    # Shape : (height, width, 2)
    coordinates = tf.stack(tf.meshgrid(
        tf.range(shape[0],dtype=tf.float32),
        tf.range(shape[1],dtype=tf.float32),
        indexing='ij',
    ), axis=-1)
    keypoint = tf.reshape(pos[...,::-1],(-1,1,1,2))
    heatmap = tf.exp(-tf.reduce_sum((coordinates-keypoint)**2,axis=-1)\
                        /(2*sigma**2))

    return heatmap



def branch_3_64_up2(inputs):
    """
    Expects featuremap to be 1/4 (not area) of the original size

    inputs: list of tensors
        [encoded_image, mouse_pos]
        mouse_pos Format: (x, y)
    """
    encoded_image, mouse_pos = inputs
    # Encoded_image.shape (batch, h, w, c)
    image_shape = encoded_image.shape[1:3]
    mouse_pos = tf.cast(mouse_pos, tf.float32)
    # Divide by 4 to match featuremap size
    mouse_pos = mouse_pos / 4
    heatmap = gaussian_heatmap(mouse_pos,image_shape,sigma=10)
    heatmap_expand = heatmap[...,tf.newaxis]

    concated = tf.concat([encoded_image, heatmap_expand],axis=-1,
                         name='heatmap_concat')
    x = clayers.high_resolution_branch(
        inputs=concated,
        filters=64,
        blocks=3,
        name='decoder_branch'
    )
    x = clayers.upscale_block(
        inputs=x,
        filters=64,
        name='upscale1'
    )
    x = clayers.upscale_block(
        inputs=x,
        filters=64,
        name='upscale2'
    )
    x = layers.Conv2D(
        filters=1,
        kernel_size=1,
        padding='same',
        name='decoder_squeeze_conv'
    )(x)
    x = tf.squeeze(x, name='decoder_squeeze')
    outputs = layers.Activation('linear',dtype='float32')(x)
    return outputs