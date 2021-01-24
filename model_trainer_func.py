import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision, layers
import time
from custom_tqdm import TqdmNotebookCallback
from tqdm.keras import TqdmCallback
import albumentations as A
import random
import io
import matplotlib.pyplot as plt
from functools import partial
import numpy as np

def AdiposeModel(
    image_size,
    encoder_f,
    decoder_f,
):
    """AdiposeModel

    Because of numerical stability, softmax layer should be
    taken out, and use it only when not training.

    Inputs are expected to be uint8 dtype.
    Rescaling will be performed in this model.

    Arguments
    ---------
        image_size : tuple
            Format (WIDTH, HEIGHT)
        encoder_f : a function that takes keras.Input and returns
        a feature map of the entire image
        decoder_f : a function that takes a feature map and mouse position
        and returns a mask indicating the cell pointed by the mouse
    Return
    ------
        adipose model : keras.Model
            Using only functional API
    """
    # Encoder model
    image_input = keras.Input([image_size[1], image_size[0], 3],
                            name='image_input',dtype=tf.uint8)
    rescaled = layers.experimental.preprocessing.Rescaling(
        scale=1./255, offset=0, name='255_to_1_rescale'
    )(image_input)
    encoded_image = encoder_f(rescaled)
    encoder = keras.Model(inputs=image_input, outputs=encoded_image,
                          name='encoder')

    # Decoder model
    feature_input = keras.Input(tensor=encoded_image,
                                name='feature_input')
    mouse_input = keras.Input([2],name='mouse_input')
    mask = decoder_f([feature_input, mouse_input])
    decoder = keras.Model(inputs=[feature_input, mouse_input],
                          outputs=mask, name='decoder')

    # Final model
    # Divide encoder/decoder so that decoder can be used seperately
    image_input = keras.Input([image_size[1], image_size[0], 3],
                            name='image',dtype=tf.uint8)
    encoded_image = encoder(image_input)
    mouse_input = keras.Input([2],name='mouse')
    mask = decoder([encoded_image,mouse_input])

    adipose_model = keras.Model(inputs=[image_input, mouse_input], 
                                outputs=mask,
                                name='adipose_model')
    return adipose_model


class AugGenerator():
    """An iterable generator that makes data

    NOTE: 
        Every img is reshaped to img_size
    NOTE: 
        The position value is like pygame. (width, height),
        which does not match with common image order (height,width)

        Image input is expected to be the shape of (height, width),
        i.e. the transformation to match two is handled in here automatically
    NOTE: 
        THE OUTPUT IMAGE WILL BE (HEIGHT, WIDTH)
    return
    ------
    X : dictionary
        'image'
            np.array, dtype= np.uint8
            shape : (HEIGHT, WIDTH, 3)
        'mouse'
            np.array, dtype = np.int64
            shape : (2,)
            Format : (X, Y)
    Y : np.array, dtype= np.float32
        shape : (HEIGHT, WIDTH)
    """
    def __init__(self, img, data, img_size):
        """ 
        arguments
        ---------
        img : list
            list of images, in the original size (height, width, 3)
        data : list of dict
            Each dict has :
                'image' : index of the image. The index should match with img
                'mask' : [xx, yy]
                        IMPORTANT : (WIDTH, HEIGHT)
                'box' : [[xmin, ymin], [xmax,ymax]]
                'size' : the size of the image that the data was created with
                        IMPORTANT : (WIDTH, HEIGHT)
        img_size : tuple
            Desired output image size
            IMPORTANT : (WIDTH, HEIGHT)
        """
        self.image = img
        self.data = data
        self.n = len(data)
        self.aug = A.Compose([
            A.OneOf([
                A.RandomGamma((40,200),p=1),
                A.RandomBrightness(limit=0.5, p=1),
                A.RandomContrast(limit=0.5,p=1),
                A.RGBShift(40,40,40,p=1),
                A.Downscale(scale_min=0.25,scale_max=0.5,p=1),
                A.ChannelShuffle(p=1),
            ], p=0.8),
            A.InvertImg(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1),
            A.Resize(img_size[1], img_size[0]),
        ],
        keypoint_params=A.KeypointParams(format='xy')
        )
        for datum in data:
            datum['mask_min'] = np.min(datum['mask'], axis=1)
            datum['mask_max'] = np.max(datum['mask'], axis=1) + 1

    def __iter__(self):
        return self
    
    def __call__(self, *args):
        return self

    def __next__(self):
        idx = random.randrange(0,self.n)
        datum = self.data[idx]
        image = self.image[datum['image']]
        x_min, y_min = datum['mask_min']
        x_max, y_max = datum['mask_max']
        size_x, size_y = datum['size']

        min_half_ratio = 1./4
        max_half_ratio = 1.
        min_half_x = int(size_x * min_half_ratio)
        max_half_x = int(size_x * max_half_ratio)
        
        min_half_y = int(size_y * min_half_ratio)
        max_half_y = int(size_y * max_half_ratio)

        crop_min = (max(0, x_min-random.randrange(min_half_x,max_half_x)),
                    max(0, y_min-random.randrange(min_half_y,max_half_y)))
        crop_max = (min(size_x,x_max+random.randrange(min_half_x,max_half_x)),
                    min(size_y,y_max+random.randrange(min_half_y,max_half_y)))
        new_mask = np.zeros(np.subtract(crop_max, crop_min), dtype=np.float32)
        xx, yy = np.array(datum['mask'],dtype=np.int)
        m_xx = xx - crop_min[0]
        m_yy = yy - crop_min[1]
        # (Width, Height)
        new_mask[m_xx,m_yy] = 1
        # (Height, Width)
        new_mask = np.swapaxes(new_mask, 0, 1)

        mouse_idx = random.randrange(len(xx))
        mouse = np.array([m_xx[mouse_idx], m_yy[mouse_idx]])

        if np.any(np.not_equal(image.shape[1::-1], datum['size'])):
            row_ratio = image.shape[0] / size_y
            col_ratio = image.shape[1] / size_x
        else:
            row_ratio = 1
            col_ratio = 1

        mouse = mouse * [col_ratio, row_ratio]
        mouse = mouse.astype(np.int64)

        cx_min = int(col_ratio*crop_min[0])
        cy_min = int(row_ratio*crop_min[1])
        
        cx_max = int(col_ratio*crop_max[0])
        cy_max = int(row_ratio*crop_max[1])

        cropped_image = image[cy_min:cy_max,cx_min:cx_max]

        distorted = self.aug(
            image=cropped_image,
            mask =new_mask,
            keypoints=[mouse]
        )

        X = {
            'image':distorted['image'],
            'mouse':distorted['keypoints'][0]
        }

        Y = distorted['mask']

        return X, Y

class ValGenerator(AugGenerator):
    """Same as AugGenerator, but without augmentation.
    """
    def __init__(self, img, data, img_size):
        """ 
        arguments
        ---------
        img : list
            list of images, in the original size (height, width, 3)
        data : list of dict
            Each dict has :
                'image' : index of the image. The index should match with img
                'mask' : [xx, yy]
                        IMPORTANT : (WIDTH, HEIGHT)
                'box' : [[xmin, ymin], [xmax,ymax]]
                'size' : the size of the image that the data was created with
                        IMPORTANT : (WIDTH, HEIGHT)
        img_size : tuple
            Desired output image size
            The axes will be swapped to match pygame.
            IMPORTANT : (WIDTH, HEIGHT)
        """
        super().__init__(img, data, img_size)
        self.aug = A.Compose([
            A.Resize(img_size[1], img_size[0]),
        ],
        keypoint_params=A.KeypointParams(format='xy'),
        )
        

def create_train_dataset(
    img,
    data,
    img_size,
    batch_size,
    val_data=False
):
    """
    NOTE:
        img_size : (WIDTH, HEIGHT)
    """
    autotune = tf.data.experimental.AUTOTUNE
    if val_data:
        generator = ValGenerator(img, data, img_size)
    else:
        generator = AugGenerator(img, data, img_size)
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {
                'image':tf.TensorSpec(
                    shape=[img_size[1],img_size[0],3],
                    dtype=tf.uint8,),
                'mouse':tf.TensorSpec(
                    shape=[2,],
                    dtype=tf.int64,),
            },
            tf.TensorSpec(
                shape=[img_size[1],img_size[0]],
                dtype=tf.float32,)
        ),
    )
    dataset = dataset.shuffle(min(len(data),1000))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(autotune)
    dataset = dataset.repeat()

    return dataset



class ValFigCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, logdir):
        super().__init__()
        self.val_ds = val_ds
        self.filewriter = tf.summary.create_file_writer(logdir+'/val_image')

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def val_result_fig(self):
        sample = self.val_ds.take(1).as_numpy_iterator()
        sample = next(sample)
        sample_x = sample[0]
        sample_y = sample[1]
        predict = self.model(sample_x, training=False).numpy()
        fig = plt.figure(figsize=(20,20))
        for i in range(3):
            ax = fig.add_subplot(3,3,3*i+1)
            img = sample_x['image'][i]
            mouse = sample_x['mouse'][i]
            img[max(0,mouse[1]-5):mouse[1]+5,
                max(0,mouse[0]-5):mouse[0]+5] = [255,0,0]
            ax.imshow(img)
            ax = fig.add_subplot(3,3,3*i+2)
            true_mask = sample_y[i]
            ax.imshow(true_mask, cmap='binary')
            ax = fig.add_subplot(3,3,3*i+3)
            p = predict[i]
            ax.imshow(p, cmap='binary')
        return fig

    def on_epoch_end(self, epoch, logs=None):
        image = self.plot_to_image(self.val_result_fig())
        with self.filewriter.as_default():
            tf.summary.image('val prediction', image, step=epoch)

def run_training(
        encoder_f,
        decoder_f,
        lr_f, 
        name, 
        epochs, 
        batch_size, 
        train_data,
        val_data,
        img,
        img_size,
        load_path = None,
        mixed_float = True,
        notebook = True,
    ):
    """
    img_size : (WIDTH, HEIGHT)
    """
    if mixed_float:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    
    st = time.time()

    mymodel = AdiposeModel(img_size, encoder_f, decoder_f)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    mymodel.compile(
        optimizer='adam',
        loss=loss,
        metrics=[
            keras.metrics.BinaryAccuracy(threshold=0.5),
        ]
    )
    mymodel.summary()
    if load_path is not None:
        mymodel.load_weights(load_path)
        print('loaded from '+load_path)

    logdir = 'logs/fit/' + name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
        profile_batch='3,5',
        update_freq='epoch'
    )
    lr_callback = keras.callbacks.LearningRateScheduler(lr_f, verbose=1)

    savedir = 'savedmodels/' + name + '/{epoch}'
    save_callback = keras.callbacks.ModelCheckpoint(
        savedir,
        save_weights_only=True,
        verbose=1
    )

    if notebook:
        tqdm_callback = TqdmNotebookCallback(metrics=['loss', 'binary_accuracy'],
                                            leave_inner=False)
    else:
        tqdm_callback = TqdmCallback()

    train_ds = create_train_dataset(img, train_data, img_size,batch_size)
    val_ds = create_train_dataset(img, val_data, img_size,batch_size,True)

    image_callback = ValFigCallback(val_ds, logdir)

    mymodel.fit(
        x=train_ds,
        epochs=epochs,
        steps_per_epoch=len(train_data)//batch_size,
        callbacks=[
            tensorboard_callback,
            lr_callback,
            save_callback,
            tqdm_callback,
            image_callback,
        ],
        verbose=0,
        validation_data=val_ds,
        validation_steps=10,
    )


    print('Took {} seconds'.format(time.time()-st))

if __name__ == '__main__':
    import os
    import imageio as io
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import draw
    import cv2
    from pathlib import Path

    data_dir = Path('data')
    data_groups = next(os.walk(data_dir))[1]
    img = []
    data = []
    img_name_dict = {}
    img_idx = 0
    for dg in data_groups[:]:
        img_dir = data_dir/dg/'done'
        img_names = os.listdir(img_dir)
        for name in img_names:
            img_path = str(img_dir/name)
            img.append(io.imread(img_path))
            img_name_dict[img_path] = img_idx
            img_idx += 1

        json_dir = data_dir/dg/'save'
        json_names = os.listdir(json_dir)
        dg_data = []
        for name in json_names[:]:
            with open(str(json_dir/name),'r') as j:
                dg_data.extend(json.load(j))
        for dg_datum in dg_data :
            long_img_name = str(img_dir/dg_datum['image'])
            dg_datum['image'] = img_name_dict[long_img_name]
        data.extend(dg_data)

    # fig = plt.figure()
    # d_idx = random.randrange(0,len(data)-5)
    # for i, d in enumerate(data[d_idx:d_idx+5]):
    #     image = img[d['image']].copy()
    #     image = cv2.resize(image, (1200,900), interpolation=cv2.INTER_LINEAR)
    #     mask = d['mask']
    #     m_idx = random.randrange(0,len(mask[0]))
    #     pos = (mask[0][m_idx], mask[1][m_idx])
    #     boxmin = d['box'][0]
    #     boxmax = d['box'][1]
    #     rr, cc = draw.disk((pos[1],pos[0]),5)
    #     image[rr, cc] = [0,255,0]
    #     rr, cc = draw.rectangle_perimeter((boxmin[1],boxmin[0]),(boxmax[1],boxmax[0]))
    #     image[rr,cc] = [255,0,0]
    #     image[mask[1],mask[0]] = [100,100,100]
    #     ax = fig.add_subplot(5,1,i+1)
    #     ax.imshow(image)
    # plt.show()

    # gen = AugGenerator(img, data, (400,400))
    # s = next(gen)

    ds = create_train_dataset(img, data, (200,200),1, False)
    sample = ds.take(5).as_numpy_iterator()
    fig = plt.figure()
    for i, s in enumerate(sample):
        ax = fig.add_subplot(5,2,2*i+1)
        img = s[0][0].swapaxes(0,1)
        ax.imshow(img)
        ax = fig.add_subplot(5,2,2*i+2)
        mask = s[1][0].swapaxes(0,1)
        ax.imshow(mask)
    plt.show()