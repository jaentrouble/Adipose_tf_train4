import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import time
from custom_tqdm import TqdmNotebookCallback
from tqdm.keras import TqdmCallback
import albumentations as A


class AdiposeModel(keras.Model):
    def __init__(self, inputs, model_function):
        """
        Because of numerical stability, softmax layer should be
        taken out, and use it only when not training.
        Args
            inputs : keras.Input
            model_function : function that takes keras.Input and returns
            output tensor of logits
        """
        super().__init__()
        outputs = model_function(inputs)
        self.logits = keras.Model(inputs=inputs, outputs=outputs)
        self.logits.summary()
        
    def call(self, inputs, training=None):
        casted = tf.cast(inputs, tf.float32) / 255.0
        if training:
            return self.logits(inputs, training=training)
        return tf.math.sigmoid(self.logits(inputs, training=training))

class AugGenerator():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.n = self.X.shape[0]
        self.aug = A.Compose([
            A.OneOf([
                A.RandomGamma((40,200),p=1),
                A.RandomBrightness(p=1),
                A.RandomContrast(p=1),
                A.RGBShift(p=1),
            ], p=0.8),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1)
        ])
        self.idx = 0

    def __iter__(self):
        return self
    
    def __call__(self, *args):
        return self

    def __next__(self):
        distorted = self.aug(
            image=self.X[self.idx],
            mask=self.Y[self.idx],
        )
        self.idx += 1
        self.idx = self.idx % self.n
        return distorted['image'], distorted['mask']

def create_train_dataset(X, Y, batch_size):
    autotune = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_generator(
        AugGenerator,
        (tf.uint8, tf.float32),
        (tf.TensorShape(X.shape[1:]), tf.TensorShape(Y.shape[1:])),
        args = [X,Y],
    )
    dataset = dataset.shuffle(X.shape[0])
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(autotune)
    dataset = dataset.repeat()

    return dataset


def get_model(model_f):
    """
    To get model only and load weights.
    """
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)
    inputs = keras.Input((200,200,3))
    test_model = AdiposeModel(inputs, model_f)
    test_model.compile(
        optimizer='adam',
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.BinaryAccuracy(threshold=0.1),
        ]
    )
    return test_model

def run_training(
        model_f, 
        lr_f, 
        name, 
        epochs, 
        batch_size, 
        X_train, 
        Y_train, 
        val_data,
        mixed_float = True,
        notebook = True,
        augment = True,
    ):
    """
    val_data : (X_val, Y_val) tuple
    """
    if mixed_float:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
    
    st = time.time()

    inputs = keras.Input((200,200,3))
    mymodel = AdiposeModel(inputs, model_f)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    mymodel.compile(
        optimizer='adam',
        loss=loss,
        metrics=[
            keras.metrics.BinaryAccuracy(threshold=0.1),
        ]
    )

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

    if augment:
        train_ds = create_train_dataset(X_train, Y_train, batch_size)
        mymodel.fit(
            x=train_ds,
            epochs=epochs,
            steps_per_epoch=X_train.shape[0]//batch_size,
            callbacks=[
                tensorboard_callback,
                lr_callback,
                save_callback,
                tqdm_callback,
            ],
            verbose=0,
            validation_data=val_data,
        )


    else:
        mymodel.fit(
            x=X_train,
            y=Y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tensorboard_callback,
                lr_callback,
                save_callback,
                tqdm_callback,
            ],
            verbose=0,
            validation_data=val_data
        )

    print('Took {} seconds'.format(time.time()-st))

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    with np.load('cell_mask_data.npz') as data:
        X = data['img']
        Y = data['mask']
    ds = create_train_dataset(X,Y)
    fig = plt.figure()
    for image, mask in ds.take(1):
        for idx in range(6):
            ax = fig.add_subplot(6,2,2*idx+1)
            ax.imshow(image[idx])
            ax = fig.add_subplot(6,2,2*idx+2)
            ax.imshow(mask[idx])
    plt.show()