import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import adipose_models_func
import model_lr
from model_trainer_func import AdiposeModel, create_train_dataset

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', dest='model')
parser.add_argument('-n','--name', dest='name')
parser.add_argument('--load',dest='load')
args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

model_f = getattr(adipose_models_func, args.model)
name = args.name
load_model_path = args.load
img_size = (200,200)

original_model = AdiposeModel(img_size, model_f)
original_model.compile(
    optimizer='adam',
    loss='mse',
)
original_model.load_weights(load_model_path)
print('loaded from : ' + load_model_path)

lr_f = model_lr.zero_lr
lr_callback = keras.callbacks.LearningRateScheduler(lr_f, verbose=1)

dummy_x = tf.zeros((1,img_size[1],img_size[0],3))
dummy_y = tf.zeros((1,img_size[1],img_size[0]))
# To draw graph
# lr is 0, so no update will happen
original_model.fit(
    x=dummy_x,
    y=dummy_y,
    epochs=1,
    callbacks=[lr_callback],
    verbose=1,
    batch_size=1,
)

converter = tf.lite.TFLiteConverter.from_keras_model(original_model)
tflite_model = converter.convert()

with open(f'tflite_models/{name}.tflite','wb') as f:
    f.write(tflite_model)