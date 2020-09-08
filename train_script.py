import numpy as np
from model_trainer import run_training
import adipose_models
import model_lr
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', dest='model')
parser.add_argument('-lr', dest='lr')
parser.add_argument('-n','--name', dest='name')
parser.add_argument('-e','--epochs', dest='epochs')
parser.add_argument('-mf','--mixedfloat', dest='mixed_float', 
                    action='store_true',default=False)
parser.add_argument('-mg','--memorygrow', dest='mem_growth',
                    action='store_true',default=False)
args = parser.parse_args()

if args.mem_growth:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

with np.load('cell_mask_data.npz') as data:
    X = data['img']
    Y = data['mask']

print('X shape:',X.shape)
print('Y shape:',Y.shape)

X_train = X[:1200]
Y_train = Y[:1200]
X_val = X[1200:1350]
Y_val = Y[1200:1350]
X_test = X[1350:]
Y_test = Y[1350:]

model_f = getattr(adipose_models, args.model)
lr_f = getattr(model_lr, args.lr)
name = args.name
epochs = int(args.epochs)
mixed_float = args.mixed_float

kwargs = {}
kwargs['model_f'] = model_f
kwargs['lr_f'] = lr_f
kwargs['name'] = name
kwargs['epochs'] = epochs
kwargs['batch_size'] = 32
kwargs['X_train'] = X_train
kwargs['Y_train'] = Y_train
kwargs['val_data'] = (X_val, Y_val)
kwargs['mixed_float'] = mixed_float
kwargs['notebook'] = False
kwargs['augment'] = True

run_training(**kwargs)