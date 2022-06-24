import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from dataloader import *
from model import *
import numpy as np
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

config = yaml.safe_load(open('config.yaml', 'r'))

train_dataset_path = "data/train_dataset"
test_dataset_path = "data/test_dataset"

train_dataset = Dataset(config['path']['data_dir'])
if os.path.exists(train_dataset_path):
    with open(train_dataset_path, "rb") as f:
        train_dataset = pickle.load(f)
else:
    train_dataset.load()
    with open(train_dataset_path, "wb") as f:
        pickle.dump(train_dataset, f)


test_dataset = Dataset(config['path']['test_dir'])
if os.path.exists(test_dataset_path):
    with open(test_dataset_path, "rb") as f:
        test_dataset = pickle.load(f)
else:
    test_dataset.load()
    with open(test_dataset_path, "wb") as f:
        pickle.dump(test_dataset, f)

beat_list = []
track = 0
for midi in train_dataset.encoded_data_list:
    beat_list += midi[track]
train_data = beat_list.copy()

beat_list = []
for midi in test_dataset.encoded_data_list:
    beat_list += midi[track]
test_data = beat_list.copy()

mySet = set(train_data) | set(test_data)
vocab_size = len(mySet)
print(np.array(beat_list).shape)

beat2idx = {beat: i for i, beat in enumerate(mySet)}


def __split_input_target(chunk):
    global vocab_size
    input_text = chunk[:-1]
    target_text = tf.one_hot(chunk[1:], vocab_size)
    return input_text, target_text


idxOfBeat = np.array([beat2idx[beat] for beat in train_data])
print(len(idxOfBeat))
tfDataset = tf.data.Dataset.from_tensor_slices(idxOfBeat)
sequences = tfDataset.batch(config['params']['sequence_length']+1, drop_remainder=True)
# o o o o o
#  \ \ \ \ \
#   x x x x x
ds = sequences.map(__split_input_target)
train_data = ds.shuffle(10000).batch(config['params']['batch_size'], drop_remainder=True)


idxOfBeat = np.array([beat2idx[beat] for beat in test_data])
print(len(idxOfBeat))
tfDataset = tf.data.Dataset.from_tensor_slices(idxOfBeat)
sequences = tfDataset.batch(config['params']['sequence_length']+1, drop_remainder=True)
# o o o o o
#  \ \ \ \ \
#   x x x x x
ds = sequences.map(__split_input_target)
test_data = ds.shuffle(10000).batch(config['params']['batch_size'], drop_remainder=True)

print(f'\nTraining dataset : {train_data}\n')


# model = LoFiLoopNet('1to1-dev', vocab_size)
# model.summary()

# model.train(train_data, test_data)
