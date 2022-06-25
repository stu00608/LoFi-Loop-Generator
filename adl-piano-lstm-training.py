import pickle
import os
from wandb.keras import WandbCallback
import wandb
from utils import midi_encoder as me
from glob import glob
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import pretty_midi
import numpy as np
import tensorflow as tf
import yaml

config = yaml.safe_load(open('config.yaml', 'r'))
params = config['params']

name = 'adl-dataset-LSTM-test'


def __split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = tf.one_hot(chunk[1:], params['vocab_size'])
    return input_text, target_text


def loadMidiInFolder(path):
    '''
    path:
        path for glob to grab midi. Must include last file name like `*.mid`.

    Returns a list included concatenated words.
    '''
    wordsList = []
    # temp = []
    for mid in tqdm(glob(path, recursive=True)):
        try:
            midiData = pretty_midi.PrettyMIDI(mid)
        except:
            print("Cannot load this midi")
        text = me.midi2text(midiData)[:-1]
        wordsList += text
        # temp.append(len(text))

    # print(f"Average length = {np.average(temp)}")
    # print(f"Min length = {np.max(temp)}")
    # print(f"Max length = {np.min(temp)}")

    return wordsList


def makeDataset(data):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(params['sequence_length'], drop_remainder=True)
    dataset = dataset.map(__split_input_target)
    dataset = dataset.shuffle(10000).batch(params['batch_size'], drop_remainder=True)
    return dataset


datasetPath = 'data/adl-train'
if os.path.exists(datasetPath):
    with open(datasetPath, "rb") as f:
        trainWordsList = pickle.load(f)
else:
    trainWordsList = loadMidiInFolder(config['path']['data_dir'])
    with open(datasetPath, "wb") as f:
        pickle.dump(trainWordsList, f)

testWordsList = loadMidiInFolder(config['path']['test_dir'])

vocabularies = set(trainWordsList)
# vocabularies = set(trainWordsList + testWordsList)
params['vocab_size'] = len(vocabularies)

word2idx = {word: i for i, word in enumerate(vocabularies)}
idx2word = {idx: char for char, idx in word2idx.items()}


trainDataRaw = np.array([word2idx[w] for w in trainWordsList])
trainDataset = makeDataset(trainDataRaw)

print(trainDataset)
input()

model = Sequential()
model.add(Embedding(params['vocab_size'], output_dim=params['embed_size'],
                    batch_input_shape=(params['batch_size'], None)))
for _ in range(params['layers']):
    model.add(LSTM(params['unit'], return_sequences=True, stateful=params['stateful'],
                   dropout=params['dropout'], recurrent_dropout=params['dropout']))
    # model.add(BatchNormalization())

model.add(Dense(params['unit'], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(params['dropout']))
model.add(Dense(params['vocab_size'], activation='softmax'))

model.compile(loss=CategoricalCrossentropy(from_logits=True), optimizer=Adam(), metrics=['acc'])

wandb.init(config=params, project='lstm-adl-dataset')
wandb_callback = WandbCallback(
    log_weights=True, log_evaluation=False, validation_steps=5)

filepath = config['path']['model_dir'] + \
    "model-"+name+"-{epoch:02d}-{loss:.4f}.hdf5"
model_checkpoint_callback = ModelCheckpoint(
    filepath=filepath,
    monitor='loss',
    mode='min',
    save_weights_only=True,
    save_best_only=True)

history = model.fit(trainDataset, batch_size=params['batch_size'], epochs=params['epochs'], verbose=1, callbacks=[
                    wandb_callback, model_checkpoint_callback])
