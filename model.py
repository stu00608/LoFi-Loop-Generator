from dataloader import *
from datetime import datetime
from wandb.keras import WandbCallback
import wandb
import os
from glob import glob
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from gensim.models import Word2Vec
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

config = yaml.safe_load(open('config.yaml', 'r'))
wandb.init(config=config, project='Shen-midi-lstm-experiment')


class BeatDictionary():

    def __init__(self, data) -> None:
        self.data = data
        self.build()

    def build(self):
        model = Word2Vec(self.data, min_count=1, window=4,
                         vector_size=config['params']['vocab_size'], workers=4)
        self.word_dict = model.wv
        self.vector_dict = model.wv.index_to_key

    def vectorize(self, beats):
        if isinstance(beats, list):
            result = []
            for beat in beats:
                result.append(self.word_dict[beat])
            return np.array(result)
        else:
            return self.word_dict[beats]

    def find_beat(self, vectors):
        # TODO: Now is only for inferencing model predict. Should be more general.

        return self.word_dict.similar_by_vector(vectors, topn=1)[0]

    def get_related_chords(self, token, topn=3):
        print("Similar chords with " + token)
        for word, similarity in self.word_dict.most_similar(positive=[token], topn=topn):
            print(word, round(similarity, 3))


class LoFiLoopNet():

    def __init__(self, name, input_shape) -> None:
        self.name = name
        self.build(input_shape)

    def build(self, input_shape):
        model = Sequential()
        model.add(LSTM(
            256,
            input_shape=input_shape,
            return_sequences=True,
        ))
        model.add(Dropout(0.2))
        model.add(LSTM(512))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(config['params']['vocab_size'], activation='sigmoid'))
        model.compile(loss='mae', optimizer='adam', metrics=['acc'])

        self.model = model

        self.model_counter = len(os.listdir(config['path']['model_dir']))+1
        self.model_checkpoint_callback = ModelCheckpoint(
            filepath=config['path']['model_dir'] +
            f'model_{self.name}_{datetime.now().strftime("%m-%d_%H-%M")}_{self.model_counter}.hdf5',
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True)

        self.wandb_callback = WandbCallback(
            log_weights=True, log_evaluation=True, validation_steps=5)

    def summary(self):
        self.model.summary()

    def train(self, x_train, y_train, verbose=1):
        self.model_counter = len(os.listdir(config['path']['model_dir']))+1
        self.history = self.model.fit(x=x_train, y=y_train, batch_size=config['params']['batch_size'], epochs=config['params']['epochs'], validation_split=0.1, verbose=verbose, callbacks=[
                                      self.model_checkpoint_callback, self.wandb_callback])

    def load_last_weight(self):

        model_list = glob(config['path']['model_dir']+'*.hdf5')
        latest_model = max(model_list, key=os.path.getctime)
        self.model.load_weights(latest_model)

        print(f"Loaded weight {os.path.split(latest_model)[-1]}")

    def generate(self, dataset: Dataset, beatdict: BeatDictionary, filename=str(len(os.listdir(config['path']['out_dir']))+1), length=config['output']['length'], bpm=config['params']['bpm'], track=0):

        beats = bpm//60*length
        # NOTE: Use only track 0 for now.
        pick = random.randint(0, 500)

        picked_seed = dataset.encoded_midi_list[pick]
        picked_seed = picked_seed.all_encoded_beats[track][:config['params']['data_size']]
        midi_sequence = picked_seed.copy()
        picked_seed = beatdict.vectorize(picked_seed)
        picked_seed = np.expand_dims(picked_seed, axis=0)
        print(picked_seed.shape)

        for _ in range(beats):
            next_beat = self.model.predict([picked_seed], verbose=0)
            (key, similarity) = beatdict.find_beat(next_beat[0])

            midi_sequence = np.append(midi_sequence, [key], axis=0)
            print(midi_sequence)

            picked_seed = np.append(picked_seed[0], [beatdict.word_dict[key]], axis=0)
            picked_seed = picked_seed[1:, :]
            picked_seed = np.expand_dims(picked_seed, axis=0)

            print(picked_seed.shape)

        return midi_sequence
