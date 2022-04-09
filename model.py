import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import yaml
import random
from glob import glob
import os
from datetime import datetime
from dataloader import *

config = yaml.safe_load(open('config.yaml', 'r'))

class LoFiLoopNet():

    def __init__(self) -> None:
        self.build()

    def build(self):
        self.model = Sequential()
        self.model = Sequential()
        self.model.add(LSTM(512, input_shape=(config['params']['data_size'], config['params']['vocab_size'])))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(config['params']['vocab_size'], activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model_counter = len(os.listdir(config['path']['model_dir']))+1
        self.model_checkpoint_callback = ModelCheckpoint(
            filepath=config['path']['model_dir']+f'model_{datetime.now().strftime("%m-%d_%H-%M")}_{self.model_counter}.hdf5',
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True)
    
    def summary(self):
        self.model.summary()

    def train(self, x_train, y_train, verbose=1):
        self.model_counter = len(os.listdir(config['path']['model_dir']))+1
        self.history = self.model.fit(x = x_train, y = y_train, batch_size=config['params']['batch_size'], epochs=config['params']['epochs'], verbose=verbose, callbacks=[self.model_checkpoint_callback])

    def plot(self):
        history_df = pd.DataFrame(self.history.history)
        fig = plt.figure(figsize=(15,4), facecolor="#97BACB")
        fig.suptitle("Learning Plot of Model for Loss")
        pl=sns.lineplot(data=history_df["loss"],color="#444160")
        pl.set(ylabel ="Training Loss")
        pl.set(xlabel ="Epochs")
    
    def load_last_weight(self):

        model_list = glob(config['path']['model_dir']+'*.hdf5')
        latest_model = max(model_list, key=os.path.getctime)
        self.model.load_weights(latest_model)

        print(f"Loaded weight {os.path.split(latest_model)[-1]}")

    def generate(self, dataset: Dataset, filename=str(len(os.listdir(config['path']['out_dir']))+1), length=config['output']['length'], bpm=config['output']['bpm']):

        beats = bpm//60*length
        # TODO: Pass the file length variable.
        pick = random.randint(0, 500)
        # NOTE: Use only track 0 for now.
        track = 0
        

        picked_seed = dataset.encoded_data_list[pick]
        picked_seed = picked_seed[track][:config['params']['data_size']]
        print(picked_seed)

        picked_seed = dataset.tokenizers[track].texts_to_sequences([picked_seed])
        picked_seed = to_categorical(picked_seed, num_classes=config['params']['vocab_size'])
        midi_sequence = picked_seed.copy()[0]

        for _ in range(beats):
            next_beat = self.model.predict(picked_seed, verbose=0)
            next_beat = np.argmax(next_beat) 
            next_beat = to_categorical(next_beat, num_classes=config['params']['vocab_size'])
            next_beat = np.array(next_beat).reshape(1, next_beat.shape[0])

            midi_sequence = np.append(midi_sequence, next_beat, axis=0)

            picked_seed = np.append(picked_seed[0], next_beat, axis=0)
            picked_seed = picked_seed[1:, :]
            picked_seed = np.expand_dims(picked_seed, axis=0)
            
        pianoroll_sequence = []
        for index, beat in enumerate(midi_sequence):
            decoded_beat = dataset.decode(beat)
            pianoroll_sequence.append(decoded_beat)
    
        pianoroll_sequence = np.array(pianoroll_sequence)
        pianoroll_sequence = pianoroll_sequence.reshape(pianoroll_sequence.shape[0]*pianoroll_sequence.shape[1], pianoroll_sequence.shape[2])
        pianoroll_sequence = pianoroll_sequence * config['params']['intensity']
    
        std_track = pypianoroll.StandardTrack(pianoroll=pianoroll_sequence)
        output = pypianoroll.Multitrack(resolution=config['params']['resolution'], tracks=[std_track])
        pypianoroll.write(config['path']['out_dir']+filename+'.mid', output)
