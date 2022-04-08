import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.utils import to_categorical
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
import os
from datetime import datetime

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