from re import I
from wandb.keras import WandbCallback
import wandb
from dataloader import *
import os
from glob import glob
import random
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from gensim.models import Word2Vec
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = yaml.safe_load(open('config.yaml', 'r'))

if config['output']['wandb']:
    wandb.init(config=config['params'],
               project='midi-lstm-exp-single-word2vec')


class BeatDictionary():

    def __init__(self, data) -> None:
        self.data = data
        self.build()

    def build(self):
        model = Word2Vec(self.data, min_count=1, window=4,
                         vector_size=config['params']['vocab_size'], workers=4)
        self.word_dict = model.wv
        self.vector_dict = list(model.wv.index_to_key)
        self.dict_size = len(self.vector_dict)

    def getDictSize(self):
        return self.dict_size

    def vectorize(self, beats):
        if isinstance(beats, (list, np.ndarray)):
            result = []
            for beat in beats:
                result.append(self.word_dict[beat])
            return np.array(result)
        else:
            return self.word_dict[beats]

    def beat2index(self, beats):
        if isinstance(beats, (list, np.ndarray)):
            result = []
            for beat in beats:
                result.append(self.vector_dict.index(beat))
            return np.array(result)
        else:
            return self.vector_dict.index(beats)

    def index2beat(self, beats):

        if isinstance(beats, (list, np.ndarray)):
            result = []
            for beat in beats:
                result.append(self.vector_dict[beat])
            return np.array(result)
        else:
            return self.vector_dict[beats]

    def find_beat(self, vectors):
        # TODO: Now is only for inferencing model predict. Should be more general.

        return self.word_dict.similar_by_vector(vectors, topn=1)[0]

    def get_related_chords(self, token, topn=3):
        print("Similar chords with " + token)
        for word, similarity in self.word_dict.most_similar(positive=[token], topn=topn):
            print(word, round(similarity, 3))


class LoFiLoopNet():

    def __init__(self, name, vocab_size) -> None:
        self.name = name
        self.build(vocab_size)

    def build(self, vocab_size):

        dropout = 0.2

        model = Sequential()
        model.add(Embedding(vocab_size, output_dim=256,
                  batch_input_shape=(config['params']['batch_size'], None)))
        for _ in range(4):
            model.add(LSTM(512, return_sequences=True, stateful=False,
                           dropout=dropout, recurrent_dropout=dropout))
            model.add(BatchNormalization())

        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(vocab_size, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(
            learning_rate=0.00001), metrics=['acc'])
        self.model = model

        self.filepath = config['path']['model_dir'] + \
            "model-"+self.name+"-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        model_checkpoint_callback = ModelCheckpoint(
            filepath=self.filepath,
            monitor='loss',
            mode='min',
            save_weights_only=True,
            save_best_only=True)

        self.callbacks = [model_checkpoint_callback]

    def summary(self):
        self.model.summary()

    def train(self, train_dataset, test_dataset, verbose=1, epochs=config['params']['epochs'], batch_size=config['params']['batch_size'], data_size=config['params']['data_size']):

        config['params']['data_size'] = data_size
        config['params']['batch_size'] = batch_size
        config['params']['epochs'] = epochs

        if config['output']['wandb']:

            wandb_callback = WandbCallback(
                log_weights=True, log_evaluation=True, validation_steps=5)
            self.callbacks.append(wandb_callback)
        else:
            print("❌ Wandb Disabled in config.")

        self.history = self.model.fit(
            train_dataset, validation_data=test_dataset, epochs=epochs, verbose=verbose, callbacks=self.callbacks)

        if config['output']['wandb']:
            wandb.save(os.path.join(config['path']['model_dir'], self.filepath))
            wandb.finish()

    def load_weights(self, path):

        self.model.load_weights(path)
        print(f"Loaded weight {path}")

    def predict(self, beatdict: BeatDictionary, beat):
        '''
        beat in, beat out. pred for visualizing.
        '''
        beat_vector = np.array([[beatdict.word_dict[beat]]])
        pred = self.model.predict(beat_vector)
        pred_beat = beatdict.index2beat(np.argmax(pred))

        return pred, pred_beat

    def generate(self, dataset: Dataset, beatdict: BeatDictionary, filename=str(len(os.listdir(config['path']['out_dir']))+1), length=config['output']['length'], bpm=config['params']['bpm'], track=0, data_size=config['params']['data_size'], topn=1):

        beats = bpm//60*length
        # NOTE: Use only track 0 for now.
        pick = random.randint(0, 500)

        picked_seed = dataset.encoded_midi_list[pick]
        picked_seed = picked_seed.all_encoded_beats[track][:data_size]
        midi_sequence = picked_seed.copy()
        picked_seed = beatdict.vectorize(picked_seed)
        picked_seed = np.expand_dims(picked_seed, axis=0)
        # print(picked_seed.shape)

        for _ in range(beats):
            next_beat = self.model.predict([picked_seed], verbose=0)
            # NOTE: Apply topn prediction randomly choose 1 as next beat.
            next_beat = np.argsort(next_beat, axis=1)[:, -topn:][0]
            next_beat = np.flip(next_beat, 0)
            # print(next_beat)
            next_beat = random.sample(list(next_beat), 1)[0]
            next_beat = beatdict.index2beat(next_beat)

            midi_sequence = np.append(midi_sequence, [next_beat], axis=0)
            # print(midi_sequence)

            picked_seed = np.append(picked_seed[0], [beatdict.word_dict[next_beat]], axis=0)
            picked_seed = picked_seed[1:, :]
            picked_seed = np.expand_dims(picked_seed, axis=0)

            # print(picked_seed.shape)

        print(midi_sequence)

        return midi_sequence
