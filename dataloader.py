from pickle import EMPTY_DICT
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.utils import to_categorical
import pypianoroll
import matplotlib.pyplot as plt
import numpy as np
import yaml
from glob import glob
from tqdm import tqdm
from midi_processing import *

MAX_PITCH = 128
MIN_PITCH = 0

config = yaml.safe_load(open('config.yaml', 'r'))

def get_key(dict, val):
    # function to return key for any value
    for key, value in dict.items():
         if val == value:
             return key
 
    return False

class Dataset():

    EMPTY_BEAT = f"0-{config['params']['resolution']}"

    def __init__(self, path=config['path']['data_dir']) -> None:
        
        self.midi_file_list = glob(path)
        self.raw_data_list = [] # List of pianoroll matrix
        self.bin_data_list = [] # List of binarized pianoroll matrix
        self.pitch_range_list = [[MAX_PITCH, MIN_PITCH] for _ in range(config['params']['tracks'])] # Pitch range for each track.
        self.encoded_data_list = []
        self.concatenated_encoded_data_list = []
        self.concatenated_encoded_data_list = [[self.EMPTY_BEAT] for _ in range(config['params']['tracks'])]

        self.tokenizers = [] # Tokenizer for each track.
        self.word_indexes = [] # Word dictionary for each track.

        self.train_data_all_tracks = []

        self.file_size = None

    def load(self):

        print("Loading data from disk.")

        self.file_size = len(self.midi_file_list)
        for file in tqdm(self.midi_file_list):
            midi = pypianoroll.read(file, resolution=config['params']['resolution'])
            bin_midi = midi.binarize()
            
            raw_pr = []
            bin_pr = []

            for i in range(config['params']['tracks']):
                raw_pr.append(midi.tracks[i].pianoroll.astype('int'))
                bin_pr.append(bin_midi.tracks[i].pianoroll.astype('int'))

                new_range = pypianoroll.pitch_range_tuple(midi.tracks[i].pianoroll)
                self.pitch_range_list[i] = [min(new_range[0], self.pitch_range_list[0][0]), max(new_range[1], self.pitch_range_list[0][1])]
            
            self.raw_data_list.append(raw_pr)
            self.bin_data_list.append(bin_pr)
    
    def encode(self, data='bin'):

        '''
        Encode the loaded data

        data: Choose 'raw' or 'bin' to take the loaded data in the object.
        '''

        if not self.raw_data_list or not self.bin_data_list:
            print("Please load the data first")
            return 

        if data=='raw':
            data_list = self.raw_data_list
        elif data=='bin':
            data_list = self.bin_data_list
        else:
            print("Please specify 'raw' or 'bin' data.")
            return
        
        print("Encoding data.")

        for midi in tqdm(data_list):
            encoded_midi = []
            encoded_midi_string = []
            for index, track in enumerate(midi):
                encoded_track = [] 
                for i in range(0, len(track)//config['params']['resolution']-1):
                    be = beat_encode(track[config['params']['resolution']*i:config['params']['resolution']*(i+1)])
                    encoded_track.append(be)

                encoded_midi.append(encoded_track)
                self.concatenated_encoded_data_list[index] += encoded_track
                self.concatenated_encoded_data_list[index].append(self.EMPTY_BEAT)

            self.encoded_data_list.append(encoded_midi)
    
    def decode(self, predict, track=0):

        '''
        Decode from the model raw output.

        find max prob position -> find key in dictionary -> decode.       
        '''
        # NOTE: Use only track 0 for now.

        word_val = np.argmax(predict)
        encoded_beat = get_key(self.word_indexes[track], word_val)
        decoded_beat = beat_decode(encoded_beat)

        return decoded_beat
    
    def tokenize(self):

        # for track in self.concatenated_encoded_data_list:
        for i in range(config['params']['tracks']):
            track_string = ",".join(self.concatenated_encoded_data_list[i])

            tokenizer = Tokenizer(num_words=config['params']['vocab_size'], split=',', filters='')

            tokenizer.fit_on_texts([track_string])
            self.tokenizers.append(tokenizer)
            self.word_indexes.append(tokenizer.word_index)

            # NOTE: texts_to_sequences will only take the top 200 words in the sentence.
            #       So it will make every track has a different length.
            train_data = tokenizer.texts_to_sequences([track_string])
            train_data = to_categorical(train_data)
            train_data = np.array(train_data).reshape(train_data.shape[1:])

            self.train_data_all_tracks.append(train_data)

    def get_train_data(self, track=0):

        x_train = []
        y_train = []

        train_data = self.train_data_all_tracks[track]
        for i in range(train_data.shape[0]-config['params']['data_size']):
            x_train.append(train_data[i:i+config['params']['data_size'], :])
            y_train.append(train_data[i+config['params']['data_size'], :])

        return np.array(x_train), np.array(y_train)
    
    def get_range(self):

        # Get the upper and lower bound of pitches.
        # TODO: Split it to each track.
        pitch_range = [MAX_PITCH, MIN_PITCH]
        for file in midi_path_list:
            midi = pypianoroll.read(file, resolution=config['params']['resolution'])

            for track in midi.tracks:
                new_range = pypianoroll.pitch_range_tuple(track.pianoroll)
                # print(new_range)
                pitch_range = [min(new_range[0], pitch_range[0]), max(new_range[1], pitch_range[1])]
    
        print(f"Pitch range = {pitch_range}")
    
    def run(self):

        self.load()
        self.encode()
        self.tokenize()