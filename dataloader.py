import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
import yaml
from glob import glob
from tqdm import tqdm
from typing import List

MAX_PITCH = 128
MIN_PITCH = 0

config = yaml.safe_load(open('config.yaml', 'r'))


def get_key(dict, val):
    # function to return key for any value
    for key, value in dict.items():
        if val == value:
            return key

    return False


class NoteData(pretty_midi.Note):

    beat_duration = 1/(config['params']['bpm']/60)

    def __init__(self, velocity, pitch, start, end, sustain):
        super().__init__(velocity, pitch, start, end)
        self.sustain = 1 if sustain else 0

    def decode(self, data, start, velocity=90):
        data = data.split('-')
        pitch = int(data[0])
        duration = float(data[1])
        end = start+duration

        super().__init__(velocity, pitch, start, end)
        self.sustain = int(data[2])
        return pretty_midi.Note(velocity, pitch, start, end)

    def encode(self):
        return f'{self.pitch}-{self.get_duration()}-{self.sustain}'

    def __repr__(self):
        # return f'Note(start={self.start}, end={self.end}, pitch={self.pitch}, velocity={self.velocity}, sustain={self.sustain})'
        return f'TrainNote(duration={self.get_duration()}, pitch={self.pitch}, sustain={self.sustain}, enc={self.encode()})'


def check_intervals(beat: List[NoteData], start: float, end: float):
    '''
    Pass a list of `MidiData`. Returns a list of note off interval.
    '''

    beat = beat.copy()
    beat.insert(0, NoteData(90, 0, None, start, 1))
    beat.append(NoteData(90, 0, end, None, 1))
    intervals = []
    for i in range(1, len(beat)):
        note_time_interval = beat[i].start-beat[i-1].end
        if(note_time_interval > 0):
            intervals.append([beat[i-1].end, beat[i].start])
        elif(note_time_interval < 0):
            print("Error! Minus note interval, is there any overlapping note exist?")

    return intervals


class MidiData:

    beat_duration = 1/(config['params']['bpm']/60)

    def __init__(self, file=None) -> None:

        self.raw_midi = None

        if file:
            self.load_from_disk(file)

    def load_from_disk(self, file):

        self.raw_midi = pretty_midi.PrettyMIDI(file)
        self.tracks = []
        self.notes = []
        for track in self.raw_midi.instruments:
            self.tracks.append(track)
            self.notes.append(track.notes)
            # print(track.notes)

    def encode(self):

        self.all_encoded_notes = []
        self.all_encoded_beats = []
        self.all_tracks_beats = []
        for track in self.notes:

            midi_start = track[0].start
            midi_end = track[-1].end
            beat_num = int((midi_end-midi_start)/self.beat_duration)

            # print(f"start={midi_start}, end={midi_end}, beat_num={beat_num}")

            encoded_notes = []
            last_pitch = None
            for t in range(beat_num):
                # print(f"Beat {t+1}")
                start = t*self.beat_duration+midi_start
                end = (t+1)*self.beat_duration+midi_start

                # Find the note in the range.
                notes = track
                notes = [note for note in notes if (note.start < end)]
                notes = [note for note in notes if (note.end > start)]

                beat = []

                # Check sustain by looking the start and end time is out of the beat range or not.
                for note in notes:
                    sustain = (note.pitch != last_pitch or note.start < start)

                    note_start = start if note.start <= start else note.start
                    note_end = end if note.end >= end else note.end

                    new_note = NoteData(90, note.pitch, note_start, note_end, sustain)
                    beat.append(new_note)
                    last_pitch = note.pitch

                # Check if there is any empty space.
                # Use a TrainNote object with pitch 0 to represent note off.
                intervals = check_intervals(beat, start, end)
                if intervals:
                    for interval in intervals:
                        note = NoteData(90, 0, interval[0], interval[1], 1)
                        beat.append(note)

                    beat.sort(key=lambda x: x.start)

                encoded_notes.append(beat)

            encoded_beats = []
            for beat in encoded_notes:
                encoded_note_list = []
                for note in beat:
                    encoded_note_list.append(note.encode())
                encoded_note_list = '#'.join(encoded_note_list)
                encoded_beats.append(encoded_note_list)

            self.all_encoded_beats.append(encoded_beats)
            self.all_encoded_notes.append(encoded_notes)

        for beat in range(len(self.all_encoded_beats[0])):
            combined_beat = []
            for i in range(4):
                combined_beat.append(self.all_encoded_beats[i][beat])
            combined_beat = ','.join(combined_beat)
            self.all_tracks_beats.append(combined_beat)

    def decode(self, data=None, filename="output.mid"):

        # TODO: Only store data to self object. And write a output function for outputing self object to midi.

        # TODO: Use ndarray.any()
        # if data == None:
        #     data = self.all_encoded_beats

        self.all_encoded_notes = []
        midi = pretty_midi.PrettyMIDI()
        timestamp = 0
        track_encoded_notes = []
        for index, beat in enumerate(data):

            notes = beat.split('#')
            encoded_notes = []
            for note in notes:
                new_note = NoteData(0, 0, 0, 0, 0)
                new_note.decode(note, timestamp)
                track_encoded_notes.append(new_note)
                timestamp += new_note.duration

            notes = []
            current_note = [-1, -1]
            for element in track_encoded_notes:
                note = [element.pitch, element.sustain]
                if element.pitch == 0:
                    pass
                elif note == current_note and note[1]:
                    notes[-1].end = element.end
                else:
                    notes.append(pretty_midi.Note(element.velocity,
                                                  element.pitch, element.start, element.end))

                current_note = note

        # TODO: only return the decode result.
        _instrument = pretty_midi.Instrument(0, False, f"Track 1")
        _instrument.pitch_bends.append(pretty_midi.PitchBend(0, 0))
        _instrument.notes = notes
        midi.instruments.append(_instrument)

        self.all_encoded_notes.append(track_encoded_notes)

        if self.raw_midi == None:
            self.raw_midi = midi
        midi.write(filename)


class Dataset():

    def __init__(self, path=config['path']['data_dir']) -> None:

        self.file_path_list = glob(path)
        self.encoded_midi_list = []
        self.encoded_data_list = []

    def load(self):

        print("Loading & encoding data from disk.")

        for file in tqdm(self.file_path_list):
            midi = MidiData(file)
            midi.encode()
            self.encoded_midi_list.append(midi)
            self.encoded_data_list.append(np.array(midi.all_encoded_beats))

    def getSingleTrackTrainingData(self, data_size=0, track=0):

        data = [tracks[track] for tracks in self.encoded_data_list]

        x_train = []
        y_train = []
        data_size = config['params']['data_size'] if data_size == 0 else data_size

        for midi in data:
            for i in range(len(midi)-data_size):
                x_train.append(midi[i:i+data_size])
                y_train.append(midi[i+data_size])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        return x_train, y_train

    def getConcatenatedTracks(self):
        '''
        Return tracks concat. For word2vec training.
        '''

        ds = [i.tolist() for i in self.encoded_data_list]
        ds = sum(ds, [])
        return ds

    def concatMidis(self):
        return np.concatenate(self.encoded_data_list, axis=1)


# class Dataset():

#     EMPTY_BEAT = f"0-{config['params']['resolution']}"

#     def __init__(self, path=config['path']['data_dir']) -> None:

#         self.midi_file_list = glob(path)

#     def load(self):

#         print("Loading data from disk.")

#         self.file_size = len(self.midi_file_list)
#         for file in tqdm(self.midi_file_list):


#     def encode(self, data='bin'):
#         '''
#         Encode the loaded data

#         '''

#     def decode(self, predict, track=0):
#         '''
#         Decode from the model raw output.

#         find max prob position -> find key in dictionary -> decode.
#         '''

#     def tokenize(self):

#         # for track in self.concatenated_encoded_data_list:
#         for i in range(config['params']['tracks']):
#             track_string = ",".join(self.concatenated_encoded_data_list[i])

#             tokenizer = Tokenizer(
#                 num_words=config['params']['vocab_size'], split=',', filters='')

#             tokenizer.fit_on_texts([track_string])
#             self.tokenizers.append(tokenizer)
#             self.word_indexes.append(tokenizer.word_index)

#             # NOTE: texts_to_sequences will only take the top 200 words in the sentence.
#             #       So it will make every track has a different length.
#             train_data = tokenizer.texts_to_sequences([track_string])
#             train_data = to_categorical(train_data)
#             train_data = np.array(train_data).reshape(train_data.shape[1:])

#             self.train_data_all_tracks.append(train_data)
#     def get_train_data(self, track=0):

#         x_train = []
#         y_train = []

#         train_data = self.train_data_all_tracks[track]
#         for i in range(train_data.shape[0]-config['params']['data_size']):
#             x_train.append(train_data[i:i+config['params']['data_size'], :])
#             y_train.append(train_data[i+config['params']['data_size'], :])

#         return np.array(x_train), np.array(y_train)

#     def get_range(self):

#         # Get the upper and lower bound of pitches.
#         # TODO: Split it to each track.
#         pitch_range = [MAX_PITCH, MIN_PITCH]
#         for file in midi_path_list:
#             midi = pypianoroll.read(
#                 file, resolution=config['params']['resolution'])

#             for track in midi.tracks:
#                 new_range = pypianoroll.pitch_range_tuple(track.pianoroll)
#                 # print(new_range)
#                 pitch_range = [min(new_range[0], pitch_range[0]), max(
#                     new_range[1], pitch_range[1])]

#         print(f"Pitch range = {pitch_range}")

#     def run(self):

#         self.load()
#         self.encode()
#         self.tokenize()
