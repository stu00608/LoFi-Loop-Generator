import pretty_midi
import os
import pickle
import yaml
import numpy as np
from typing import List
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Embedding, Dropout, concatenate, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

config = yaml.safe_load(open('config.yaml', 'r'))
pathes = config['path']
params = config['params']
outputs = config['output']
params['beat_duration'] = 1/(params['bpm']/60)


class NoteData(pretty_midi.Note):

    def __init__(self, velocity, pitch, start, end, sustain):
        super().__init__(velocity, pitch, start, end)
        self.sustain = 1 if sustain else 0

    @staticmethod
    def decode(data, start, velocity=90):
        '''
        Return (`pretty_midi.Note`, sustain)
        '''
        data = data.split('-')
        pitch = int(data[0])
        duration = float(data[1])
        end = start+duration

        return pretty_midi.Note(velocity, pitch, start, end), int(data[2])

    def encode(self):
        return f'{self.pitch}-{self.get_duration()}-{self.sustain}'

    def __repr__(self):
        # return f'Note(start={self.start}, end={self.end}, pitch={self.pitch}, velocity={self.velocity}, sustain={self.sustain})'
        return f'NoteData(pitch={self.pitch}, duration={self.get_duration()}, sustain={self.sustain}, enc={self.encode()})'


def checkIntervals(beat: List[NoteData], start: float, end: float):
    '''
    Pass a list of encoded beat string. Returns a list of note off intervals.
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


def expandNoteList(notes):

    # First note on time, in second
    midiStartTime = notes[0].start
    # Last note off time, in second
    midiEndTime = notes[-1].end
    # How many beats in this note list. One in 4/4.
    beatNum = int((midiEndTime-midiStartTime)/params['beat_duration'])

    lastPitch = None
    totalNoteDataList = []
    for beatIndex in range(beatNum):
        # Get start and end time in this beat.
        start = midiStartTime + beatIndex*params['beat_duration']
        end = midiStartTime + (beatIndex+1)*params['beat_duration']

        # Find notes that cover this beat interval.
        coveredNotes = [note for note in notes if (note.start < end)]
        coveredNotes = [note for note in coveredNotes if (note.end > start)]

        # encodedBeatList = []
        noteDataList = []
        for note in coveredNotes:
            sustain = (note.pitch != lastPitch or note.start < start)

            # Clip the note start and end time.
            noteStart = start if note.start <= start else note.start
            noteEnd = end if note.end >= end else note.end

            # Add a new NoteData object to the list.
            noteData = NoteData(90, note.pitch, noteStart, noteEnd, sustain)
            noteDataList.append(noteData)
            lastPitch = note.pitch

        # Check if there is any interval between notes, cause it won't count
        # any note off intervals as a Note.
        intervals = checkIntervals(noteDataList, start, end)
        if intervals:
            for interval in intervals:
                # We add a pitch 0 Note object to mark as rest.
                noteData = NoteData(90, 0, interval[0], interval[1], 1)
                noteDataList.append(noteData)

        # Sort the list base on start time, so the list will stay as time series.
        noteDataList.sort(key=lambda x: x.start)

        totalNoteDataList.append(noteDataList)

    return totalNoteDataList


def encodeExpandedNoteList(expandedNoteDataList: List[List[NoteData]]):
    '''
    Takes the expanded note list and encode each NoteData object.

    Returns a dataset ready encoded note representation.
    '''
    encodedBeatList = []
    for expandedNoteData in expandedNoteDataList:
        encodedBeatList.append('#'.join([noteData.encode() for noteData in expandedNoteData]))
    return encodedBeatList


def decodeBeatList(encodedBeatList):
    '''
    Decode beat from encoded music sequence. 

    Returns note list that can use in `pretty_midi.Instrument.notes`.
    '''

    timestamp = 0
    decodedNoteDataList = []
    for beat in encodedBeatList:
        encodedNotes = beat.split('#')
        for encodedNote in encodedNotes:
            decodedNote, sustain = NoteData.decode(encodedNote, timestamp)
            decodedNoteDataList.append((decodedNote, sustain))
            timestamp += decodedNote.get_duration()

    decodedNoteList = []
    lastPitchAndSustain = (-1, -1)
    for decodedNote, sustain in decodedNoteDataList:
        if decodedNote.pitch == 0:
            pass
        elif decodedNote.pitch == lastPitchAndSustain[0] and sustain:
            decodedNoteList[-1].end = decodedNote.end
        else:
            decodedNoteList.append(decodedNote)

        lastPitchAndSustain = (decodedNote.pitch, sustain)

    return decodedNoteList


def generateMidi(filename: str, tracks: List[List[str]], programName: str = 'Acoustic Grand Piano'):
    '''
    tracks `List[List[str]]`:
        Encoded beat sequence. [['70-0.5-1', '68-0.25-1#70-0.25-1'..], ...]

    Write the midi file from encoded tracks.

    '''
    midi = pretty_midi.PrettyMIDI()
    for index, track in enumerate(tracks):
        _instrument = pretty_midi.Instrument(
            pretty_midi.instrument_name_to_program(programName), False, f"Track {index+1}")
        _instrument.pitch_bends.append(pretty_midi.PitchBend(0, 0))
        _instrument.notes = decodeBeatList(track)
        midi.instruments.append(_instrument)

    midi.write(filename)


def loadMidiInFolder(path, pkl):
    '''
    path:
        path for glob to grab midi. Must include last file name like `*.mid`.

    Return tracks read from midi, Note has been encoded to string.
    '''
    wordsList = []
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            wordsList = pickle.load(f)
    else:
        for mid in tqdm(glob(path, recursive=True)):
            try:
                midiData = pretty_midi.PrettyMIDI(mid)
            except:
                print("Cannot load this midi")
                continue

            tracks = [instrument.notes for instrument in midiData.instruments]
            encodedTracks = [encodeExpandedNoteList(expandNoteList(track)) for track in tracks]
            wordsList.append(encodedTracks)
        with open(pkl, 'wb') as f:
            pickle.dump(wordsList, f)

    return wordsList


def concatMidi(data):
    '''
    data:
        trainDataRaw or testDataRaw, will concatenate each track.

    Returns np.array (4, total_length)
    '''

    concatData = None
    for mid in data:
        if concatData == None:
            concatData = mid
        else:
            for i in range(len(concatData)):
                concatData[i] += mid[i]

    return np.array(concatData)


def __split_input_target(chunk):
    '''
    Split x and y for training, in our case, x is the current word, y is the next word.
    '''
    input_text = chunk[:-1]
    target_text = tf.one_hot(chunk[1:], params['current_vocab_size'])
    return input_text, target_text


def makeDataset(data, track):
    '''
    data:
        1-d list.
    track:
        specify the track number to get correct vocab_size.

    Make a tf dataset from a 1-d array.

    It will batch for `sequence_length` to split each sentence. 
    Then map it to x, y. Finally batch to the batch size we set.

    Returns a BatchDataset.
    '''
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(params['sequence_length']+1, drop_remainder=True)
    params['current_vocab_size'] = params['vocab_size'][track]
    dataset = dataset.map(__split_input_target)
    del params['current_vocab_size']
    dataset = dataset.shuffle(10000).batch(params['batch_size'], drop_remainder=True)
    return dataset


def makeSingleTrackModel(track, batch_size=params['batch_size']):
    '''
    Create `Model` object for single track training.
    '''

    model = Sequential()
    model.add(Embedding(params['vocab_size'][track], output_dim=params['embed_size'],
                        batch_input_shape=(batch_size, None)))
    for _ in range(params['layers']):
        model.add(LSTM(params['unit'], return_sequences=True, stateful=params['stateful'],
                       dropout=params['dropout'], recurrent_dropout=params['dropout']))

    model.add(Dense(params['unit'], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['vocab_size'][track], activation='softmax'))
    model.compile(loss=CategoricalCrossentropy(from_logits=False),
                  optimizer=Adam(learning_rate=0.0001), metrics=['acc'])

    return model


def makeLSTMLayer(name, track, batch_size=params['batch_size']):
    '''
    Make input embedding, LSTM layers of multi track model.
    '''

    inputLayer = Input(batch_input_shape=([batch_size, None]))
    embedLayer = Embedding(params['vocab_size'][track],
                           params['embed_size'], name=name+'-embed')(inputLayer)
    lstmLayer = LSTM(params['unit'], return_sequences=True, stateful=params['stateful'],
                     dropout=params['dropout'], recurrent_dropout=params['dropout'], name=name+'-LSTM1')(embedLayer)
    lstmLayer = LSTM(params['unit'], return_sequences=True, stateful=params['stateful'],
                     dropout=params['dropout'], recurrent_dropout=params['dropout'], name=name+'-LSTM2')(lstmLayer)

    return inputLayer, lstmLayer


def makeOutputLayer(name, layer, track):
    '''
    Make output combined linear dense layers of multi track model.
    '''

    outputLayer = layer
    # outputLayer = Dense(params['unit'], activation='relu', name=name+'-linear2')(layer)
    # outputLayer = BatchNormalization()(outputLayer)
    # outputLayer = Dropout(params['dropout'])(outputLayer)
    outputLayer = Dense(params['vocab_size'][track], activation='softmax',
                        name=name+'-output')(outputLayer)

    return outputLayer


def makeMultiTrackModel(batch_size=params['batch_size']):
    '''
    Create `Model` object for multi track training. 
    '''

    inputLayer1, lstmLayer1 = makeLSTMLayer('input1', 0, batch_size)
    inputLayer2, lstmLayer2 = makeLSTMLayer('input2', 1, batch_size)
    inputLayer3, lstmLayer3 = makeLSTMLayer('input3', 2, batch_size)
    inputLayer4, lstmLayer4 = makeLSTMLayer('input4', 3, batch_size)

    concatLayers = concatenate([lstmLayer1, lstmLayer2, lstmLayer3, lstmLayer4])

    x = LSTM(params['unit'], return_sequences=True, name='concatLSTMLayer1')(concatLayers)
    # x = Dropout(params['dropout'])(x)
    # x = LSTM(params['unit'], return_sequences=True, name='concatLSTMLayer2')(x)
    x = Dense(params['unit'], activation='relu', name='concatLinear1')(x)
    x = BatchNormalization()(x)
    x = Dropout(params['dropout'])(x)

    outputLayers = [makeOutputLayer('output'+str(i), x, i) for i in range(params['track_size'])]

    model = Model(inputs=[inputLayer1, inputLayer2, inputLayer3, inputLayer4], outputs=outputLayers)

    model.compile(loss=CategoricalCrossentropy(from_logits=False),
                  optimizer=Adam(learning_rate=0.0001), metrics=['acc'])

    return model
