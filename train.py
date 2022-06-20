from wandb.keras import WandbCallback
import wandb
import argparse
from tensorflow.keras.utils import to_categorical
from dataloader import *
from model import *
import numpy as np
import os
import yaml
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

config = yaml.safe_load(open('config.yaml', 'r'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64, help='Model batch size.')
    parser.add_argument('--epochs', type=int, default=1000, help='Model training epochs.')
    parser.add_argument('--datasize', type=int, default=10, help='Model input data size.')
    parser.add_argument('--name', type=str, default='default', help='Model name.')

    args = parser.parse_args()

    dataset_path = "data/loaded_dataset"
    dataset = Dataset()
    sentences = None
    if os.path.exists(dataset_path):
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset.load()
        with open(dataset_path, "wb") as f:
            pickle.dump(dataset, f)

    sentences = dataset.getConcatenatedTracks()
    beatdict = BeatDictionary(sentences)

    x_train, y_train = dataset.getSingleTrackTrainingData(track=0, data_size=args.datasize)

    x_train = np.array([beatdict.vectorize(beat.tolist()) for beat in x_train])

    y_train = beatdict.beat2index(y_train)
    y_train = to_categorical(y_train, num_classes=beatdict.getDictSize())

    model = LoFiLoopNet(args.name, x_train.shape[1:], beatdict.getDictSize())
    model.summary()

    model.train(x_train, y_train, epochs=args.epochs,
                batch_size=args.batch, data_size=args.datasize)

    # TODO: Make few inferences and upload to wandb.
    sequence = model.generate(dataset, beatdict)
    new_midi = MidiData()
    new_midi.decode(sequence, filename=args.name+'-'+str(args.datasize) +
                    '-'+str(args.batch)+'-'+str(args.epochs)+'.mid')
