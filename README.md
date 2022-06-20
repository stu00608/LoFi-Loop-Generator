## Environment

- Python 3.8.12

Some description :

    Data :                js-fakes https://github.com/omarperacha/js-fakes
                          It's a midi dataset included 500 4 tracks midi files.

    Representation :      Read file as pianoroll, every beat will represent as an array with shape (resolution, 128).
                          Then make the pianoroll matrix to a time series data. The encoded beat data is our "word". Use
                          Tokenizer to map most frequently used 200 data into a dictionary, and map it as a integer. Finally
                          turn the integer to one_hot array.

```
pitch
  4| 0 0 0 0 0 0
  3| 1 1 0 0 0 0                       Tokenizer
  2| 0 0 1 1 0 0 ... ---> 3-2#2-2#1-2 -----------> int -> one_hot array.
  1| 0 0 0 0 1 1
  0| 0 0 0 0 0 0
              beat
```

## Docker

- Please follow Tensorflow Docker Installation first.
- Modify the params in `entry.sh`.

```
docker build -t lofi . --no-cache
docker run --gpus all -it --rm lofi
```

### Start training

```
./entry.sh
```

### Generate using trained weight.

- Need to give the same datasize, batch_size.
- Modify the config file, disable wandb.
  - TODO: Need a way to detect if it's using generate.py

```
path:
  data_dir: js-fakes/midi/*.mid
  model_dir: models/
  out_dir: outputs/
params:
  data_size: 1
  vocab_size: 100
  batch_size: 64
  epochs: 1000
  intensity: 90
  resolution: 24
  tracks: 4
  bpm: 120
output:
  wandb: False                    <---------- Here
  length: 20
```

- Copy the weight path.

```
WEIGHT=models/model-exp0620-14-2.5837-bigger.hdf5
```

- Then run the script.

```
python generate.py --name exp0620 --weights $WEIGHT --datasize 1 --batch 128
```

- To move the output midi file to your disk, use `docker cp`.

```
docker cp <container_id>:/lofi/<midi_file_name> /path/in/your/disk
```

## Trouble Shooting

```
Traceback (most recent call last):
  File "train.py", line 30, in <module>
    dataset = pickle.load(f)
ValueError: unsupported pickle protocol: 5
```

- Need to do `rm data/loaded_dataset`
