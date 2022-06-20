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

```
docker build --build-arg $MY_WANDB_API -t lofi . --no-cache
```

- To move the output midi file to your disk, use `docker cp`.

```
docker cp <container_id>:/LoFi-Loop-Generator/default10-64-1000.mid /path/in/your/disk
```
