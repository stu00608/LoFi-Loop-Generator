# Run training.

```
git clone -b note https://github.com/stu00608/LoFi-Loop-Generator.git
cd LoFi-Loop-Generator
git submodule init
git submodule update
```

## Docker

- Please follow Tensorflow Docker Installation first.
- Modify the params in `entry.sh`.
- **Remember to put your wandb api key** into `entry.sh`.

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
# For example.
WEIGHT=models/model-exp0620-14-2.5837-bigger.hdf5
```

- Then run the script.

```
python generate.py --name default-output --weights $WEIGHT --datasize 1 --batch 64
```

- To move the output midi file to your disk, open another terminal and use `docker cp`.

```
docker cp <container_id>:/lofi/<midi_file_name> /path/in/your/disk
```
