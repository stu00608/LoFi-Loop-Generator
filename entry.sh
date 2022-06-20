!#/bin/bash

API_KEY=<Input your api key here>
EXP_NAME=exp0620
DATA_SIZE=1
BATCH_SIZE=128
EPOCHS=100

cd /LoFi-Loop-Generator
wandb login $API_KEY
python train.py --name $EXP_NAME --datasize $DATA_SIZE --batch $BATCH_SIZE --epochs $EPOCHS