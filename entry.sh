API_KEY=<your_api_key>
EXP_NAME=default-experiment
DATA_SIZE=1
BATCH_SIZE=64
EPOCHS=1000

wandb login $API_KEY
python train.py --name $EXP_NAME --datasize $DATA_SIZE --batch $BATCH_SIZE --epochs $EPOCHS