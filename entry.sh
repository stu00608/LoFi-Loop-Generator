API_KEY=<your_api_key>
EXP_NAME=default-experiment
DATA_SIZE=1
BATCH_SIZE=128
EPOCHS=100

wandb login $API_KEY
python train.py --name $EXP_NAME --datasize $DATA_SIZE --batch $BATCH_SIZE --epochs $EPOCHS