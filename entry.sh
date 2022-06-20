API_KEY=0bb49583df6f40334eacd0032d1ae863761d4dbb
EXP_NAME=exp0620
DATA_SIZE=1
BATCH_SIZE=128
EPOCHS=100

wandb login $API_KEY
python train.py --name $EXP_NAME --datasize $DATA_SIZE --batch $BATCH_SIZE --epochs $EPOCHS