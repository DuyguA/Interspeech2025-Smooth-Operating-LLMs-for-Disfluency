TRAIN_BATCH_SIZE=16
VAL_BATCH_SIZE=16
EPOCHS=2
LR=5e-5
TEXT_KEY="plain_text" 
OUTPUT_DIR="exp-plain-text"




python3 -u trainer.py --train_batch_size=$TRAIN_BATCH_SIZE \
                      --val_batch_size=$VAL_BATCH_SIZE \
                      --epochs=$EPOCHS \
                      --lr=$LR \
                      --text_key=$TEXT_KEY \
		      --output_dir=$OUTPUT_DIR


