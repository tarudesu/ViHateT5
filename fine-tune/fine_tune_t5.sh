SAVE_MODEL_NAME=$1
PRE_TRAINED_CKPT=$2
OUTPUT_DIR=$5
BATCH_SIZE=$3
NUM_EPOCHS='' # Adjust for your needs
LEARNING_RATE='' # or 2e-5
WANDB_PROJECT="your wandb project here"
WANDB_RUN_NAME="$1"
GPU="$4"

python ./fine_tune_t5.py \
  --save_model_name=$SAVE_MODEL_NAME \
  --pre_trained_ckpt=$PRE_TRAINED_CKPT \
  --output_dir=$OUTPUT_DIR \
  --batch_size=$BATCH_SIZE \
  --num_epochs=$NUM_EPOCHS \
  --learning_rate=$LEARNING_RATE \
  --wandb_project=$WANDB_PROJECT \
  --wandb_run_name=$WANDB_RUN_NAME \
  --gpu=$GPU

# bash fine_tune_t5.sh <MODEL NAME, used for wandb> <PRE-TRAINED CPKT HUGGINGFACE> <BATCH SIZE> <IDs of GPUs> <OUTPUT DIR>

