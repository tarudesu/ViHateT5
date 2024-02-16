#!/bin/bash

MODEL_NAME=$1 # large or base
GPU=$2 # 0,1,2,3
BATCH_SIZE=$3 #
TRAIN_FILE=$4
VALIDATION_FILE=$5
EPOCHS=$6

python /tmp/pre-train/run_t5_mlm_flax.py \
	--output_dir="your output dir" \
	--config_name="your prepared config" \
	--tokenizer_name="your prepared tokenizer" \
	--max_seq_length="256" \
	--per_device_train_batch_size="$BATCH_SIZE" \
	--per_device_eval_batch_size="$BATCH_SIZE" \
	--adafactor \
	--learning_rate="5e-3" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="5000" \
	--eval_steps="5000" \
  --report_to="wandb" \
  --preprocessing_num_workers=$(nproc) \
  --num_train_epochs="$EPOCHS" \
  --project_name="your project for wandb" \
  --run_name="$MODEL_NAME" \
  --total_gpu="$GPU" \
  --train_file="$TRAIN_FILE" \
  --validation_file="$VALIDATION_FILE" \
  --model_name_or_path='VietAI/vit5-base' # If you pre-train from scratch, remove this line


# bash pre-train.sh <MODEL NAME> <IDs of GPUs> <BATCH SIZE> <TRAIN FILE> <VALIDATION FILE> <EPOCHS>



