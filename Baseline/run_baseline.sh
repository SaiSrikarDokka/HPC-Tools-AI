#!/bin/bash
set -e

MODEL="google-bert/bert-base-uncased"
DATASET="rajpurkar/squad"
OUTPUT_DIR="./output"

python run_qa.py \
    --model_name_or_path "$MODEL" \
    --dataset_name "$DATASET" \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir "$OUTPUT_DIR" \
    --logging_steps 100 \
    --save_strategy epoch \
    --eval_strategy epoch