#!/bin/bash
set -e

MODEL="google-bert/bert-base-uncased"
DATASET="rajpurkar/squad"
OUTPUT_DIR="./output_ddp"

torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node=2 \
    --rdzv-backend=c10d \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    run_qa.py \
    --model_name_or_path "$MODEL" \
    --dataset_name "$DATASET" \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir "$OUTPUT_DIR" \
    --logging_steps 100 \
    --save_strategy epoch \
    --eval_strategy epoch