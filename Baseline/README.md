# BERT Baseline for SQuAD Question Answering

This project contains a Hugging Face Transformers baseline for fine-tuning BERT on the Stanford Question Answering Dataset (SQuAD). The training job is executed on a single-GPU SLURM node and saves model checkpoints and evaluation metrics in the `output/` directory.

## Objective

The goal of this baseline is to adapt a pretrained BERT model for extractive question answering on SQuAD and evaluate its performance using standard QA metrics.

## Model and dataset

- Base model: `google-bert/bert-base-uncased`
- Dataset: `rajpurkar/squad`
- Task: Question answering / span extraction
- Framework: Hugging Face Transformers + PyTorch

## Project structu  re

```text
HPC_Tools/Baseline/
├── baseline.slurm          # SLURM job script for GPU execution
├── run_baseline.sh         # Training command wrapper
├── run_qa.py               # HF question-answering training script
├── trainer_qa.py           # Custom trainer for QA evaluation
├── utils_qa.py             # Postprocessing utilities
├── requirements.txt        # Python dependencies
├── output/                 # Model checkpoints, metrics, and logs
├── baseline_*.err          # SLURM stderr logs
├── baseline_*.out          # SLURM stdout logs
└── README.md               # Project documentation
```

## Training configuration

The baseline uses the following settings:

- Model: `google-bert/bert-base-uncased`
- Dataset: `rajpurkar/squad`
- Training epochs: `2`
- Train batch size: `16`
- Eval batch size: `16`
- Learning rate: `3e-5`
- Max sequence length: `384`
- Document stride: `128`
- Output directory: `./output`
- Logging interval: every `100` steps
- Save strategy: every epoch
- Evaluation strategy: every epoch

## Running the baseline

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Submit the SLURM job

```bash
sbatch baseline.slurm
```

### 3. Or run locally

```bash
bash run_baseline.sh
```

The script executes:

```bash
python run_qa.py \
  --model_name_or_path "google-bert/bert-base-uncased" \
  --dataset_name "rajpurkar/squad" \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir "./output" \
  --logging_steps 100 \
  --save_strategy epoch \
  --eval_strategy epoch
```

## Output artifacts

The results are written under `output/` and include:

- `all_results.json` — summary of final training and evaluation metrics
- `eval_results.json` — evaluation metrics from the final validation pass
- `train_results.json` — training summary statistics
- `eval_predictions.json` — model predictions on evaluation data
- `eval_nbest_predictions.json` — top candidate answers per question
- `checkpoint-*` — saved model checkpoints for each epoch
- `config.json`, `tokenizer.json`, `tokenizer_config.json` — model/tokenizer metadata

## Measured results

The completed run produced the following evaluation metrics:

- Exact match: `80.87`
- F1 score: `88.33`
- Training loss: `0.9934`
- Training samples: `88,492`
- Training runtime: about `48.36` minutes
- Validation samples: `10,753`

The aggregate result file `output/all_results.json` contains:

```json
{
  "epoch": 2.0,
  "eval_exact_match": 80.87038789025544,
  "eval_f1": 88.33304725923448,
  "train_loss": 0.9933756519842484,
  "training_time_minutes": 48.35991259686804
}
```

## Notes

- The job was configured for a single GPU with `--gres=gpu:1`.
- It uses a standard BERT-base uncased checkpoint and fine-tunes it on SQuAD without additional architecture changes.
- This is intended as a baseline model: it provides a simple, reproducible training setup that can be used as a benchmark for later experiments.
- The generated log files (`baseline_*.err` and `baseline_*.out`) capture the training run, environment information, and final result summary.

## References

- Hugging Face Transformers: https://huggingface.co/transformers/
- SQuAD dataset: https://rajpurkar.github.io/SQuAD-explorer/
- BERT base uncased: https://huggingface.co/google-bert/bert-base-uncased
