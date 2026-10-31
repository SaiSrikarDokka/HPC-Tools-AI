# BERT Baseline for SQuAD Question Answering

This project fine-tunes a pretrained BERT model on the Stanford Question Answering Dataset (SQuAD) using Hugging Face Transformers. The baseline is intended as a reproducible reference workflow for extractive QA and is executed on a single-GPU SLURM node.

## Objective

The goal is to adapt `google-bert/bert-base-uncased` to SQuAD-v1.1 and evaluate its answer extraction quality using standard QA metrics, namely exact match and F1 score.

## Model and dataset

- Base model: `google-bert/bert-base-uncased`
- Dataset: `rajpurkar/squad`
- Task: question answering / extractive span prediction
- Framework: PyTorch + Hugging Face Transformers

## Project structure

```text
Baseline/
├── baseline.slurm          # SLURM job script for GPU execution
├── baseline_*.err          # Slurm error logs
├── baseline_*.out          # Slurm standard output logs
├── run_baseline.sh         # Training wrapper script
├── run_qa.py               # Hugging Face QA training script
├── trainer_qa.py           # Custom QA trainer logic
├── utils_qa.py             # Postprocessing utilities for answer extraction
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── output_baseline_retry/  # Artifacts from the latest baseline run (kept local)
└── __pycache__/            # Local cache files
```

## Training configuration

The latest run (SLURM job `11143433`) used the configuration in `run_baseline.sh`:

- model: `google-bert/bert-base-uncased`
- dataset: `rajpurkar/squad`
- epochs: `2`
- train batch size: `16`
- eval batch size: `16`
- learning rate: `3e-5`
- max sequence length: `384`
- doc stride: `128`
- random seed: `42`
- output directory: `./output_baseline_retry`
- logging each 100 steps
- save strategy: `epoch`
- evaluation strategy: `epoch`

The run used one NVIDIA A100 GPU and completed both training epochs and the final evaluation.

## Run commands

### Install dependencies

```bash
pip install -r requirements.txt
```

### Local run

```bash
bash run_baseline.sh
```

### SLURM job

```bash
sbatch baseline.slurm
```

## Latest measured results

The final evaluation from `baseline_11143433.out` produced:

- Exact match: `80.7569`
- F1 score: `88.142`
- Training loss: `1.0005`
- Training samples: `88,492`
- Validation samples: `10,753`
- Training runtime: `2,883.10` seconds (`48.05` minutes)
- Final evaluation runtime: `55.31` seconds
- SLURM job wall-clock time: `2,975` seconds (`49` minutes, `35` seconds)

The latest run's `all_results.json` contains the combined training and evaluation results:

```json
{
    "epoch": 2.0,
    "eval_exact_match": 80.75685903500474,
    "eval_f1": 88.14200496107321,
    "eval_runtime": 55.3198,
    "eval_samples": 10753,
    "eval_samples_per_second": 194.379,
    "eval_steps_per_second": 12.166,
    "total_flos": 3.468400889562317e+16,
    "train_loss": 1.0005138913107876,
    "train_runtime": 2883.1003,
    "train_samples": 88492,
    "train_samples_per_second": 61.387,
    "train_steps_per_second": 3.837,
    "training_time_minutes": 48.05385364205576,
    "training_time_seconds": 2883.231218523346
}
```

The separate `eval_results.json` and `train_results.json` files contain the corresponding evaluation and training metrics.

## Output artifacts

The latest run's `output_baseline_retry/` directory contains:

- `eval_results.json` — final validation metrics
- `train_results.json` — training summary record
- `all_results.json` — combined final training and evaluation metrics
- `eval_predictions.json` — model predictions on validation data
- `eval_nbest_predictions.json` — top answer candidates for each question
- `checkpoint-*` — saved model checkpoints for each epoch
- `model.safetensors` — the trained model weights
- `optimizer.pt` — optimizer state files for checkpoint recovery
- `config.json`, `tokenizer.json`, `tokenizer_config.json` — metadata for model loading

## Important note about Git

The generated `output_baseline_retry/` folder is large because it contains trained model weights and optimizer state. These files are intentionally kept local and excluded from Git to avoid pushing large binary artifacts to GitHub.

The small JSON summary files (`eval_results.json` and `train_results.json`) are lightweight records that are useful to keep in version control.

## References

- Hugging Face Transformers: https://huggingface.co/transformers/
- SQuAD dataset: https://rajpurkar.github.io/SQuAD-explorer/
- BERT base uncased: https://huggingface.co/google-bert/bert-base-uncased
