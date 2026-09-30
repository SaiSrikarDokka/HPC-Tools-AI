# BERT Baseline for SQuAD Question Answering

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
├── output/                 # Local generated artifacts (large files kept local only)
└── __pycache__/            # Local cache files
```

## Training configuration

The current baseline run uses:

- model: `google-bert/bert-base-uncased`
- dataset: `rajpurkar/squad`
- epochs: `2`
- train batch size: `16`
- eval batch size: `16`
- learning rate: `3e-5`
- max sequence length: `384`
- doc stride: `128`
- output directory: `./output`
- logging each 100 steps
- save strategy: `epoch`
- evaluation strategy: `epoch`

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

## Current measured results

The latest rerun produced the following validation metrics:

- Exact match: `81.02`
- F1 score: `88.40`
- Training loss: `0.9955`
- Training samples: `88,492`
- Validation samples: `10,753`
- Training runtime: `48.33` minutes

The aggregate metrics file `output/all_results.json` contains:

```json
{
    "epoch": 2.0,
    "eval_exact_match": 81.02175969725639,
    "eval_f1": 88.39871553578057,
    "eval_runtime": 55.535,
    "eval_samples": 10753,
    "train_loss": 0.9955241624538689,
    "train_runtime": 2899.824,
    "train_samples": 88492,
    "training_time_minutes": 48.332531325798485
}
```

## Output artifacts

The generated local output directory contains:

- `all_results.json` — final aggregate training + evaluation statistics
- `eval_results.json` — final validation metrics
- `train_results.json` — training summary record
- `eval_predictions.json` — model predictions on validation data
- `eval_nbest_predictions.json` — top answer candidates for each question
- `checkpoint-*` — saved model checkpoints for each epoch
- `model.safetensors` — the trained model weights
- `optimizer.pt` — optimizer state files for checkpoint recovery
- `config.json`, `tokenizer.json`, `tokenizer_config.json` — metadata for model loading

## Important note about Git

The generated `output/` folder is large because it contains trained model weights and optimizer state. These files are intentionally kept local and excluded from Git to avoid pushing large binary artifacts to GitHub.

The small JSON summary files (`all_results.json`, `eval_results.json`, `train_results.json`) are the lightweight records that are useful to keep in version control.

## References

- Hugging Face Transformers: https://huggingface.co/transformers/
- SQuAD dataset: https://rajpurkar.github.io/SQuAD-explorer/
- BERT base uncased: https://huggingface.co/google-bert/bert-base-uncased
