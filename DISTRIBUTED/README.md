# Distributed BERT Training for SQuAD Question Answering

This deliverable fine-tunes `google-bert/bert-base-uncased` on SQuAD v1.1 using PyTorch through Hugging Face Transformers. The `QuestionAnsweringTrainer` inherits from Hugging Face `Trainer`; when launched with `torchrun`, Trainer uses its built-in DistributedDataParallel (DDP) support for process/device setup, data sharding, and gradient synchronization.

## Distributed configuration

- Nodes: 2
- GPUs per node: 2 NVIDIA A100
- Total DDP workers: 4
- Epochs: 2
- Per-device train/eval batch size: 4
- Effective global train batch size: 16 (4 workers x 4 examples)
- Learning rate: `3e-5`
- Maximum sequence length: 384
- Document stride: 128
- Output directory: `./output_ddp`

The run uses a global training batch size of 16 across four GPU workers.

## Files

- `ddp.slurm` requests two nodes with two A100 GPUs per node, establishes rendezvous settings, then starts one `torchrun` agent on each node with `srun`.
- `run_baseline.sh` launches two training workers per node and passes the QA training options to `run_qa.py`.
- `run_qa.py`, `trainer_qa.py`, and `utils_qa.py` load and preprocess SQuAD, train/evaluate the QA model, and post-process predictions.
- `smoke_single_gpu.slurm` runs a short one-GPU check that imports the required Python packages before training; it is not the distributed benchmark.
- `requirements.txt` lists Python dependencies.

The SLURM scripts use the `short` partition. The single-GPU smoke test can be used to verify Python dependencies on an allocated node before launching DDP.

## Run

From this directory, create the SLURM log directory before submitting:

```bash
mkdir -p logs
sbatch ddp.slurm
```

SLURM creates its output files before executing the script, so `logs/` must already exist before submitting either job. Check the DDP job output for `distributed training: True` and confirm all four workers start and the job completes successfully.

To check the environment and dataset/model loading on one A100 before the DDP run, submit the smoke test instead:

```bash
sbatch smoke_single_gpu.slurm
```

It trains on at most 32 samples for one epoch and writes to `output_env_smoke/`. Its runtime is only a sanity check, not the DDP benchmark.

## Timing

Measured DDP results for job `11142338`:

| Metric | Result |
|---|---:|
| Hardware | 2 nodes, 4 NVIDIA A100 GPUs |
| Epochs | 2 |
| Per-GPU batch size | 4 |
| Effective global batch size | 16 |
| Training runtime (`train_runtime`) | 1,194.10 s (19.91 min) |
| Training loss | 0.9994 |
| Exact match | 80.64 |
| F1 | 88.21 |

These are Hugging Face Trainer metrics; `train_runtime` measures the training phase, not total SLURM job elapsed time. The job log contains rendezvous heartbeat/connection warnings during shutdown after the training and evaluation metrics were produced.

## References

- Hugging Face Transformers: https://huggingface.co/transformers/
- SQuAD: https://rajpurkar.github.io/SQuAD-explorer/
- BERT base uncased: https://huggingface.co/google-bert/bert-base-uncased
