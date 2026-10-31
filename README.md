# HPC Tools for AI Assignment

This snapshot is the **BASELINE** deliverable for the HPC Tools for AI course. It fine-tunes `google-bert/bert-base-uncased` on SQuAD v1.1 for extractive question answering.

## Repository layout

- [`BASELINE/`](./BASELINE/) — single-GPU training and evaluation on one NVIDIA A100. Its README documents the scripts, configuration, and measured results.

The completed baseline run (SLURM job `11143433`) trained for two epochs with a batch size of 16. Training took **2,883.10 seconds (48.05 minutes)**; validation exact match was **80.7569** and F1 was **88.1420**. Runtime is the Hugging Face Trainer training-phase runtime, not total SLURM job elapsed time.
