# HPC Tools for AI Assignment

This repository contains an assignment for the HPC Tools for AI course. It compares fine-tuning BERT for question answering on SQuAD v1.1 with a single-GPU baseline and a multi-node, multi-GPU distributed run.

Both experiments use `google-bert/bert-base-uncased`, the `rajpurkar/squad` dataset, and two training epochs. The baseline uses one GPU; the distributed run uses four GPUs across two nodes.

## Repository layout

- [`BASELINE/`](./BASELINE/) — single-GPU training and evaluation, using one NVIDIA A100.
- [`DISTRIBUTED/`](./DISTRIBUTED/) — DistributedDataParallel training and evaluation, using two nodes with two NVIDIA A100 GPUs per node (four GPUs total).

Each directory contains its own run scripts and detailed README with configuration and run instructions.

## Results comparison

Results below are from the completed baseline SLURM job `11143433` and distributed SLURM job `11142338`. Both runs trained for two epochs with an effective global training batch size of 16.

| Metric | Baseline: 1 node, 1 GPU | Distributed: 2 nodes, 4 GPUs |
|---|---:|---:|
| Training runtime | 2,883.10 s (48.05 min) | 1,194.10 s (19.91 min) |
| Training loss | 1.0005 | 0.9994 |
| Validation exact match | 80.7569 | 80.6433 |
| Validation F1 | 88.1420 | 88.2102 |

The distributed run achieved about **2.41x training speedup**, reducing training runtime by **58.6%** compared with the single-GPU baseline. Validation quality was comparable: distributed exact match was lower by about 0.11 points, while F1 was higher by about 0.07 points. These are measurements from individual runs; small metric differences can occur from training variability.

Runtime is the Hugging Face Trainer training-phase runtime, not total SLURM job elapsed time.
