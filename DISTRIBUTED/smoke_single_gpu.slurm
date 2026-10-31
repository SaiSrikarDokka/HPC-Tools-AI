#!/bin/bash

#SBATCH --job-name=bert_squad_baseline
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=00:30:00

echo "======================================"
echo "BERT BASELINE - SQuAD"
echo "======================================"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

echo ""
echo "GPU information:"
nvidia-smi

echo ""
echo "Python:"
python --version

echo ""
echo "PyTorch / CUDA:"
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
"
