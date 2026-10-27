#!/bin/bash

#SBATCH --job-name=roi_ablation
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH -N 1
### Partition and node — update these for your cluster
#SBATCH -p YOUR_PARTITION
#SBATCH -w YOUR_NODE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00

# Project root — update this to your clone location
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"

EXP_NAME="${EXP_NAME:-train_tr_debug}"
LOG_DIR="${PROJECT_ROOT}/outputs/experimental_results/roi_ablation_adni"
mkdir -p "$LOG_DIR"

exec > "$LOG_DIR/ablation_${SLURM_JOB_ID}.out" 2>"$LOG_DIR/ablation_${SLURM_JOB_ID}.err"

echo "Job started on node: $(hostname) | Time: $(date)"

# Activate your Conda environment — update the env name if needed
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dcp

cd "$PROJECT_ROOT"

python predict_analysis/roi_ablation_adni.py \
    --exp_name "$EXP_NAME" \
    --folds 1 2 3 \
    --batch_size 8

echo "Job completed at: $(date)"
