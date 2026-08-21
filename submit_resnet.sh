#!/bin/bash

# ================= Resource Request =================
### Job name (visible in squeue)
#SBATCH --job-name=DCP_train

### Logs are redirected by exec below; discard SLURM's own output
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

### Request 1 node
#SBATCH -N 1

### Partition and node — update these for your cluster
#SBATCH -p YOUR_PARTITION
#SBATCH -w YOUR_NODE

### Request 2 GPU
#SBATCH --gres=gpu:2

### Request 16 CPU cores for DTI data loading
#SBATCH --cpus-per-task=16

### Maximum wall time
#SBATCH --time=36:00:00


# ================= Experiment Directory Setup =================
EXP_NAME=${EXP_NAME:-"train_job_${SLURM_JOB_ID}"}
export EXP_NAME

# Project root — update this to your clone location
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"

LOG_ARCHIVE_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}/logs"
mkdir -p "$LOG_ARCHIVE_DIR"

# Redirect all subsequent output to experiment-specific logs
exec > "$LOG_ARCHIVE_DIR/train_log_${SLURM_JOB_ID}.out" 2>"$LOG_ARCHIVE_DIR/train_log_${SLURM_JOB_ID}.err"


# ================= Environment & Execution =================
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo "Log directory: $LOG_ARCHIVE_DIR"

# 1. Activate your Conda environment — update the env name if needed
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dcp

# 2. Enter the project directory
cd "$PROJECT_ROOT"

echo "Starting cross-validation pipeline (Job ID: $SLURM_JOB_ID)"

# # Optionally specify a pre-trained best model path (overrides fold-level checkpoint lookup)
# export BEST_MODEL_PATH="/path/to/your/best_model.pth"

# Run folds sequentially (adjust range as needed, e.g., {1..5} for 5-fold CV)
for CURRENT_FOLD in {1..5}; do
# for CURRENT_FOLD in 1; do

    export FOLD=$CURRENT_FOLD

    echo ""
    echo "=========================================================="
    echo "Starting end-to-end pipeline for Fold $FOLD"
    echo "=========================================================="

    # 1. Train model
    echo ">>> [$FOLD] Step 1: Training model..."
    python train_engine.py

    # 2. Generate predictions
    echo ">>> [$FOLD] Step 2: Generating predictions..."
    python predict_analysis/predict_model.py

    echo "Fold $FOLD completed."

done

echo "Running multi-dimensional evaluation across all folds..."

python predict_analysis/evaluate_all_metrics.py \
    --csv_dir "outputs/${EXP_NAME}/predictions" \
    --all_folds \
    --label "DCP (Ours)"


echo "=========================================================="
echo "All folds completed."
echo "Finished at: $(date)"
echo "=========================================================="

# echo "Running Optuna hyperparameter search..."
# python run_optuna.py

echo "Pipeline execution complete."
echo "Finished at: $(date)"
