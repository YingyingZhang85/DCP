#!/bin/bash

# ================= Resource Request =================
### Job name (visible in squeue)
#SBATCH --job-name=regression

### Logs are redirected by exec below; discard SLURM's own output
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

### Request 1 node
#SBATCH -N 1

### Partition and node — update these for your cluster
#SBATCH -p YOUR_PARTITION
#SBATCH -w YOUR_NODE

### No GPU needed for regression preprocessing
#SBATCH --gres=gpu:0

### CPU and memory — update for your cluster
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

### Maximum wall time
#SBATCH --time=11:59:00


# ================= Log Setup =================
# Project root — update this to your clone location
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"

LOG_DIR="${PROJECT_ROOT}/outputs/regression_logs"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/regression_${SLURM_JOB_ID}.out" 2>&1


# ================= Environment & Execution =================
echo "Job started on node: $(hostname)"
echo "Time: $(date)"

# 1. Activate your Conda environment — update the env name if needed
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dcp

# 2. Enter the project directory
cd "$PROJECT_ROOT"

# Force multi-threaded matrix operations
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16

echo "Running normative residualization (regressing out age, sex, education)..."
python -u datasets/regression/nilearn_regression.py

echo "Task completed at: $(date)"
