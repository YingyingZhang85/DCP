# config.py
import os
import torch

class Config:
    # Read the FOLD environment variable from the submission script; defaults to 1
    CURRENT_FOLD = int(os.environ.get('FOLD', 1))

    # ================= Experiment Isolation =================
    # Priority: explicit EXP_NAME > SLURM_JOB_ID > local debug fallback
    _slurm_job_id = os.environ.get('SLURM_JOB_ID')
    _default_exp = f'train_tr_{_slurm_job_id}' if _slurm_job_id else 'train_tr_debug'
    EXP_NAME = os.environ.get('EXP_NAME', _default_exp)

    # ================= Path Configuration =================
    PROJECT_ROOT = os.environ.get('DCP_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_ROOT = os.environ.get('DATA_ROOT', os.path.join(PROJECT_ROOT, 'data', 'dti_residuals'))
    CSV_PATH = os.environ.get('CSV_PATH', os.path.join(PROJECT_ROOT, 'datasets', 'csv', '5_folds', 'dataset_5_folds_DTI.csv'))
    BASE_OUTPUT_DIR = os.path.join(os.environ.get('OUTPUT_ROOT', os.path.join(PROJECT_ROOT, 'outputs')), EXP_NAME)

    SAVE_DIR = os.path.join(BASE_OUTPUT_DIR, 'predictions')
    CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', os.path.join(BASE_OUTPUT_DIR, 'checkpoints'))
    BEST_MODEL_PATH = os.environ.get('BEST_MODEL_PATH', None)
    OPTUNA_DB_PATH = f"sqlite:///{BASE_OUTPUT_DIR}/optuna_study.db"
    LOG_DIR = os.path.join(BASE_OUTPUT_DIR, 'logs')

    # ================= Model & Hyperparameters =================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LATENT_DIM = 128
    MODALITIES = ['FA', 'MD', 'RD', 'AD']
    IN_CHANNELS = len(MODALITIES)
    BATCH_SIZE = 96
    NUM_WORKERS = 4

    LR = 0.00011529569553696
    WEIGHT_DECAY = 0.05
    DROPOUT = 0.6
    EPOCHS = 80
    WARMUP_EPOCHS = 5

    # ================= Loss Weights =================
    ANCHOR_SPACING = 2.73933916900962
    MARGIN = 1.62525751894736
    RANK_WEIGHT = 1.30032947859896
    MONO_WEIGHT = 0.641609948934167
    RECON_WEIGHT = 0.1
    KL_WEIGHT = 0.0198728770174581
    CROSS_C_RATIO = 0.6    # Cross-sectional C-index weight
    LONG_C_RATIO = 0.4     # Longitudinal C-index weight
    AUC_12_RATIO = 0.5     # AUC(1->2) weight
    AUC_23_RATIO = 0.5     # AUC(2->3) weight

# Global singleton for convenient access
cfg = Config()
