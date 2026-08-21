import os
import copy
import torch
import optuna
from optuna.pruners import MedianPruner

# Import from training engine
from train_engine import run_experiment
from configs.config import cfg


def objective(trial):
    """Optuna optimization objective function

    Args:
        trial (optuna.Trial): Optuna trial object

    Returns:
        float: Best composite score
    """
    # Extract current config as baseline
    trial_config = {
        'warmup_epochs': cfg.WARMUP_EPOCHS,
        'dropout': cfg.DROPOUT,
        'anchor_spacing': cfg.ANCHOR_SPACING,
        'recon_weight': cfg.RECON_WEIGHT,
        'weight_decay': cfg.WEIGHT_DECAY,
        'lr': cfg.LR,
        'rank_weight': cfg.RANK_WEIGHT,
        'mono_weight': cfg.MONO_WEIGHT,
        'margin': cfg.MARGIN,
        'kl_weight': cfg.KL_WEIGHT,
    }
    
    # Current fold
    current_fold = cfg.CURRENT_FOLD
    
    # ==========================================
    # Core search space (keeping only the 3 most critical variables for fast search)
    # ==========================================
    # 1. Learning rate (log-scale search, extended lower bound for fine convergence)
    trial_config['lr'] = trial.suggest_float('lr', 1e-5, 2e-4, log=True)
    
    # 2. Ranking and monotonicity weights (relative constraint vs absolute time constraint)
    trial_config['rank_weight'] = trial.suggest_float('rank_weight', 1.0, 1.5)
    trial_config['mono_weight'] = trial.suggest_float('mono_weight', 0.5, 1.0)
    
    # 3. Macro spatial scale
    # anchor_spacing: controls the physical distance between CN, MCI, AD anchor points
    trial_config['anchor_spacing'] = trial.suggest_float('anchor_spacing', 1.5, 3.0)
    
    # 4. Micro repulsion force (Margin must be smaller than Anchor Spacing, leaving a buffer manifold)
    # Instead of searching independently, let margin be a ratio of spacing (e.g., 30%-70% of spacing)
    margin_ratio = trial.suggest_float('margin_ratio', 0.3, 0.7)
    trial_config['margin'] = trial_config['anchor_spacing'] * margin_ratio
    
    # 5. Gaussian aggregation prior (prevent collapse boundary)
    # 0.001 ensures high-fidelity image reconstruction, 0.05 ensures distribution stability
    trial_config['kl_weight'] = trial.suggest_float('kl_weight', 0.001, 0.05, log=True)
    
    # Apply trial_config values to global cfg
    cfg.RANK_WEIGHT = trial_config['rank_weight']
    cfg.MONO_WEIGHT = trial_config['mono_weight']
    cfg.MARGIN = trial_config['margin']
    cfg.KL_WEIGHT = trial_config['kl_weight']
    cfg.ANCHOR_SPACING = trial_config['anchor_spacing']

    # Call training engine
    try:
        best_score = run_experiment(trial=trial)
    except optuna.exceptions.TrialPruned:
        raise
    
    return best_score


if __name__ == "__main__":
    # 1. Ensure the parent directory of the database is created
    db_path = cfg.OPTUNA_DB_PATH.replace('sqlite:///', '')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    study_name = f"{cfg.EXP_NAME}_fold_{cfg.CURRENT_FOLD}"

    study = optuna.create_study(
        study_name=study_name,
        storage=cfg.OPTUNA_DB_PATH,
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=20),
        direction="maximize"
    )

    print(f"Starting optimization: {study_name}")
    print(f"Storage: {cfg.OPTUNA_DB_PATH}")

    n_trials = 50
    timeout = 24 * 60 * 60

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print("Optimization complete!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_trial.value}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances

        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(cfg.BASE_OUTPUT_DIR, f"optuna_history_fold_{cfg.CURRENT_FOLD}.png"))

        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(cfg.BASE_OUTPUT_DIR, f"optuna_importance_fold_{cfg.CURRENT_FOLD}.png"))

        print("Visualization results saved.")
    except:
        print("Could not generate visualizations. Make sure plotly is installed.")