import os
import torch
import torch.optim as optim
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import time
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import optuna
import copy

from configs.config import cfg
from models.resnet18 import DiseaseTrajectoryVAE
from datasets.dataset import DTI_Sequence_Dataset, DynamicImageSampler, dynamic_visit_collate
from losses.loss import DiseaseProgressionLoss

import random

def set_seed(seed=72):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_all_metrics(scores, labels, times, subj_ids):
    """
    Compute training-time evaluation metrics.
    Returns: Cross C, Long C, ANOVA F, T-tests, Conversion AUC
    Note: Spearman (vs MMSE) is computed separately in evaluate_all_metrics.py
    """
    S = scores.cpu().numpy()
    L = labels.cpu().numpy()
    T = times.cpu().numpy()
    ID = subj_ids.cpu().numpy()

    df = pd.DataFrame({
        'Subject': ID,
        'Time': T,
        'Label': L,
        'Score': S
    })

    # ==========================================
    # 1. Longitudinal C-index
    # ==========================================
    score_diff = S[:, None] - S[None, :]
    time_diff = T[:, None] - T[None, :]
    subj_same = ID[:, None] == ID[None, :]

    valid_long = subj_same & (time_diff > 0)
    total_long = np.sum(valid_long)
    long_c = (np.sum((score_diff > 0) & valid_long) + 0.5 * np.sum((score_diff == 0) & valid_long)) / total_long if total_long > 0 else np.nan

    # ==========================================
    # 2. Extract baseline V0 for cross-sectional metrics
    # ==========================================
    df_v0 = df.sort_values(['Subject', 'Time']).drop_duplicates(subset=['Subject'], keep='first')

    S_v0 = df_v0['Score'].values
    L_v0 = df_v0['Label'].values

    # Cross-sectional C-index
    label_diff_v0 = L_v0[:, None] - L_v0[None, :]
    score_diff_v0 = S_v0[:, None] - S_v0[None, :]
    valid_cross = (label_diff_v0 > 0)
    total_cross = np.sum(valid_cross)
    cross_c = (np.sum((score_diff_v0 > 0) & valid_cross) + 0.5 * np.sum((score_diff_v0 == 0) & valid_cross)) / total_cross if total_cross > 0 else np.nan

    # ANOVA F-value & T-tests
    groups = {k: g['Score'].values for k, g in df_v0.groupby('Label')}
    if len(groups) >= 3:
        anova_f, _ = stats.f_oneway(groups.get(1, []), groups.get(2, []), groups.get(3, []))
    else:
        anova_f = np.nan

    t1v2, _ = stats.ttest_ind(groups.get(1, []), groups.get(2, [])) if 1 in groups and 2 in groups else (np.nan, np.nan)
    t2v3, _ = stats.ttest_ind(groups.get(2, []), groups.get(3, [])) if 2 in groups and 3 in groups else (np.nan, np.nan)

    # ==========================================
    # 3. Conversion AUC
    # ==========================================
    def get_auc(base_label, target_condition):
        base_g = df_v0[df_v0['Label'] == base_label].copy()
        if len(base_g) == 0:
            return np.nan
        y_true = []
        for _, row in base_g.iterrows():
            subj = row['Subject']
            bl_time = row['Time']
            future = df[(df['Subject'] == subj) & (df['Time'] > bl_time)]
            y_true.append(1 if target_condition(future['Label']).any() else 0)
        y_true = np.array(y_true)
        if len(set(y_true)) > 1:
            X = base_g['Score'].values.reshape(-1, 1)
            clf = LogisticRegression(class_weight='balanced', max_iter=1000)
            clf.fit(X, y_true)
            proba = clf.predict_proba(X)[:, 1]
            return roc_auc_score(y_true, proba)
        return np.nan

    auc_12 = get_auc(1, lambda d: d >= 2)
    auc_23 = get_auc(2, lambda d: d == 3)

    return {
        'cross_c': cross_c if not np.isnan(cross_c) else 0.5,
        'long_c': long_c if not np.isnan(long_c) else 0.5,
        'anova': anova_f if not np.isnan(anova_f) else 0.0,
        't1v2': np.abs(t1v2) if not np.isnan(t1v2) else 0.0,
        't2v3': np.abs(t2v3) if not np.isnan(t2v3) else 0.0,
        'auc_12': auc_12 if not np.isnan(auc_12) else 0.5,
        'auc_23': auc_23 if not np.isnan(auc_23) else 0.5
    }


def run_training(trial=None):
    """
    Main training loop.

    Args:
        trial (optuna.Trial, optional): Optuna trial object for hyperparameter optimization.

    Returns:
        dict: Best metrics dictionary.
    """
    set_seed(72)

    device = cfg.DEVICE

    # Dynamic path management
    base_dir = os.path.join(cfg.CHECKPOINT_DIR, f"fold_{cfg.CURRENT_FOLD}")
    if trial is not None:
        save_dir = os.path.join(base_dir, f"trial_{trial.number}")
    else:
        save_dir = base_dir

    os.makedirs(save_dir, exist_ok=True)

    fold = cfg.CURRENT_FOLD
    print(f"\n" + "="*50)
    print(f"Device: {device} | Fold {fold}")
    print("="*50)

    data_root = cfg.DATA_ROOT
    csv_path = cfg.CSV_PATH
    print(f"Reading master dataset from: {csv_path}")

    df_master = pd.read_csv(csv_path)

    train_df = df_master[df_master['Fold'] != fold].reset_index(drop=True)
    val_df = df_master[df_master['Fold'] == fold].reset_index(drop=True)

    # Load datasets
    print(f"Loading training set for Fold {fold} (Size: {len(train_df)})...")
    train_ds = DTI_Sequence_Dataset(df=train_df, data_root=data_root, modalities=cfg.MODALITIES, mode='train')
    train_sampler = DynamicImageSampler(train_ds, max_images=cfg.BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=dynamic_visit_collate, num_workers=cfg.NUM_WORKERS)

    print(f"Loading validation set for Fold {fold} (Size: {len(val_df)})...")
    val_ds = DTI_Sequence_Dataset(df=val_df, data_root=data_root, modalities=cfg.MODALITIES, mode='val')
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=dynamic_visit_collate, num_workers=cfg.NUM_WORKERS)

    # Initialize model
    model = DiseaseTrajectoryVAE(
        latent_dim=cfg.LATENT_DIM,
        dropout_p=cfg.DROPOUT,
        in_channels=cfg.IN_CHANNELS
        ).to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs, enabling DataParallel acceleration!")
        model = nn.DataParallel(model)

    # Initialize loss function
    criterion = DiseaseProgressionLoss(
        margin=cfg.MARGIN,
        rank_weight=cfg.RANK_WEIGHT,
        mono_weight=cfg.MONO_WEIGHT,
        recon_weight=cfg.RECON_WEIGHT,
        kl_weight=cfg.KL_WEIGHT,
        anchor_spacing=cfg.ANCHOR_SPACING
    ).to(device)

    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # Cosine annealing learning rate scheduler (decays to 1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)

    # Mixed precision training
    scaler = GradScaler('cuda')

    # Checkpoint resume logic
    checkpoint_path = os.path.join(save_dir, 'latest_model.pth')
    log_path = os.path.join(save_dir, 'training_log.csv')
    start_epoch = 1
    history = []
    best_metrics = {
        'comp': 0.0, 'cross_c': 0.0, 'long_c': 0.0, 'auc_12': 0.0, 'auc_23': 0.0,
        'anova': 0.0, 't1v2': 0.0, 't2v3': 0.0, 'epoch': 0
    }

    if os.path.exists(checkpoint_path) and os.path.exists(log_path):
        print(f"Checkpoint detected for Fold {fold}, loading weights: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        existing_log = pd.read_csv(log_path)
        if not existing_log.empty:
            start_epoch = int(existing_log['Epoch'].max()) + 1
            history = existing_log.values.tolist()
            if 'Comp_Score' in existing_log.columns:
                best_idx = existing_log['Comp_Score'].idxmax()
                best_metrics.update({
                    'comp': existing_log.loc[best_idx, 'Comp_Score'],
                    'cross_c': existing_log.loc[best_idx, 'Cross_C'],
                    'long_c': existing_log.loc[best_idx, 'Long_C'],
                    'auc_12': existing_log.loc[best_idx, 'AUC_12'] if 'AUC_12' in existing_log.columns else 0.5,
                    'auc_23': existing_log.loc[best_idx, 'AUC_23'] if 'AUC_23' in existing_log.columns else 0.5,
                    'anova': existing_log.loc[best_idx, 'ANOVA_F'] if 'ANOVA_F' in existing_log.columns else 0.0,
                    'epoch': existing_log.loc[best_idx, 'Epoch']
                })

            for _ in range(1, start_epoch):
                scheduler.step()

            print(f"Resuming from epoch {start_epoch}, LR adjusted to: {optimizer.param_groups[0]['lr']:.2e}")
    else:
        print(f"First training run for Fold {fold}, starting from epoch 1.")

    epochs = cfg.EPOCHS
    warmup_epochs = cfg.WARMUP_EPOCHS

    if start_epoch > epochs:
        print(f"Fold {fold} has already completed all {epochs} epochs!")
        return best_metrics

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()

        # Curriculum learning: monotonicity loss warmup
        if epoch <= warmup_epochs:
            criterion.mono_weight = 0.0
            print(f"[Epoch {epoch}] Stage 1: Monotonicity loss disabled (mono_weight=0.0)")
        else:
            criterion.mono_weight = cfg.MONO_WEIGHT
            if epoch == warmup_epochs + 1:
                print(f"[Epoch {epoch}] Stage 2: Monotonicity loss activated")

        # Training phase
        model.train()
        train_loss_sum = 0
        train_steps = 0
        train_rank_sum = 0
        train_mono_sum = 0
        train_recon_sum = 0
        train_kl_sum = 0


        pbar = tqdm(train_loader, desc=f"Fold {fold} - Epoch {epoch}/{epochs} [Train]", leave=False)
        for batch in pbar:
            imgs = batch['imgs'].to(device)     # (N, C, 91, 109, 91)

            # DataParallel guard: ensure batch size is compatible with GPU count
            N, D = imgs.size(0), torch.cuda.device_count()
            if D > 1:
                chunk_size = (N + D - 1) // D
                if N <= D or N % chunk_size == 1:
                    continue
            elif N <= 1:
                continue

            train_steps += 1
            times = batch['times'].to(device)
            labels = batch['labels'].to(device)
            subj_ids = batch['subj_ids'].to(device)

            optimizer.zero_grad()

            with autocast('cuda', enabled=True):
                scores, recon_imgs, mu, logvar = model(imgs)
                scores = scores.squeeze(-1)  # (N,)
                loss_dict = criterion(scores, labels, times, subj_ids, mu, logvar, imgs, recon_imgs)
                loss = loss_dict['total_loss']

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_rank_sum += loss_dict['loss_rank'].item()
            train_mono_sum += loss_dict['loss_mono'].item()
            train_recon_sum += loss_dict['loss_recon'].item()
            train_kl_sum += loss_dict['loss_kl'].item()
            pbar.set_postfix({
                'total': f"{loss.item():.4f}",
                'rank': f"{loss_dict['loss_rank'].item():.4f}",
                'mono': f"{loss_dict['loss_mono'].item():.4f}",
                'recon': f"{loss_dict['loss_recon'].item():.4f}",
                'kl': f"{loss_dict['loss_kl'].item():.4f}"
            })

        avg_train_loss = train_loss_sum / train_steps if train_steps > 0 else 0.0
        avg_train_rank = train_rank_sum / train_steps if train_steps > 0 else 0.0
        avg_train_mono = train_mono_sum / train_steps if train_steps > 0 else 0.0
        avg_train_recon = train_recon_sum / train_steps if train_steps > 0 else 0.0
        avg_train_kl = train_kl_sum / train_steps if train_steps > 0 else 0.0

        # Validation phase
        model.eval()
        val_loss_sum = 0
        val_rank_sum = 0
        val_mono_sum = 0
        val_recon_sum = 0
        val_kl_sum = 0

        all_val_scores = []
        all_val_labels = []
        all_val_times = []
        all_val_subj_ids = []

        val_pbar = tqdm(val_loader, desc=f"Fold {fold} - Epoch {epoch}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                imgs = batch['imgs'].to(device)

                N = imgs.size(0)
                D = torch.cuda.device_count()
                if D > 1:
                    chunk_size = (N + D - 1) // D
                    if N <= D or N % chunk_size == 1:
                        continue
                elif N <= 1:
                    continue

                times = batch['times'].to(device)
                labels = batch['labels'].to(device)
                subj_ids = batch['subj_ids'].to(device)

                scores, recon_imgs, mu, logvar = model(imgs)
                scores = scores.squeeze(-1)
                loss_dict = criterion(scores, labels, times, subj_ids, mu, logvar, imgs, recon_imgs)
                loss = loss_dict['total_loss']

                val_loss_sum += loss.item()
                val_rank_sum += loss_dict['loss_rank'].item()
                val_mono_sum += loss_dict['loss_mono'].item()
                val_recon_sum += loss_dict['loss_recon'].item()
                val_kl_sum += loss_dict['loss_kl'].item()

                val_pbar.set_postfix({
                    'total': f"{loss.item():.4f}",
                    'rank': f"{loss_dict['loss_rank'].item():.4f}",
                    'mono': f"{loss_dict['loss_mono'].item():.4f}",
                    'recon': f"{loss_dict['loss_recon'].item():.4f}",
                    'kl': f"{loss_dict['loss_kl'].item():.4f}"
                })

                all_val_scores.append(scores)
                all_val_labels.append(labels)
                all_val_times.append(times)
                all_val_subj_ids.append(subj_ids)

        actual_val_batches = len(all_val_scores)
        avg_val_loss = val_loss_sum / actual_val_batches if actual_val_batches > 0 else 0.0
        avg_val_rank = val_rank_sum / actual_val_batches if actual_val_batches > 0 else 0.0
        avg_val_mono = val_mono_sum / actual_val_batches if actual_val_batches > 0 else 0.0
        avg_val_recon = val_recon_sum / actual_val_batches if actual_val_batches > 0 else 0.0
        avg_val_kl = val_kl_sum / actual_val_batches if actual_val_batches > 0 else 0.0

        metrics = compute_all_metrics(
            torch.cat(all_val_scores),
            torch.cat(all_val_labels),
            torch.cat(all_val_times),
            torch.cat(all_val_subj_ids)
        )

        composite_score = cfg.CROSS_C_RATIO * metrics['cross_c'] + cfg.LONG_C_RATIO * metrics['long_c'] + cfg.AUC_12_RATIO * metrics['auc_12'] + cfg.AUC_23_RATIO * metrics['auc_23']

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time


        print(f"Ep {epoch} | T_Loss: {avg_train_loss:.4f} | V_Loss: {avg_val_loss:.4f} | Cross: {metrics['cross_c']:.4f} | Long: {metrics['long_c']:.4f} | AUC(12): {metrics['auc_12']:.4f} | AUC(23): {metrics['auc_23']:.4f} | Comp: {composite_score:.4f} | Time: {epoch_time:.1f}s")

        history.append([epoch, avg_train_loss, avg_val_loss, avg_train_rank, avg_train_mono, avg_train_recon, avg_train_kl, avg_val_rank, avg_val_mono, avg_val_recon, avg_val_kl, metrics['cross_c'], metrics['long_c'], metrics['auc_12'], metrics['auc_23'], metrics['anova'], composite_score])

        # Optuna pruning
        if trial is not None:
            trial.report(composite_score, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Save best model
        if composite_score > best_metrics['comp']:
            best_metrics.update({
                'comp': composite_score,
                'cross_c': metrics['cross_c'],
                'long_c': metrics['long_c'],
                'auc_12': metrics['auc_12'],
                'auc_23': metrics['auc_23'],
                'anova': metrics['anova'],
                't1v2': metrics['t1v2'],
                't2v3': metrics['t2v3'],
                'epoch': epoch
            })
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

            print(f"--> [Fold {fold}] Best Model Saved at Epoch {epoch} (Comp: {composite_score:.4f} | Cross: {metrics['cross_c']:.4f}, Long: {metrics['long_c']:.4f})")

        # Save latest model for checkpoint resume
        torch.save(model.state_dict(), os.path.join(save_dir, 'latest_model.pth'))

        pd.DataFrame(history, columns=['Epoch', 'Train_Loss', 'Val_Loss', 'Train_Rank', 'Train_Mono', 'Train_Recon', 'Train_KL', 'Val_Rank', 'Val_Mono', 'Val_Recon', 'Val_KL', 'Cross_C', 'Long_C', 'AUC_12', 'AUC_23', 'ANOVA_F', 'Comp_Score']).to_csv(os.path.join(save_dir, 'training_log.csv'), index=False)


        scheduler.step()

    # =======================================================
    # End-of-trial scoring report and summary CSV
    # =======================================================
    print(f"\n" + "="*80)
    print(f"[Trial {trial.number if trial else 'N/A'} Summary] Best Epoch: {best_metrics['epoch']}")
    print(f"Hyperparams  : LR={cfg.LR:.2e}, Rank={cfg.RANK_WEIGHT:.2f}, Mono={cfg.MONO_WEIGHT:.2f}, Margin={cfg.MARGIN:.2f}, KL={cfg.KL_WEIGHT:.4f}, Anchor={cfg.ANCHOR_SPACING:.2f}")
    print(f"Global Metrics: ANOVA F: {best_metrics['anova']:.2f} | T(1v2): {best_metrics['t1v2']:.2f} | T(2v3): {best_metrics['t2v3']:.2f}")
    print(f"AUC & C-Index : Cross C: {best_metrics['cross_c']:.4f} | Long C: {best_metrics['long_c']:.4f} | AUC(1->2): {best_metrics['auc_12']:.4f} | AUC(2->3): {best_metrics['auc_23']:.4f}")
    print(f"Optuna Target : Final Composite Score: {best_metrics['comp']:.4f}")
    print("="*80 + "\n")

    summary_path = os.path.join(cfg.BASE_OUTPUT_DIR, "optuna_trials_summary.csv")
    row_dict = {
        "Trial": trial.number if trial else -1,
        "LR": cfg.LR,
        "Rank_W": cfg.RANK_WEIGHT,
        "Mono_W": cfg.MONO_WEIGHT,
        "Margin": cfg.MARGIN,
        "KL_W": cfg.KL_WEIGHT,
        "Anchor": cfg.ANCHOR_SPACING,
        "Best_Epoch": best_metrics['epoch'],
        "Comp_Score": best_metrics['comp'],
        "Cross_C": best_metrics['cross_c'],
        "Long_C": best_metrics['long_c'],
        "AUC_1_2": best_metrics['auc_12'],
        "AUC_2_3": best_metrics['auc_23'],
        "ANOVA_F": best_metrics['anova'],
        "T_1v2": best_metrics['t1v2'],
        "T_2v3": best_metrics['t2v3']
    }
    df_summary = pd.DataFrame([row_dict])
    if os.path.exists(summary_path):
        df_summary.to_csv(summary_path, mode='a', header=False, index=False)
    else:
        df_summary.to_csv(summary_path, index=False)

    return best_metrics


def run_experiment(trial=None):
    fold = cfg.CURRENT_FOLD
    print("\n" + "="*50)
    print(f"   STARTING EXPERIMENT: RUNNING FOLD {fold}")
    print("="*50)
    best_metrics = run_training(trial=trial)
    return best_metrics['comp']


if __name__ == "__main__":
    run_experiment()
