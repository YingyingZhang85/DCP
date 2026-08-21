import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
import random

from models.resnet18 import DiseaseTrajectoryVAE
from datasets.dataset import DTI_Sequence_Dataset, DynamicImageSampler, dynamic_visit_collate

from configs.config import cfg


def set_seed(seed=72):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_c_index(scores, labels, times, subj_ids):
    """Core function to calculate Concordance Index (C-index)"""
    S, L, T, ID = scores.numpy(), labels.numpy(), times.numpy(), subj_ids.numpy()

    # Pairwise differences
    label_diff = L[:, None] - L[None, :]
    score_diff = S[:, None] - S[None, :]
    time_diff = T[:, None] - T[None, :]
    subj_diff = ID[:, None] != ID[None, :]
    subj_same = ~subj_diff

    # Cross-sectional C-index (V0 baseline only, one record per subject)
    v0_mask = np.zeros(len(ID), dtype=bool)
    for sid in np.unique(ID):
        idx = np.where(ID == sid)[0]
        v0_mask[idx[np.argmin(T[idx])]] = True
    S_v0, L_v0 = S[v0_mask], L[v0_mask]
    ld_v0 = L_v0[:, None] - L_v0[None, :]
    sd_v0 = S_v0[:, None] - S_v0[None, :]
    valid_cross = (ld_v0 > 0)
    total_cross = np.sum(valid_cross)
    c_cross = (np.sum((sd_v0 > 0) & valid_cross) + 0.5 * np.sum((sd_v0 == 0) & valid_cross)) / total_cross if total_cross > 0 else 0.0

    # Longitudinal C-index (Comparing visits within the same subject)
    valid_long = subj_same & (time_diff > 0)
    total_long = np.sum(valid_long)
    c_long = (np.sum((score_diff > 0) & valid_long) + 0.5 * np.sum((score_diff == 0) & valid_long)) / total_long if total_long > 0 else 0.0

    return c_cross, c_long

def run_inference(fold):
    set_seed(72)

    fold_save_dir = os.path.join(cfg.SAVE_DIR, f'fold_{fold}')
    os.makedirs(fold_save_dir, exist_ok=True)

    print(f"\n" + "="*50)
    print(f"Starting inference for Fold {fold} ...")
    print("="*50)

    # 1. Read the master CSV
    csv_path = cfg.CSV_PATH
    if not os.path.exists(csv_path):
         raise FileNotFoundError(f"Master CSV not found: {csv_path}")
    
    df_master = pd.read_csv(csv_path)
    
    # Dynamic split: extract only the validation data for the current fold
    val_df = df_master[df_master['Fold'] == fold].reset_index(drop=True)

    # 1. Load Data
    print(f"Loading validation set for Fold {fold} (Size: {len(val_df)})...")
    val_ds = DTI_Sequence_Dataset(df=val_df, data_root=cfg.DATA_ROOT, modalities=cfg.MODALITIES, mode='val')
    # Use the native DataLoader without any oversampler
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=dynamic_visit_collate, num_workers=cfg.NUM_WORKERS)

    # 2. Initialize Model
    model = DiseaseTrajectoryVAE(latent_dim=cfg.LATENT_DIM, dropout_p=cfg.DROPOUT, in_channels=cfg.IN_CHANNELS).to(cfg.DEVICE)

    checkpoint_path = cfg.BEST_MODEL_PATH or os.path.join(cfg.CHECKPOINT_DIR, f'fold_{fold}', 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Model weights not found for Fold {fold} -> {checkpoint_path}")
        return

    print(f"Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)    
    
    model.eval()

    # 3. Inference Loop
    all_scores, all_labels, all_times, all_subj_ids, all_dates = [], [], [], [], []
    all_sigmas = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Predicting Fold {fold}"):
            imgs = batch['imgs'].to(cfg.DEVICE)
            
            # Note: During inference, never skip/discard data, otherwise the exported CSV will be missing subjects!
            if imgs.size(0) == 0: 
                continue

            # Predict scores (S), ignore reconstructed images
            with autocast('cuda', enabled=True):
                scores, recon_imgs, mu, logvar = model(imgs)
                scores = scores.squeeze(-1)
                sigmas = torch.exp(0.5 * logvar[:, 0])

            all_scores.append(scores.cpu())
            all_sigmas.append(sigmas.cpu())
            all_labels.append(batch['labels'].cpu())
            all_times.append(batch['times'].cpu())
            all_subj_ids.append(batch['subj_ids'].cpu())

            all_dates.extend(batch['dates'])

    # Concatenate results and compute metrics
    scores_tensor = torch.cat(all_scores, dim=0)
    sigmas_tensor = torch.cat(all_sigmas, dim=0)  
    labels_tensor = torch.cat(all_labels, dim=0)
    times_tensor = torch.cat(all_times, dim=0)
    subj_tensor = torch.cat(all_subj_ids, dim=0)

    cross_c, long_c = compute_c_index(scores_tensor, labels_tensor, times_tensor, subj_tensor)
    print(f"\nTest Results (Fold {fold}):")
    print(f"Cross-sectional C-index: {cross_c:.4f}")
    print(f"Longitudinal C-index:    {long_c:.4f}")

    # 4. Export results to CSV
    # Map internal numeric subject IDs back to original string identifiers
    real_subjects = [val_ds.subjects[i.item()] for i in subj_tensor]
    
    label_map_inverse = {1: 'CN', 2: 'MCI', 3: 'AD'}
    
    real_labels_str = [label_map_inverse[l.item()] for l in labels_tensor]

    df_out = pd.DataFrame({
        'Subject': real_subjects,
        'Acq Date': all_dates,
        'Time_Point': times_tensor.numpy(),
        'True_Label': real_labels_str,
        'Diagnosis': labels_tensor.numpy(),
        'Pred_Score': scores_tensor.numpy(),
        'Pred_Sigma': sigmas_tensor.numpy()
    })

    # Sort results by subject and chronological order
    df_out = df_out.sort_values(by=['Subject', 'Time_Point']).reset_index(drop=True)
    
    out_csv = os.path.join(fold_save_dir, f'predictions_fold_{fold}.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"Detailed predictions saved to: {out_csv}")


if __name__ == "__main__":
    # ================= Prediction Mode Selection =================

    # Mode A: Regular experiment - just finished a single fold specified by environment variable, check detailed results
    predict_folds = [cfg.CURRENT_FOLD]
    
    # Mode B: Final validation after all 5 folds - generate all 5 prediction CSVs at once
    # predict_folds = [1, 2, 3, 4, 5]
    
    for f in predict_folds:
        run_inference(f)
