"""
ROI Ablation Analysis — ADNI Data (multi-modality, 4-channel FA+MD+RD+AD)

Metric:
  importance(r) = mean_over_visits( |DCS_full - DCS_masked(r)| )
  DCS = posterior mean mu[:, 0] (model.eval -> reparameterize returns mu).
  Each fold uses its own held-out visits and corresponding fold checkpoint.

Statistical test:
  T_obs = mean_{k in AD}(importance_k) - mean_{k not in AD}(importance_k)
  Permutation (10 000x) on ROI labels.
  Run twice: raw importance and per-voxel importance (controls for ROI size).

Usage:
    python predict_analysis/roi_ablation_adni.py \
        --exp_name <experiment_name> --folds 1 2 3
"""

import os
import sys
import argparse

import torch
import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet18 import DiseaseTrajectoryVAE
from datasets.dataset import DTI_Sequence_Dataset, dynamic_visit_collate
from configs.config import cfg

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS_PATH    = os.path.join(PROJECT_ROOT, 'datasets', 'atlas', 'aparc+aseg_2mm.nii')
ROI_NAME_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'atlas', 'FS_ROI_name.txt')

AD_VULNERABLE_NAMES = {
    'Left-Hippocampus', 'Right-Hippocampus',
    'ctx-lh-entorhinal', 'ctx-rh-entorhinal',
    'ctx-lh-fusiform', 'ctx-rh-fusiform',
    'ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal',
    'ctx-lh-middletemporal', 'ctx-rh-middletemporal',
    'ctx-lh-superiortemporal', 'ctx-rh-superiortemporal',
    'ctx-lh-precuneus', 'ctx-rh-precuneus',
    'ctx-lh-posteriorcingulate', 'ctx-rh-posteriorcingulate',
    'ctx-lh-isthmuscingulate', 'ctx-rh-isthmuscingulate',
}


# --------------------------------------------------------------------------- #
# Atlas helpers
# --------------------------------------------------------------------------- #
def load_roi_info():
    roi_info = []
    with open(ROI_NAME_PATH) as f:
        for roi_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts    = line.split(None, 1)
            fs_label = int(parts[0])
            roi_name = parts[1].strip() if len(parts) > 1 else f'label_{fs_label}'
            roi_info.append((roi_idx, fs_label, roi_name))
    return roi_info


def build_roi_masks(roi_info):
    atlas_data = nib.load(ATLAS_PATH).get_fdata().astype(np.int32)
    masks = {}
    voxel_counts = {}
    for roi_idx, fs_label, _ in roi_info:
        m = atlas_data == fs_label
        masks[roi_idx] = torch.from_numpy(m)
        voxel_counts[roi_idx] = int(m.sum())
    return masks, voxel_counts


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(fold, exp_name, device):
    model = DiseaseTrajectoryVAE(
        latent_dim=cfg.LATENT_DIM,
        dropout_p=cfg.DROPOUT,
        in_channels=cfg.IN_CHANNELS,
    ).to(device)

    ckpt_path = os.path.join(
        PROJECT_ROOT, 'outputs', exp_name, 'checkpoints', f'fold_{fold}', 'best_model.pth'
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_sd)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def forward_batched(model, imgs_cpu, device, batch_size=8):
    scores_list = []
    use_amp = torch.cuda.is_available()
    for start in range(0, len(imgs_cpu), batch_size):
        chunk = imgs_cpu[start: start + batch_size].to(device)
        with autocast('cuda', enabled=use_amp):
            scores, _, _, _ = model(chunk)
            scores = scores.view(-1)
        scores_list.append(scores.cpu())
    return torch.cat(scores_list, dim=0)


def collect_tensors(val_loader, val_ds):
    imgs_list, subj_ids_list, dates_all = [], [], []
    for batch in tqdm(val_loader, desc='  Collecting tensors', leave=False):
        if batch['imgs'].size(0) == 0:
            continue
        imgs_list.append(batch['imgs'].cpu())
        subj_ids_list.append(batch['subj_ids'].cpu())
        dates_all.extend(batch['dates'])
    all_imgs     = torch.cat(imgs_list,     dim=0)
    all_subj_ids = torch.cat(subj_ids_list, dim=0)
    all_subjects = [val_ds.subjects[i.item()] for i in all_subj_ids]
    return all_imgs, all_subjects, dates_all


def load_baseline_scores(fold, exp_name, subjects, dates):
    pred_csv = os.path.join(
        PROJECT_ROOT, 'outputs', exp_name, 'predictions',
        f'fold_{fold}', f'predictions_fold_{fold}.csv'
    )
    df = pd.read_csv(pred_csv)
    df['_key'] = df['Subject'].astype(str) + '_' + df['Acq Date'].astype(str)
    score_lookup = dict(zip(df['_key'], df['Pred_Score']))

    baseline, missing = [], 0
    for subj, date in zip(subjects, dates):
        score = score_lookup.get(f'{subj}_{date}', None)
        if score is None:
            missing += 1
            score = 0.0
        baseline.append(score)
    if missing:
        print(f'  WARNING: {missing} visits had no matching baseline — set to 0.0')
    return torch.tensor(baseline, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Per-fold ablation
# --------------------------------------------------------------------------- #
def ablate_fold(fold, exp_name, data_root, csv_path,
                roi_info, roi_masks, device, batch_size=8):
    df_master = pd.read_csv(csv_path)
    val_df    = df_master[df_master['Fold'] == fold].reset_index(drop=True)
    print(f'  Fold {fold}: {len(val_df)} validation records')

    val_ds = DTI_Sequence_Dataset(
        df=val_df, data_root=data_root,
        modalities=cfg.MODALITIES, mode='val',
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False,
        collate_fn=dynamic_visit_collate, num_workers=2,
    )
    model = load_model(fold, exp_name, device)

    print(f'  Fold {fold}: Loading image tensors...')
    all_imgs, all_subjects, all_dates = collect_tensors(val_loader, val_ds)
    print(f'  Fold {fold}: {len(all_imgs)} visits')

    print(f'  Fold {fold}: Loading baseline scores...')
    baseline_scores = load_baseline_scores(fold, exp_name, all_subjects, all_dates)

    results = []
    per_visit_abs = {}

    for roi_idx, fs_label, roi_name in tqdm(roi_info, desc='  ROI ablation'):
        mask = roi_masks[roi_idx]
        orig_vals = all_imgs[:, :, mask].clone()
        all_imgs[:, :, mask] = 0.0
        masked_scores = forward_batched(model, all_imgs, device, batch_size=batch_size)
        all_imgs[:, :, mask] = orig_vals

        delta = baseline_scores - masked_scores
        abs_delta = delta.abs()
        per_visit_abs[roi_idx] = abs_delta

        results.append({
            'fold':             fold,
            'roi_idx':          roi_idx,
            'fs_label':         fs_label,
            'roi_name':         roi_name,
            'mean_abs_drop':    abs_delta.mean().item(),
            'mean_signed_drop': delta.mean().item(),
            'n_visits':         len(delta),
        })

    return pd.DataFrame(results), per_visit_abs


# --------------------------------------------------------------------------- #
# Permutation test (ROI-level)
# --------------------------------------------------------------------------- #
def permutation_test(importance_values, ad_mask, n_perm=10000):
    n_ad = ad_mask.sum()
    ad_mean    = importance_values[ad_mask].mean()
    other_mean = importance_values[~ad_mask].mean()
    observed   = ad_mean - other_mean

    count = 0
    n_total = len(importance_values)
    for _ in range(n_perm):
        perm = np.random.permutation(n_total)
        perm_ad = importance_values[perm[:n_ad]].mean()
        perm_other = importance_values[perm[n_ad:]].mean()
        if (perm_ad - perm_other) >= observed:
            count += 1

    p_value = (count + 1) / (n_perm + 1)
    return float(observed), float(p_value), float(ad_mean), float(other_mean)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',   default='train_tr_debug')
    parser.add_argument('--folds',      nargs='+', type=int, default=None)
    parser.add_argument('--data_root',  default=cfg.DATA_ROOT)
    parser.add_argument('--csv_path',   default=cfg.CSV_PATH)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_perm',     type=int, default=10000)
    args = parser.parse_args()

    device = cfg.DEVICE
    print(f'Device: {device}')

    ckpt_base = os.path.join(PROJECT_ROOT, 'outputs', args.exp_name, 'checkpoints')
    if args.folds is None:
        args.folds = sorted([
            int(d.split('_')[1])
            for d in os.listdir(ckpt_base)
            if d.startswith('fold_') and
               os.path.exists(os.path.join(ckpt_base, d, 'best_model.pth'))
        ])
    print(f'Folds: {args.folds}')

    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, 'outputs', 'experimental_results', 'roi_ablation_adni'
    )
    os.makedirs(out_dir, exist_ok=True)

    print('Loading ROI definitions...')
    roi_info  = load_roi_info()
    roi_masks, voxel_counts = build_roi_masks(roi_info)
    print(f'  {len(roi_info)} ROIs loaded')

    ad_idx_set = {ri for ri, _, name in roi_info if name in AD_VULNERABLE_NAMES}
    print(f'  AD-vulnerable group: {len(ad_idx_set)} ROIs')

    # ---- Per-fold ablation ----
    all_dfs = []
    pooled_abs = {}

    for fold in args.folds:
        print(f'\n{"="*65}\nFold {fold}\n{"="*65}')
        df_fold, pv_abs = ablate_fold(
            fold, args.exp_name, args.data_root, args.csv_path,
            roi_info, roi_masks, device, batch_size=args.batch_size,
        )
        fold_csv = os.path.join(out_dir, f'roi_ablation_fold_{fold}.csv')
        df_fold.to_csv(fold_csv, index=False)
        print(f'  Saved: {fold_csv}')
        all_dfs.append(df_fold)

        for ri, tensor in pv_abs.items():
            pooled_abs.setdefault(ri, []).append(tensor)

    for ri in pooled_abs:
        pooled_abs[ri] = torch.cat(pooled_abs[ri], dim=0)

    # ---- Aggregate: importance = mean(|delta|) over all pooled visits ----
    roi_indices = [ri for ri, _, _ in roi_info]
    roi_names   = {ri: name for ri, _, name in roi_info}
    roi_fs      = {ri: fs for ri, fs, _ in roi_info}

    rows = []
    for ri in roi_indices:
        pooled = pooled_abs[ri]
        rows.append({
            'roi_idx':          ri,
            'fs_label':         roi_fs[ri],
            'roi_name':         roi_names[ri],
            'mean_abs_drop':    pooled.mean().item(),
            'mean_signed_drop': 0.0,
            'n_voxels':         voxel_counts[ri],
            'ad_vulnerable':    roi_names[ri] in AD_VULNERABLE_NAMES,
        })

    df_all = pd.concat(all_dfs, ignore_index=True)
    signed_agg = df_all.groupby('roi_idx')['mean_signed_drop'].mean()
    for row in rows:
        row['mean_signed_drop'] = signed_agg.get(row['roi_idx'], 0.0)

    df_agg = pd.DataFrame(rows).sort_values('mean_abs_drop', ascending=False).reset_index(drop=True)
    df_agg.insert(0, 'rank', range(1, len(df_agg) + 1))

    agg_csv = os.path.join(out_dir, 'roi_ablation_aggregated.csv')
    df_agg.to_csv(agg_csv, index=False)
    print(f'\nAggregated results -> {agg_csv}')

    # ---- Top-10 ----
    print(f'\n{"="*70}')
    print(f'TOP-10 BRAIN REGIONS  (by mean |delta|)')
    print('='*70)
    for _, row in df_agg.head(10).iterrows():
        tag = ' *AD*' if row['ad_vulnerable'] else ''
        sign = '+' if row['mean_signed_drop'] > 0 else '-'
        print(f'  #{row["rank"]:2.0f}  {row["roi_name"]:<45s}  '
              f'|delta|={row["mean_abs_drop"]:.4f}  (sign:{sign}){tag}')

    df_agg.head(10).to_csv(os.path.join(out_dir, 'top10.csv'), index=False)

    # ---- Permutation tests ----
    importance_raw = np.array([pooled_abs[ri].mean().item() for ri in roi_indices])
    importance_pv  = np.array([pooled_abs[ri].mean().item() / max(voxel_counts[ri], 1)
                               for ri in roi_indices])
    ad_mask = np.array([ri in ad_idx_set for ri in roi_indices])

    print(f'\n{"="*70}')
    print(f'PERMUTATION TESTS: AD-vulnerable ({ad_mask.sum()} ROIs) vs '
          f'Remaining ({(~ad_mask).sum()} ROIs)')
    print('='*70)

    diff_r, p_r, ad_r, oth_r = permutation_test(importance_raw, ad_mask, args.n_perm)
    print(f'\n  [Raw importance]')
    print(f'    AD-vulnerable  : {ad_r:.6f}')
    print(f'    Remaining      : {oth_r:.6f}')
    print(f'    Difference     : {diff_r:+.6f}')
    print(f'    p-value        : {p_r:.4f}')

    diff_n, p_n, ad_n, oth_n = permutation_test(importance_pv, ad_mask, args.n_perm)
    print(f'\n  [Per-voxel importance (size-controlled)]')
    print(f'    AD-vulnerable  : {ad_n:.8f}')
    print(f'    Remaining      : {oth_n:.8f}')
    print(f'    Difference     : {diff_n:+.8f}')
    print(f'    p-value        : {p_n:.4f}')

    if p_r < 0.05 or p_n < 0.05:
        print(f'\n  *** SIGNIFICANT: model relies more on AD-vulnerable regions ***')
    print('='*70)

    pd.DataFrame([{
        'n_ad_rois':                int(ad_mask.sum()),
        'n_other_rois':             int((~ad_mask).sum()),
        'ad_importance_raw':        ad_r,
        'other_importance_raw':     oth_r,
        'diff_raw':                 diff_r,
        'p_value_raw':              p_r,
        'ad_importance_per_voxel':  ad_n,
        'other_importance_per_voxel': oth_n,
        'diff_per_voxel':           diff_n,
        'p_value_per_voxel':        p_n,
        'n_perm':                   args.n_perm,
    }]).to_csv(os.path.join(out_dir, 'ad_vulnerable_test.csv'), index=False)

    print(f'\nDone. All results in: {out_dir}')


if __name__ == '__main__':
    main()
