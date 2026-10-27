"""
Brain Region Volume Correlation Analysis

Computes Pearson & Spearman correlations between predicted DCS
and FreeSurfer-derived brain region volumes (Hippocampus, Entorhinal, etc.).

Usage:
    python predict_analysis/volume_correlation.py \
        --exp_name <experiment_name> --folds 1 2 3
"""

import os
import sys
import argparse

import pandas as pd
import scipy.stats as stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import cfg

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VOLUME_COLS = [
    'Hippocampus volume',
    'Entorhinal',
    'Fusiform',
    'Inferior temporal',
    'Middle temporal',
    'Inferior parietal',
    'Precuneus',
]


def load_predictions(exp_name, folds):
    dfs = []
    for fold in folds:
        path = os.path.join(
            PROJECT_ROOT, 'outputs', exp_name, 'predictions',
            f'fold_{fold}', f'predictions_fold_{fold}.csv'
        )
        df = pd.read_csv(path)
        df['Fold'] = fold
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='train_tr_debug')
    parser.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--csv_path', default=os.path.join(
        PROJECT_ROOT, 'datasets', 'csv', '5_folds', 'dataset_5_folds_DTI.csv'),
        help='CSV with FreeSurfer volume columns merged in')
    parser.add_argument('--output_dir', default=os.path.join(
        PROJECT_ROOT, 'outputs', 'experimental_results'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading predictions...')
    df_pred = load_predictions(args.exp_name, args.folds)
    df_pred['_key'] = df_pred['Subject'].astype(str) + '_' + df_pred['Acq Date'].astype(str)
    print(f'  {len(df_pred)} prediction records across folds {args.folds}')

    print('Loading volume data...')
    df_vol = pd.read_csv(args.csv_path)
    df_vol['_key'] = df_vol['Subject (PTID)'].astype(str) + '_' + df_vol['Acq Date'].astype(str)

    df = df_pred.merge(df_vol[['_key'] + VOLUME_COLS], on='_key', how='inner')
    print(f'  {len(df)} matched records')

    corr_results = []

    print(f'\n{"="*70}')
    print(f'{"Region":<25s}  {"Pearson r":>10s}  {"p-value":>12s}  {"Spearman rho":>12s}  {"p-value":>12s}')
    print('='*70)

    for col in VOLUME_COLS:
        valid = df[['Pred_Score', col]].dropna()
        if len(valid) < 3:
            print(f'{col:<25s}  insufficient data (n={len(valid)})')
            continue

        pr, pp = stats.pearsonr(valid['Pred_Score'], valid[col])
        sr, sp = stats.spearmanr(valid['Pred_Score'], valid[col])

        sig = ''
        if pp < 0.001 or sp < 0.001:
            sig = '***'
        elif pp < 0.01 or sp < 0.01:
            sig = '**'
        elif pp < 0.05 or sp < 0.05:
            sig = '*'

        print(f'{col:<25s}  {pr:>+10.4f}  {pp:>12.4e}  {sr:>+10.4f}  {sp:>12.4e}  {sig}')

        corr_results.append({
            'Region': col,
            'N': len(valid),
            'Pearson_r': pr,
            'Pearson_p': pp,
            'Spearman_rho': sr,
            'Spearman_p': sp,
        })

    print('='*70)

    df_corr = pd.DataFrame(corr_results)
    corr_csv = os.path.join(args.output_dir, 'volume_correlation.csv')
    df_corr.to_csv(corr_csv, index=False)
    print(f'\nCorrelation table saved: {corr_csv}')

    detail_csv = os.path.join(args.output_dir, 'volume_correlation_detail.csv')
    df[['Subject', 'Acq Date', 'Fold', 'Pred_Score', 'Diagnosis'] + VOLUME_COLS].to_csv(
        detail_csv, index=False)
    print(f'Per-visit detail saved: {detail_csv}')


if __name__ == '__main__':
    main()
