#!/usr/bin/env python3
"""
evaluate_all_metrics.py (Mixed-Mode Universal Referee)
=====================================================
Clinical-grade multi-dimensional evaluation script.

Hybrid Logic:
  - df_v0 (Baseline only): Cross C-index, Spearman, ANOVA, T-tests, AUC
  - df_all (All visits): Long C-index, Inversion Rate (InvR)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MMSE_SOURCE = os.path.join(
    SCRIPT_DIR, "..", "datasets", "csv", "5_folds", "dataset_5_folds_DTI.csv"
)

def load_fold(csv_dir, fold, suffix=""):
    csv_path = os.path.join(csv_dir, f"fold_{fold}", f"predictions_fold_{fold}{suffix}.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    
    # Automatic column name compatibility
    if "Diagnosis" in df.columns and "diagnosis" not in df.columns:
        df.rename(columns={"Diagnosis": "diagnosis"}, inplace=True)
    elif "True_Label_Int" in df.columns and "diagnosis" not in df.columns:
        df.rename(columns={"True_Label_Int": "diagnosis"}, inplace=True)
        
    # Convert date format to numeric for t1 < t2 time axis comparison
    df["Acq Date_num"] = pd.to_numeric(df["Acq Date"].astype(str).str.replace('-', ''), errors='coerce')
    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors='coerce')
    
    df = df.dropna(subset=["Acq Date_num", "diagnosis", "Pred_Score"])
    return df

def load_mmse_lookup(csv_path):
    """Load MMSCORE from source CSV for Spearman correlation."""
    df = pd.read_csv(csv_path)
    if "Subject (PTID)" in df.columns:
        df = df.rename(columns={"Subject (PTID)": "Subject"})
    df["Acq Date_num"] = pd.to_numeric(
        df["Acq Date"].astype(str).str.replace('-', ''), errors="coerce")
    df["MMSCORE"] = pd.to_numeric(df["MMSCORE"], errors="coerce")
    return df[["Subject", "Acq Date_num", "MMSCORE"]].dropna(
        subset=["MMSCORE", "Acq Date_num"])

def compute_v0_metrics(df_v0, df_all):
    r = {}
    
    # 1. Spearman (vs MMSE)
    tmp = df_v0.dropna(subset=["MMSE"])
    if len(tmp) >= 3:
        rho, _ = stats.spearmanr(tmp["Pred_Score"], tmp["MMSE"])
        r["Spearman"] = rho if not np.isnan(rho) else np.nan
    else:
        r["Spearman"] = np.nan
    
    # 2. ANOVA & T-tests
    groups = {k: g["Pred_Score"].values for k, g in df_v0.groupby("diagnosis") if len(g) > 1}
    
    if len(groups) >= 3:
        r["ANOVA_F"], _ = stats.f_oneway(groups.get(1, []), groups.get(2, []), groups.get(3, []))
    else:
        r["ANOVA_F"] = np.nan
        
    r["T_1v2"], _ = stats.ttest_ind(groups.get(1, []), groups.get(2, [])) if 1 in groups and 2 in groups else (np.nan, np.nan)
    r["T_2v3"], _ = stats.ttest_ind(groups.get(2, []), groups.get(3, [])) if 2 in groups and 3 in groups else (np.nan, np.nan)
    
    r["T_1v2"] = abs(r["T_1v2"]) if not np.isnan(r["T_1v2"]) else np.nan
    r["T_2v3"] = abs(r["T_2v3"]) if not np.isnan(r["T_2v3"]) else np.nan

    # 3. Cross C-index (Different Subjects, Baseline only)
    S = df_v0["Pred_Score"].values
    L = df_v0["diagnosis"].values
    label_diff = L[:, None] - L[None, :]
    score_diff = S[:, None] - S[None, :]
    
    valid_cross = (label_diff > 0)
    total_cross = np.sum(valid_cross)
    
    if total_cross > 0:
        r["Cross_C"] = (np.sum((score_diff > 0) & valid_cross) + 0.5 * np.sum((score_diff == 0) & valid_cross)) / total_cross
    else:
        r["Cross_C"] = np.nan

    # 4. Conversion AUC
    def get_auc(base_diag, target_condition):
        base_g = df_v0[df_v0["diagnosis"] == base_diag].copy()
        if len(base_g) == 0: return np.nan
        
        converts = []
        for _, row in base_g.iterrows():
            subj = row["Subject"]
            bl_date = row["Acq Date_num"]
            future = df_all[(df_all["Subject"] == subj) & (df_all["Acq Date_num"] > bl_date)]
            if len(future) == 0:
                base_g = base_g.drop(index=row.name)
                continue
            converts.append(1 if target_condition(future["diagnosis"]).any() else 0)
        
        base_g["Convert"] = converts
        if base_g["Convert"].nunique() > 1:
            X = base_g[["Pred_Score"]]
            y = base_g["Convert"]
            clf = LogisticRegression(class_weight="balanced")
            clf.fit(X, y)
            y_prob = clf.predict_proba(X)[:, 1]
            return roc_auc_score(y, y_prob)
        return np.nan

    r["AUC_CN_MCI"] = get_auc(1, lambda d: d >= 2)
    r["AUC_MCI_AD"] = get_auc(2, lambda d: d == 3)
    
    return r

def compute_longitudinal_metrics(df_all):
    S = df_all["Pred_Score"].values
    T = df_all["Acq Date_num"].values
    subj_codes, _ = pd.factorize(df_all["Subject"])
    ID = subj_codes
    
    score_diff = S[:, None] - S[None, :]
    time_diff = T[:, None] - T[None, :]
    subj_same = ID[:, None] == ID[None, :]
    
    valid_long = subj_same & (time_diff > 0)
    total_long = np.sum(valid_long)
    
    if total_long > 0:
        c_long = (np.sum((score_diff > 0) & valid_long) + 0.5 * np.sum((score_diff == 0) & valid_long)) / total_long
        # Severe inversion rate: score decreases over time by more than 1e-4
        inv_r = np.sum((score_diff < -1e-4) & valid_long) / total_long
        return c_long, inv_r
    return np.nan, np.nan

def evaluate_fold(df_all, mmse_lookup=None):
    # Extract each subject's first visit as the V0 baseline
    df_v0 = df_all.sort_values("Acq Date_num").drop_duplicates(subset=["Subject"], keep="first").reset_index(drop=True)

    if mmse_lookup is not None:
        df_v0 = df_v0.merge(
            mmse_lookup, on=["Subject", "Acq Date_num"], how="left"
        ).rename(columns={"MMSCORE": "MMSE"})

    r = {}
    # Group A: compute using only independent V0 baselines
    r.update(compute_v0_metrics(df_v0, df_all))

    # Group B: compute using all visits with temporal information
    r["Long_C"], r["InvR"] = compute_longitudinal_metrics(df_all)

    return r

def _mean_std(values):
    valid = [v for v in values if not pd.isna(v)]
    if not valid: return np.nan, np.nan
    return np.mean(valid), np.std(valid)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", required=True, help="Path to the predictions folder")
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--fold", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--suffix", default="")
    parser.add_argument("--label", default="Model")
    parser.add_argument("--mmse_csv", default=DEFAULT_MMSE_SOURCE,
                        help="CSV with MMSCORE column for Spearman")
    args = parser.parse_args()

    folds = range(1, 6) if args.all_folds else ([args.fold] if args.fold else range(1, 6))

    mmse_lookup = None
    if os.path.exists(args.mmse_csv):
        mmse_lookup = load_mmse_lookup(args.mmse_csv)
        print(f"  MMSE source loaded: {len(mmse_lookup)} rows")

    results = []
    for f in folds:
        df = load_fold(args.csv_dir, f, args.suffix)
        if df is not None:
            results.append(evaluate_fold(df, mmse_lookup))
            
    if not results:
        print("No data loaded. Check your --csv_dir path.")
        return

    keys = ["Spearman", "ANOVA_F", "T_1v2", "T_2v3", "Cross_C", "AUC_CN_MCI", "AUC_MCI_AD", "Long_C", "InvR"]
    
    print(f"\n{'='*90}\n  Hybrid Evaluation Summary (5-Fold Mean ± STD) : {args.label}\n{'='*90}")
    
    row_cells = []
    for k in keys:
        m, s = _mean_std([r[k] for r in results])
        if pd.isna(m):
            row_cells.append("N/A")
        elif k in ["ANOVA_F", "T_1v2", "T_2v3"]:
            row_cells.append(f"{m:.2f}±{s:.2f}")
        else:
            row_cells.append(f"{m:.4f}±{s:.4f}")
            
    # Markdown Table Output
    header = "| Model | Spearman rho | ANOVA F | T(1v2) | T(2v3) | Cross C | AUC(1->2) | AUC(2->3) | Long C | InvR(down) |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    row = f"| **{args.label}** | " + " | ".join(row_cells) + " |"
    
    print(header)
    print(sep)
    print(row)
    print("="*90)

if __name__ == "__main__":
    main()