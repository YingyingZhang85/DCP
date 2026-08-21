import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# ---- Font settings ----
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False

SAVE_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', 'merged_folds', 'new_plots', 'ROC', 'conversion_roc.pdf')
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# ====== Each method: prediction file path template (with {f}) + score column + color ======
METHODS = {
    'DCP':          (os.path.join(PROJECT_ROOT, 'outputs', 'final_aldm', 'predictions', 'fold_{f}', 'predictions_fold_{f}.csv'), 'Pred_Score', '#e74c3c', [1, 2, 3]),

    # Update these paths to point to baseline method results
    'SuStaIn':       (os.path.join(PROJECT_ROOT, '..', 'ALDM_baseline', 'SuStaIn', 'results', 'fold_{f}', 'predictions_fold_{f}.csv'),   'Pred_Score', '#2ecc71', [1, 2, 3, 4, 5]),

    'DLMRI':   (os.path.join(PROJECT_ROOT, '..', 'ALDM_baseline', 'Deep-learning-MRI-AD-prediction', 'results_retrained', 'fold_{f}', 'predictions_fold_{f}_regressed.csv'),    'Pred_Score', '#3498db', [1, 2, 3, 4, 5]),

    'LSSL': (os.path.join(PROJECT_ROOT, '..', 'ALDM_baseline', 'Ouyang_2022', 'predict_analysis', 'results', 'fold_{f}', 'predictions_fold_{f}.csv'),  'Pred_Score', '#9b59b6', [1, 2, 3, 4, 5]),

    'MedicalNet':     (os.path.join(PROJECT_ROOT, '..', 'ALDM_baseline', 'MultitaskCognition', 'results_retrained', 'fold_{f}', 'predictions_fold_{f}.csv'),      'Pred_Score', '#f39c12', [1, 2, 3, 4, 5]),
}

# ---- Build conversion task (y_true, score) from one fold prediction ----
def label_from(base_diag, fmax):
    if base_diag == 1:
        if fmax == 1:   return 'CN', 0
        if fmax >= 2:   return 'CN', 1
    elif base_diag == 2:
        if fmax == 2:   return 'MCI', 0
        if fmax == 3:   return 'MCI', 1
    return None, None

# ====== Build conversion labels from ground truth data (used for single-row files) ======
META_CSV = os.path.join(PROJECT_ROOT, 'datasets', 'csv', '5_folds', 'dataset_5_folds_DTI.csv')

def build_conv_lookup(meta_csv):
    meta = pd.read_csv(meta_csv)
    if 'Subject (PTID)' in meta.columns:
        meta = meta.rename(columns={'Subject (PTID)': 'Subject'})
    if 'Diagnosis' not in meta.columns and 'DIAGNOSIS' in meta.columns:
        meta = meta.rename(columns={'DIAGNOSIS': 'Diagnosis'})
    tcol = 'Acq Date' if 'Acq Date' in meta.columns else 'Time_Point'
    meta['_t'] = pd.to_numeric(meta[tcol].astype(str).str.replace('-', ''), errors='coerce')
    meta = meta.sort_values(['Subject', '_t'])
    lookup = {}
    for subj, g in meta.groupby('Subject'):
        if len(g) < 2:
            continue
        cohort, label = label_from(g.iloc[0]['Diagnosis'], g.iloc[1:]['Diagnosis'].max())
        if cohort is not None:
            lookup[subj] = (cohort, label)
    return lookup

CONV_LOOKUP = build_conv_lookup(META_CSV)
print(f"Conversion label table: {len(CONV_LOOKUP)} subjects")

def build_conversion(df, score_col):
    df = df.copy()
    if 'Diagnosis' not in df.columns and 'diagnosis' in df.columns:
        df = df.rename(columns={'diagnosis': 'Diagnosis'})
    tcol = 'Time_Point' if 'Time_Point' in df.columns else 'Acq Date'
    df['_t'] = pd.to_numeric(df[tcol].astype(str).str.replace('-', ''), errors='coerce')
    df = df.sort_values(['Subject', '_t'])
    cn, mci = [], []
    for subj, g in df.groupby('Subject'):
        base_score = g.iloc[0][score_col]                       # V0 score (baseline visit)
        if len(g) >= 2:                                         # File contains follow-ups
            cohort, label = label_from(g.iloc[0]['Diagnosis'], g.iloc[1:]['Diagnosis'].max())
        elif subj in CONV_LOOKUP:                               # V0 only -> look up labels from ground truth
            cohort, label = CONV_LOOKUP[subj]
        else:
            continue
        if cohort is None:
            continue
        (cn if cohort == 'CN' else mci).append((label, base_score))
    to_arr = lambda lst: (np.array([a for a, _ in lst]), np.array([b for _, b in lst]))
    return to_arr(cn), to_arr(mci)

# ---- Cross-fold mean ROC +/- std ----
def mean_roc(fold_data):
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    for y, s in fold_data:
        if len(y) == 0 or len(np.unique(y)) < 2:
            continue
        X = s.reshape(-1, 1)                                   # Single feature
        clf = LogisticRegression(class_weight='balanced')     # Logistic regression
        clf.fit(X, y)
        proba = clf.predict_proba(X)[:, 1]                    # Predict probability
        fpr, tpr, _ = roc_curve(y, proba)                     # ROC from probability
        t = np.interp(mean_fpr, fpr, tpr); t[0] = 0.0
        tprs.append(t)
        aucs.append(roc_auc_score(y, proba))
    if not tprs:
        return None
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    return mean_fpr, mean_tpr, np.std(tprs, axis=0), np.mean(aucs), np.std(aucs)

# ---- Collect per-fold data for each method ----
results = {m: {'CN': [], 'MCI': []} for m in METHODS}
for m, (tmpl, score_col, _, method_folds) in METHODS.items():
    for f in method_folds:                                       # Use method-specific folds
        p = tmpl.format(f=f)
        if not os.path.exists(p):
            print(f"Warning: missing file ({m}): {p}"); continue
        df = pd.read_csv(p)
        cn, mci = build_conversion(df, score_col)
        results[m]['CN'].append(cn); results[m]['MCI'].append(mci)

# ---- Plot: two subplots ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
tasks = [('CN', 'CN$\\rightarrow$MCI'), ('MCI', 'MCI$\\rightarrow$AD')]
for ax, (key, title) in zip(axes, tasks):
    for m, (_, _, color, _) in METHODS.items():
        out = mean_roc(results[m][key])
        if out is None: continue
        fpr, tpr, std, a_m, a_s = out
        lw = 3.0 if m == 'Ours' else 1.8
        ax.plot(fpr, tpr, color=color, lw=lw, label=f'{m} (AUC={a_m:.3f}±{a_s:.3f})')
        ax.fill_between(fpr, np.clip(tpr - std, 0, 1), np.clip(tpr + std, 0, 1), color=color, alpha=0.12)
    ax.plot([0, 1], [0, 1], ls='--', color='gray', lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontweight='bold', fontsize=12)
    ax.set_ylabel("True Positive Rate", fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=13)
    leg = ax.legend(loc='lower right', fontsize=9, frameon=True)
    for t in leg.get_texts(): t.set_fontweight('bold')
    for lab in ax.get_xticklabels() + ax.get_yticklabels(): lab.set_fontweight('bold')

plt.tight_layout()
plt.savefig(SAVE_PATH, bbox_inches='tight')
print(f"ROC curves saved to: {SAVE_PATH}")

# ---- Print AUC (mean +/- std) for all models ----
print("\n" + "="*60)
print(f"{'Method':<16}{'CN→MCI AUC':<22}{'MCI→AD AUC':<22}")
print("="*60)
for m in METHODS:
    row = f"{m:<16}"
    for key in ['CN', 'MCI']:
        out = mean_roc(results[m][key])
        if out is None:
            row += f"{'N/A':<22}"
        else:
            _, _, _, a_m, a_s = out
            row += f"{a_m:.4f} ± {a_s:.4f}      "
    print(row)
print("="*60)