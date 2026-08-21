import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def set_times():
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 0. Data preparation and Ambiguity (sigma) extraction
# ==========================================
folds = [1, 2, 3] # Can be changed to single fold [3] or multi-fold [1, 2, 3]
fold_str = "_".join(map(str, folds)) if len(folds) > 1 else f"fold_{folds[0]}"
df_list = []

print(f"Reading and merging specified Fold(s): {folds} (Variance Distribution experiment)...")
for f in folds:
    csv_path = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', f'fold_{f}', f'predictions_fold_{f}.csv')
    if os.path.exists(csv_path):
        df_fold = pd.read_csv(csv_path)
        df_fold['Fold'] = f
        df_list.append(df_fold)

df_all = pd.concat(df_list, ignore_index=True)
df_all = df_all.sort_values(by=['Subject', 'Time_Point']).reset_index(drop=True)

# Dynamic save path
if len(folds) > 1:
    save_dir = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', 'merged_folds', 'new_plots')
else:
    save_dir = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', f'fold_{folds[0]}', 'new_plots')
os.makedirs(save_dir, exist_ok=True)

# Auto-detect variance/standard deviation column name
sigma_col = None
for col in df_all.columns:
    if 'sigma' in col.lower() or 'var' in col.lower() or 'std' in col.lower():
        sigma_col = col
        break

# Extract cohort data and progression speed calculation
cohort_data = []

for subject, group in df_all.groupby('Subject'):
    if len(group) < 2:
        continue # No future data, cannot determine conversion
        
    base_row = group.iloc[0]
    base_diag = base_row['Diagnosis']
    base_mu = base_row['Pred_Score']
    
    # Extract baseline sigma
    if sigma_col and not pd.isna(base_row[sigma_col]):
        if 'logvar' in sigma_col.lower():
            base_sigma = np.sqrt(np.exp(base_row[sigma_col]))
        else:
            base_sigma = base_row[sigma_col]
    else:
        base_sigma = 0.0 # Fallback

    # Find highest severity in future follow-ups
    future_max_diag = group.iloc[1:]['Diagnosis'].max()
    
    # Compute longitudinal progression speed (Mu_last - Mu_0) / total_days (for experiment 2)
    mu_last = group.iloc[-1]['Pred_Score']
    if 'Days' in group.columns:
        total_days = group.iloc[-1]['Days'] - group.iloc[0]['Days']
    else:
        total_days = group.iloc[-1]['Time_Point'] - group.iloc[0]['Time_Point']
    prog_speed = (mu_last - base_mu) / max(total_days, 1.0)

    if base_diag == 1:
        if future_max_diag == 1:
            status = 'sCN'
            label = 0
        elif future_max_diag >= 2:
            status = 'pCN'
            label = 1
        else:
            continue
        cohort_data.append({
            'Subject': subject, 'Baseline_Stage': 'CN', 'Status': status, 
            'Conversion': label, 'Base_Score': base_sigma, 'Base_Mu': base_mu, 'Prog_Speed': prog_speed
        })
        
    elif base_diag == 2:
        if future_max_diag == 2:
            status = 'sMCI'
            label = 0
        elif future_max_diag == 3:
            status = 'pMCI'
            label = 1
        else:
            continue
        cohort_data.append({
            'Subject': subject, 'Baseline_Stage': 'MCI', 'Status': status, 
            'Conversion': label, 'Base_Score': base_sigma, 'Base_Mu': base_mu, 'Prog_Speed': prog_speed
        })

df_cohort = pd.DataFrame(cohort_data)

def calc_r_effect_size(u_stat, n1, n2):
    return 1 - (2 * u_stat) / (n1 * n2)

# ==========================================
# Experiment 1 (Mu control exp 1): Baseline Sigma discrimination for future conversion and combined prediction
# ==========================================
print("\n" + "="*50)
print("=== Experiment 1 (Sigma): Baseline Ambiguity discrimination for future conversion and combined prediction ===")
print("="*50)

for stage in ['CN', 'MCI']:
    df_stage = df_cohort[df_cohort['Baseline_Stage'] == stage]
    stable = df_stage[df_stage['Conversion'] == 0]['Base_Score']
    prog = df_stage[df_stage['Conversion'] == 1]['Base_Score']
    
    n_s, n_p = len(stable), len(prog)
    if n_s == 0 or n_p == 0:
        print(f"[{stage}] Insufficient samples: stable={n_s}, progressive={n_p}")
        continue
        
    print(f"\n[{stage} Cohort] (sigma analysis) - Total: {n_s + n_p} (Stable: {n_s}, Progressive: {n_p})")
    print(f"   Stable Mean: {stable.mean():.4f} | Progressive Mean: {prog.mean():.4f}")
    
    u_stat, p_val = stats.mannwhitneyu(stable, prog, alternative='two-sided')
    r_size = calc_r_effect_size(u_stat, n_s, n_p)
    print(f"   -> Mann-Whitney U ($\sigma$): {u_stat:.1f}, p-value: {p_val:.2e}, Effect Size (r): {abs(r_size):.4f}")
    
    # AUC using Sigma only
    clf_sigma = LogisticRegression(class_weight='balanced')
    clf_sigma.fit(df_stage[['Base_Score']], df_stage['Conversion'])
    auc_sigma = roc_auc_score(df_stage['Conversion'], clf_sigma.predict_proba(df_stage[['Base_Score']])[:, 1])
    print(f"   -> ML Model Prediction AUC (Only $\sigma$): {auc_sigma:.4f}")

    # Combined model (mu + sigma) prediction AUC
    clf_combo = LogisticRegression(class_weight='balanced')
    clf_combo.fit(df_stage[['Base_Mu', 'Base_Score']], df_stage['Conversion'])
    auc_combo = roc_auc_score(df_stage['Conversion'], clf_combo.predict_proba(df_stage[['Base_Mu', 'Base_Score']])[:, 1])
    print(f"   -> ML Model Prediction AUC (mu + sigma combined): {auc_combo:.4f}")

# ==========================================
# Experiment 1.2 (Mu control exp 2): Sigma risk stratification (Quartile Analysis)
# ==========================================
print("\n" + "="*50)
print("=== Experiment 1.2 (Sigma): Risk Stratification by Sigma Quartiles ===")
print("="*50)

df_cohort['Quartile'] = pd.Categorical([np.nan] * len(df_cohort), categories=['Q1', 'Q2', 'Q3', 'Q4'], ordered=True)
for stage in ['CN', 'MCI']:
    idx = df_cohort['Baseline_Stage'] == stage
    if idx.sum() > 3:
        q_labels = pd.qcut(df_cohort.loc[idx, 'Base_Score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        df_cohort.loc[idx, 'Quartile'] = q_labels

quartile_stats = df_cohort.dropna(subset=['Quartile']).groupby(['Baseline_Stage', 'Quartile'], observed=False)['Conversion'].agg(['count', 'mean']).reset_index()
quartile_stats['Conversion_Rate'] = quartile_stats['mean'] * 100

for stage in ['CN', 'MCI']:
    print(f"\n[{stage} Cohort] Conversion rate by baseline sigma quartiles (Q1: lowest -> Q4: highest):")
    subset = quartile_stats[quartile_stats['Baseline_Stage'] == stage]
    for _, row in subset.iterrows():
        print(f"   {row['Quartile']}: samples={row['count']:2.0f} | conversion_rate={row['Conversion_Rate']:5.2f}%")


# ==========================================
# Experiment 2: Position Ambiguity and disease progression speed correlation
# ==========================================
print("\n" + "="*50)
print("=== Experiment 2: Position Ambiguity and disease progression speed correlation ===")
print("="*50)

for stage in ['CN', 'MCI']:
    df_stage = df_cohort[df_cohort['Baseline_Stage'] == stage].dropna(subset=['Prog_Speed', 'Base_Score'])
    if len(df_stage) < 3: continue
    
    corr_p, p_val_p = stats.pearsonr(df_stage['Base_Score'], df_stage['Prog_Speed'])
    corr_s, p_val_s = stats.spearmanr(df_stage['Base_Score'], df_stage['Prog_Speed'])
    
    print(f"\n[{stage} Cohort] (samples: {len(df_stage)})")
    print(f"   -> Pearson Correlation: r = {corr_p:.4f}, p-value = {p_val_p:.2e}")
    print(f"   -> Spearman Correlation: rho = {corr_s:.4f}, p-value = {p_val_s:.2e}")


# ==========================================
# Plot section (2x2 layout: replicating Mu visualization style)
# ==========================================
sns.set_theme(style="whitegrid")
set_times()
fig = plt.figure(figsize=(14, 11), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)
YLAB = "DCS Uncertainty of Baseline Scans"

# --- Subplot 1: CN cohort Violin comparison (sCN vs pCN) ---
ax1 = fig.add_subplot(gs[0, 0])
df_plot_cn = df_cohort[df_cohort['Baseline_Stage'] == 'CN']
if len(df_plot_cn) > 0 and df_plot_cn['Status'].nunique() > 1:
    sns.violinplot(x='Status', y='Base_Score', hue='Status', data=df_plot_cn, order=['sCN', 'pCN'],
                   palette={'sCN':'#a9dfbf', 'pCN':'#27ae60'}, inner=None, alpha=0.7, legend=False, cut=0, ax=ax1)
    sns.pointplot(x='Status', y='Base_Score', data=df_plot_cn, order=['sCN', 'pCN'],
                  color='black', errorbar='sd', capsize=0.1, marker='D', linestyle="none", ax=ax1)
else:
    ax1.text(0.5, 0.5, "Insufficient CN Progression Data", ha='center', va='center', transform=ax1.transAxes, fontweight='bold')
ax1.set_title(""); ax1.set_ylabel(YLAB, fontweight='bold'); ax1.set_xlabel("")
ax1.set_ylim(0.88, 1.00)                                   # Unified range
for lab in (ax1.get_xticklabels() + ax1.get_yticklabels()): lab.set_fontweight('bold')


# --- Subplot 2: MCI cohort Violin comparison (sMCI vs pMCI) ---
ax2 = fig.add_subplot(gs[0, 1])
df_plot_mci = df_cohort[df_cohort['Baseline_Stage'] == 'MCI']
if len(df_plot_mci) > 0 and df_plot_mci['Status'].nunique() > 1:
    sns.violinplot(x='Status', y='Base_Score', hue='Status', data=df_plot_mci, order=['sMCI', 'pMCI'],
                   palette={'sMCI':'#f8c471', 'pMCI':'#e67e22'}, inner=None, alpha=0.7, legend=False, cut=0, ax=ax2)
    sns.pointplot(x='Status', y='Base_Score', data=df_plot_mci, order=['sMCI', 'pMCI'],
                  color='black', errorbar='sd', capsize=0.1, marker='D', linestyle="none", ax=ax2)
else:
    ax2.text(0.5, 0.5, "Insufficient MCI Progression Data", ha='center', va='center', transform=ax2.transAxes, fontweight='bold')
ax2.set_title(""); ax2.set_ylabel(YLAB, fontweight='bold'); ax2.set_xlabel("")
ax2.set_ylim(0.88, 1.00)                                   # Unified range
for lab in (ax2.get_xticklabels() + ax2.get_yticklabels()): lab.set_fontweight('bold')


# --- Subplot 3: Risk stratification Bar plot (spanning bottom row) ---
ax3 = fig.add_subplot(gs[1, :])
if len(quartile_stats) > 0:
    quartile_stats['xkey'] = quartile_stats['Baseline_Stage'] + ' ' + quartile_stats['Quartile'].astype(str)
    xorder = [f'{s} Q{i}' for s in ['CN', 'MCI'] for i in range(1, 5)]
    sns.barplot(data=quartile_stats, x='xkey', y='Conversion_Rate', hue='Baseline_Stage',
                order=xorder, hue_order=['CN', 'MCI'], dodge=False,
                palette={'CN':'#2ecc71', 'MCI':'#f39c12'}, ax=ax3)
    ax3.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4'], fontweight='bold')
    for p in ax3.patches:
        h = p.get_height()
        if h > 0:
            ax3.annotate(f'{h:.1f}%', (p.get_x() + p.get_width()/2., h), ha='center', va='center',
                         xytext=(0, 7), textcoords='offset points', fontsize=10, fontweight='bold')
    ax3.set_title(""); ax3.set_ylabel("Conversion Rate (%)", fontweight='bold'); ax3.set_xlabel("")
    ax3.set_ylim(0, 30)
    for lab in (ax3.get_xticklabels() + ax3.get_yticklabels()): lab.set_fontweight('bold')
    ax3.legend(title=None, fontsize=10, frameon=True, loc='upper left')
    plt.setp(ax3.get_legend().get_texts(), fontweight='bold')
    sns.despine(ax=ax3)

sns.despine()
plot_path = os.path.join(save_dir, "exp_ambiguity_and_risk.pdf")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nSaved to: {plot_path}")