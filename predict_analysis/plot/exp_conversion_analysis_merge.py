import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib
from matplotlib.ticker import FormatStrFormatter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def set_times():
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 0. Auto-merge prediction results from all folds
# ==========================================
folds = [1, 2, 3] # Can be changed to [1, 2, 3, 4, 5] for 5-fold
df_list = []

print("Reading and merging all fold prediction results (for global conversion and risk analysis)...")
for f in folds:
    csv_path = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', f'fold_{f}', f'predictions_fold_{f}.csv')
    if os.path.exists(csv_path):
        df_fold = pd.read_csv(csv_path)
        df_fold['Fold'] = f
        df_list.append(df_fold)
        print(f"  -> Loaded Fold {f}, records: {len(df_fold)}")
    else:
        print(f"Warning: file not found {csv_path}")

if len(df_list) > 0:
    df_all = pd.concat(df_list, ignore_index=True)
else:
    raise FileNotFoundError("No prediction result files found for any fold, please check paths!")

df_all = df_all.sort_values(by=['Subject', 'Time_Point']).reset_index(drop=True)

# Create directory for merged results
save_dir = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', 'merged_folds', 'new_plots')
os.makedirs(save_dir, exist_ok=True)

# Extract Conversion Cohorts
cohort_data = []
for subject, group in df_all.groupby('Subject'):
    if len(group) < 2:
        continue # No future data, cannot determine conversion

    base_diag = group.iloc[0]['Diagnosis']
    base_score = group.iloc[0]['Pred_Score']

    # Find the highest severity in future follow-ups
    future_max_diag = group.iloc[1:]['Diagnosis'].max()

    if base_diag == 1:
        # Conversion logic for CN (diagnosis=1)
        if future_max_diag == 1:
            status = 'sCN'  # Stable CN (remained at 1)
            label = 0
        elif future_max_diag >= 2:
            status = 'pCN'  # Progressive CN (converted to MCI or AD)
            label = 1
        else:
            continue
        cohort_data.append({'Subject': subject, 'Baseline_Stage': 'CN', 'Status': status, 'Conversion': label, 'Base_Score': base_score})

    elif base_diag == 2:
        # Conversion logic for MCI (diagnosis=2)
        if future_max_diag == 2:
            status = 'sMCI' # Stable MCI (remained at 2)
            label = 0
        elif future_max_diag == 3:
            status = 'pMCI' # Progressive MCI (converted to AD)
            label = 1
        else:
            continue
        cohort_data.append({'Subject': subject, 'Baseline_Stage': 'MCI', 'Status': status, 'Conversion': label, 'Base_Score': base_score})

df_cohort = pd.DataFrame(cohort_data)

# Clip Base_Score lower bound to 0 to prevent small negative values from mathematical smoothing
# df_cohort['Base_Score'] = df_cohort['Base_Score'].clip(lower=0.0)

def calc_r_effect_size(u_stat, n1, n2):
    return 1 - (2 * u_stat) / (n1 * n2)

# ==========================================
# Experiment 1: Baseline Position Comparison & ML Prediction (Merged)
# ==========================================
print("\n" + "="*50)
print("=== [MERGED] Experiment 1: Baseline score ($\mu$) discrimination of future conversion ===")
print("="*50)

for stage in ['CN', 'MCI']:
    df_stage = df_cohort[df_cohort['Baseline_Stage'] == stage]
    stable = df_stage[df_stage['Conversion'] == 0]['Base_Score']
    prog = df_stage[df_stage['Conversion'] == 1]['Base_Score']

    n_s, n_p = len(stable), len(prog)
    if n_s == 0 or n_p == 0:
        print(f"[{stage}] Insufficient samples: stable={n_s}, progressive={n_p}")
        continue

    print(f"\n[{stage} Cohort] - Total: {n_s + n_p} (Stable: {n_s}, Progressive: {n_p})")
    print(f"   Stable Mean: {stable.mean():.4f} | Progressive Mean: {prog.mean():.4f}")

    u_stat, p_val = stats.mannwhitneyu(stable, prog, alternative='two-sided')
    r_size = calc_r_effect_size(u_stat, n_s, n_p)
    print(f"   -> Mann-Whitney U: {u_stat:.1f}, p-value: {p_val:.2e}, Effect Size (r): {abs(r_size):.4f}")

    X = df_stage[['Base_Score']]
    y = df_stage['Conversion']
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, y)
    y_pred_proba = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    print(f"   -> ML Model Prediction AUC: {auc:.4f}")

# ==========================================
# Experiment 2: Risk Stratification (Quartile Analysis - Merged)
# ==========================================
print("\n" + "="*50)
print("=== [MERGED] Experiment 2: Risk Stratification by Quartiles ===")
print("="*50)

# Initialize Categorical column to avoid future pandas warnings
df_cohort['Quartile'] = pd.Categorical([np.nan] * len(df_cohort), categories=['Q1', 'Q2', 'Q3', 'Q4'], ordered=True)

for stage in ['CN', 'MCI']:
    idx = df_cohort['Baseline_Stage'] == stage
    if idx.sum() > 3:
        q_labels = pd.qcut(df_cohort.loc[idx, 'Base_Score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        df_cohort.loc[idx, 'Quartile'] = q_labels

quartile_stats = df_cohort.dropna(subset=['Quartile']).groupby(['Baseline_Stage', 'Quartile'], observed=False)['Conversion'].agg(['count', 'mean']).reset_index()
quartile_stats['Conversion_Rate'] = quartile_stats['mean'] * 100

for stage in ['CN', 'MCI']:
    print(f"\n[{stage} Cohort] Conversion rate by baseline score quartile (Q1: lowest -> Q4: highest):")
    subset = quartile_stats[quartile_stats['Baseline_Stage'] == stage]
    for _, row in subset.iterrows():
        print(f"   {row['Quartile']}: n={row['count']:2.0f} | conversion_rate={row['Conversion_Rate']:5.2f}%")


# ==========================================
# Plotting section (same smoothing as single fold)
# ==========================================
sns.set_theme(style="whitegrid")
set_times()  # Set Times New Roman font and math font
fig = plt.figure(figsize=(14, 11), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

# --- Subplot 1: CN cohort (sCN vs pCN) ---
ax1 = fig.add_subplot(gs[0, 0])
df_plot_cn = df_cohort[df_cohort['Baseline_Stage'] == 'CN']
if len(df_plot_cn) > 0 and df_plot_cn['Status'].nunique() > 1:
    sns.violinplot(x='Status', y='Base_Score', hue='Status', data=df_plot_cn, order=['sCN', 'pCN'],
                   palette={'sCN':'#a9dfbf', 'pCN':'#27ae60'}, inner=None, alpha=0.7, legend=False, cut=0, ax=ax1)  # Both green
    sns.pointplot(x='Status', y='Base_Score', data=df_plot_cn, order=['sCN', 'pCN'],
                  color='black', errorbar='sd', capsize=0.1, marker='D', linestyle="none", ax=ax1)
else:
    ax1.text(0.5, 0.5, "Insufficient CN Progression Data", ha='center', va='center', transform=ax1.transAxes)
ax1.set_title("")                                     # Remove top title
ax1.set_ylabel("DCS of Baseline Scans", fontweight='bold')
ax1.set_xlabel("")
ax1.set_ylim(-1, 7)                                   # Unified range
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))   # Format y-axis as decimal
for lab in (ax1.get_xticklabels() + ax1.get_yticklabels()): lab.set_fontweight('bold')


# --- Subplot 2: MCI cohort Violin (sMCI vs pMCI) ---
ax2 = fig.add_subplot(gs[0, 1])
df_plot_mci = df_cohort[df_cohort['Baseline_Stage'] == 'MCI']
if len(df_plot_mci) > 0 and df_plot_mci['Status'].nunique() > 1:
    sns.violinplot(x='Status', y='Base_Score', hue='Status', data=df_plot_mci, order=['sMCI', 'pMCI'],
                   palette={'sMCI':'#f8c471', 'pMCI':'#e67e22'}, inner=None, alpha=0.7, legend=False, cut=0, ax=ax2)  # Both orange
    sns.pointplot(x='Status', y='Base_Score', data=df_plot_mci, order=['sMCI', 'pMCI'],
                  color='black', errorbar='sd', capsize=0.1, marker='D', linestyle="none", ax=ax2)
else:
    ax2.text(0.5, 0.5, "Insufficient MCI Progression Data", ha='center', va='center', transform=ax2.transAxes)
ax2.set_title("")
ax2.set_ylabel("DCS of Baseline Scans", fontweight='bold')
ax2.set_xlabel("")
ax2.set_ylim(-1, 7)                                   # Unified range
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
for lab in (ax2.get_xticklabels() + ax2.get_yticklabels()): lab.set_fontweight('bold')


# --- Subplot 3: Risk stratification bar plot (spanning full bottom row) ---
ax3 = fig.add_subplot(gs[1, :])
if len(quartile_stats) > 0:
    quartile_stats['xkey'] = quartile_stats['Baseline_Stage'] + ' ' + quartile_stats['Quartile'].astype(str)
    xorder = [f'{s} Q{i}' for s in ['CN', 'MCI'] for i in range(1, 5)]   # CN Q1-Q4 | MCI Q1-Q4
    sns.barplot(data=quartile_stats, x='xkey', y='Conversion_Rate', hue='Baseline_Stage',
                order=xorder, hue_order=['CN', 'MCI'], dodge=False,
                palette={'CN':'#2ecc71', 'MCI':'#f39c12'}, ax=ax3)

    ax3.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4'], fontweight='bold')

    for p in ax3.patches:
        h = p.get_height()
        if h > 0:
            ax3.annotate(f'{h:.1f}%', (p.get_x() + p.get_width()/2., h), ha='center', va='center',
                         xytext=(0, 7), textcoords='offset points', fontsize=10, fontweight='bold')

    ax3.set_title("")                                 # Remove top title
    ax3.set_ylabel("Conversion Rate (%)", fontweight='bold')
    ax3.set_xlabel("")                                # Remove bottom label
    for lab in (ax3.get_xticklabels() + ax3.get_yticklabels()): lab.set_fontweight('bold')
    ax3.legend(title=None, fontsize=10, frameon=False)   # Remove "Baseline Stage" title
    plt.setp(ax3.get_legend().get_texts(), fontweight='bold')
    sns.despine(ax=ax3)

sns.despine()
plot_path = os.path.join(save_dir, "exp_conversion_and_risk_merged.pdf")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nSaved to: {plot_path}")
