import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# ==========================================
# 0. Data Preparation and Conversion Label Extraction
# ==========================================
FOLD_TO_ANALYZE = 3
csv_path = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', f'fold_{FOLD_TO_ANALYZE}', f'predictions_fold_{FOLD_TO_ANALYZE}.csv')
save_dir = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', f'fold_{FOLD_TO_ANALYZE}', 'new_plots')
os.makedirs(save_dir, exist_ok=True)

df_all = pd.read_csv(csv_path)
df_all = df_all.sort_values(by=['Subject', 'Time_Point']).reset_index(drop=True)

# Extract Conversion Cohorts
# Only consider patients with baseline 1 (CN) or 2 (MCI) who have follow-up visits
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
            status = 'sCN'  # Stable CN
            label = 0
        elif future_max_diag >= 2:
            status = 'pCN'  # Progressive CN (converted to 2 or 3)
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
            status = 'pMCI' # Progressive MCI (converted to 3 AD)
            label = 1
        else:
            continue # Possible reversion 2->1, typically excluded from progression prediction analysis
        cohort_data.append({'Subject': subject, 'Baseline_Stage': 'MCI', 'Status': status, 'Conversion': label, 'Base_Score': base_score})

df_cohort = pd.DataFrame(cohort_data)

def calc_r_effect_size(u_stat, n1, n2):
    return 1 - (2 * u_stat) / (n1 * n2)

# ==========================================
# Experiment 1: Baseline Position Comparison & ML Prediction
# ==========================================
print("\n" + "="*50)
print("=== Experiment 1: Baseline score (Baseline mu) discrimination of future conversion ===")
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
    
    # Statistical test
    u_stat, p_val = stats.mannwhitneyu(stable, prog, alternative='two-sided')
    r_size = calc_r_effect_size(u_stat, n_s, n_p)
    print(f"   -> Mann-Whitney U: {u_stat:.1f}, p-value: {p_val:.2e}, Effect Size (r): {abs(r_size):.4f}")
    
    # ML prediction model (Logistic Regression based on Baseline mu)
    X = df_stage[['Base_Score']]
    y = df_stage['Conversion']
    
    # Note: fitting on all data to demonstrate baseline score prediction capability
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, y)
    y_pred_proba = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    print(f"   -> ML Model Prediction AUC: {auc:.4f}")

# ==========================================
# Experiment 2: Risk Stratification (Quartile Analysis)
# ==========================================
print("\n" + "="*50)
print("=== Experiment 2: Risk Stratification by Quartiles ===")
print("="*50)

df_cohort['Quartile'] = np.nan
for stage in ['CN', 'MCI']:
    idx = df_cohort['Baseline_Stage'] == stage
    if idx.sum() > 3: # Ensure enough data for four quartiles
        # Split current stage by Base_Score into Q1 (lowest) to Q4 (highest)
        df_cohort.loc[idx, 'Quartile'] = pd.qcut(df_cohort.loc[idx, 'Base_Score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

quartile_stats = df_cohort.dropna(subset=['Quartile']).groupby(['Baseline_Stage', 'Quartile'])['Conversion'].agg(['count', 'mean']).reset_index()
quartile_stats['Conversion_Rate'] = quartile_stats['mean'] * 100

for stage in ['CN', 'MCI']:
    print(f"\n[{stage} Cohort] Conversion rate by baseline score quartile (Q1: lowest -> Q4: highest):")
    subset = quartile_stats[quartile_stats['Baseline_Stage'] == stage]
    for _, row in subset.iterrows():
        print(f"   {row['Quartile']}: n={row['count']:2.0f} | conversion_rate={row['Conversion_Rate']:5.2f}%")


# ==========================================
# Plotting section (2-row 2-column layout: CN and MCI Violin comparison)
# ==========================================
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(14, 11), dpi=300)

# Use GridSpec for fine-grained subplot layout
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

# --- Subplot 1: CN cohort Violin comparison (sCN vs pCN) ---
ax1 = fig.add_subplot(gs[0, 0])
df_plot_cn = df_cohort[df_cohort['Baseline_Stage'] == 'CN']
if len(df_plot_cn) > 0 and len(df_plot_cn['Status'].unique()) > 1:
    sns.violinplot(x='Status', y='Base_Score', data=df_plot_cn, order=['sCN', 'pCN'],
                   hue='Status', palette={'sCN':'#2ecc71', 'pCN':'#e74c3c'}, inner=None, alpha=0.7,
                   legend=False, ax=ax1)
    sns.pointplot(x='Status', y='Base_Score', data=df_plot_cn, order=['sCN', 'pCN'], 
                  color='black', errorbar='sd', capsize=0.1, marker='D', linestyle="none", ax=ax1)
    ax1.set_title("Baseline $\mu$ Difference (CN Cohort)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Baseline Position Score $\mu$", fontweight='bold')
    ax1.set_xlabel("")
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontweight('bold')
else:
    ax1.text(0.5, 0.5, "Insufficient CN Progression Data", ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title("Baseline $\mu$ Difference (CN Cohort)", fontsize=13, fontweight='bold')

# --- Subplot 2: MCI cohort Violin comparison (sMCI vs pMCI) ---
ax2 = fig.add_subplot(gs[0, 1])
df_plot_mci = df_cohort[df_cohort['Baseline_Stage'] == 'MCI']
if len(df_plot_mci) > 0 and len(df_plot_mci['Status'].unique()) > 1:
    sns.violinplot(x='Status', y='Base_Score', data=df_plot_mci, order=['sMCI', 'pMCI'], 
                   hue='Status', palette={'sMCI':'#3498db', 'pMCI':'#e74c3c'}, inner=None, alpha=0.7, 
                   legend=False, ax=ax2)
    sns.pointplot(x='Status', y='Base_Score', data=df_plot_mci, order=['sMCI', 'pMCI'], 
                  color='black', errorbar='sd', capsize=0.1, marker='D', linestyle="none", ax=ax2)
    ax2.set_title("Baseline $\mu$ Difference (MCI Cohort)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Baseline Position Score $\mu$", fontweight='bold')
    ax2.set_xlabel("")
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontweight('bold')
else:
    ax2.text(0.5, 0.5, "Insufficient MCI Progression Data", ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title("Baseline $\mu$ Difference (MCI Cohort)", fontsize=13, fontweight='bold')

# --- Subplot 3: Risk stratification bar plot (spanning full bottom row) ---
ax3 = fig.add_subplot(gs[1, :])
if len(quartile_stats) > 0:
    sns.barplot(x='Quartile', y='Conversion_Rate', hue='Baseline_Stage', data=quartile_stats, 
                palette={'CN':'#2ecc71', 'MCI':'#f39c12'}, ax=ax3)
    ax3.set_title("Conversion Rate by Baseline $\mu$ Quartiles", fontsize=13, fontweight='bold')
    ax3.set_ylabel("Conversion Rate to Next Stage (%)", fontweight='bold')
    ax3.set_xlabel("Baseline Score Quartile (Low $\mu$ $\\rightarrow$ High $\mu$)", fontweight='bold')
    
    # Annotate percentage on top of bars
    for p in ax3.patches:
        height = p.get_height()
        if height > 0:
            ax3.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height), 
                         ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=10, fontweight='bold')
            
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontweight('bold')
        
    ax3.legend(title='Baseline Stage', title_fontsize=11, fontsize=10, frameon=False)
    sns.despine(ax=ax3)

sns.despine()
plot_path = os.path.join(save_dir, f"exp_conversion_and_risk_all_folds_{FOLD_TO_ANALYZE}.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nConversion and risk prediction plot (CN and MCI) saved to: {plot_path}")