import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt
import matplotlib.collections as collections
import os
from scipy import stats
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def set_times():
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    matplotlib.rcParams['mathtext.fontset'] = 'stix'   # Times-style math font
    matplotlib.rcParams['axes.unicode_minus'] = False
set_times()

# ==========================================
# 0. Auto-merge data from all folds and map labels
# ==========================================
folds = [1, 2, 3]
df_list = []

print("Reading and merging data from all folds...")
for f in folds:
    csv_path = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', f'fold_{f}', f'predictions_fold_{f}.csv')
    if os.path.exists(csv_path):
        df_list.append(pd.read_csv(csv_path))
    else:
        print(f"Warning: file not found {csv_path}")

# Concatenate all fold data
df_all = pd.concat(df_list, ignore_index=True)

# Map 1, 2, 3 to clinical abbreviations CN, MCI, AD
diag_map = {1: 'CN', 2: 'MCI', 3: 'AD', '1': 'CN', '2': 'MCI', '3': 'AD'}
df_all['Diagnosis'] = df_all['Diagnosis'].map(diag_map)

# Extract V0 baseline
df_v0 = df_all.sort_values(by=['Subject', 'Time_Point']).drop_duplicates(subset=['Subject'], keep='first').copy()

print(f"Merge complete. Total longitudinal records: {len(df_all)}")
print(f"Extracted baseline (V0) subjects: {len(df_v0)}")

# Create directory for merged results
save_dir = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', 'merged_folds', 'new_plots')
os.makedirs(save_dir, exist_ok=True)

# Save merged V0 data for later reference
v0_output_path = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', 'merged_folds', 'v0_predictions_merged.csv')
df_v0.to_csv(v0_output_path, index=False)
print(f"Merged V0 Baseline table saved to: {v0_output_path}")

# Global config: dict and list color palettes
order = ['CN', 'MCI', 'AD']
palette_list = ['#2ecc71', '#f39c12', '#e74c3c']
palette_dict = {'CN': '#2ecc71', 'MCI': '#f39c12', 'AD': '#e74c3c'}

# ==========================================
# 1. Experiment 1: Statistical Analysis (Kruskal-Wallis, Mann-Whitney U, FDR, Effect Size)
# ==========================================
print("\n" + "="*40)
print(f"=== Experiment 1: V0 Statistical Analysis Report (Merged Folds) ===")
print("="*40)

group_CN = df_v0[df_v0['Diagnosis'] == 'CN']['Pred_Score']
group_MCI = df_v0[df_v0['Diagnosis'] == 'MCI']['Pred_Score']
group_AD = df_v0[df_v0['Diagnosis'] == 'AD']['Pred_Score']

h_stat, p_kw = stats.kruskal(group_CN, group_MCI, group_AD)
print(f"Kruskal-Wallis H-statistic: {h_stat:.4f}, p-value: {p_kw:.2e}")

comparisons = [
    ('CN', 'MCI', group_CN, group_MCI),
    ('MCI', 'AD', group_MCI, group_AD),
    ('CN', 'AD', group_CN, group_AD)
]

p_values_uncorrected = []
effect_sizes = []

def calc_r_effect_size(u_stat, n1, n2):
    return 1 - (2 * u_stat) / (n1 * n2)

print("\n--- Pairwise Comparison (Uncorrected) ---")
for name1, name2, g1, g2 in comparisons:
    u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    p_values_uncorrected.append(p_val)
    r_size = calc_r_effect_size(u_stat, len(g1), len(g2))
    effect_sizes.append(abs(r_size))
    print(f"{name1} vs {name2} | U: {u_stat:.1f} | p-value: {p_val:.2e} | Effect Size (r): {abs(r_size):.4f}")

reject, pvals_corrected, _, _ = multipletests(p_values_uncorrected, alpha=0.05, method='fdr_bh')
print("\n--- Pairwise Comparison (FDR Corrected) ---")
for idx, (name1, name2, _, _) in enumerate(comparisons):
    p_val_corr = pvals_corrected[idx]
    print(f"{name1} vs {name2} | Corrected p-value = {p_val_corr:.2e} (approx. {p_val_corr:.6f})")

# ==========================================
# 2. Experiment 1: Violin Plot with Mean/SD
# ==========================================
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")
set_times()

ax1 = sns.violinplot(
    x="Diagnosis", y="Pred_Score", hue="Diagnosis", data=df_v0,
    order=order, palette=palette_dict, inner=None, linewidth=1.5, alpha=0.6, legend=False, cut=0
)
sns.pointplot(
    x="Diagnosis", y="Pred_Score", data=df_v0, order=order, color="black",
    errorbar="sd", capsize=0.1, marker="D", linestyle="none", ax=ax1
)

ax1.set_title("")
ax1.set_ylabel("DCS", fontsize=12, fontweight='bold')
ax1.set_xlabel("")
ax1.set_ylim(-0.5, 15)                       # Match raincloud range
ax1.set_yticks(np.arange(-2.5, 15.1, 2.5))      # Same tick interval
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "exp1_v0_violin_merged.pdf"), bbox_inches='tight')

# ==========================================
# 3. Experiment 3: Raincloud Plot of Continuous Distribution
# ==========================================
plt.figure(figsize=(12, 8))
sns.set_theme(style="ticks")
set_times()

pt.half_violinplot(
    x='Pred_Score', y='Diagnosis', data=df_v0,
    palette=palette_list, order=order,           # Unified palette
    orient='h', width=0.7, inner=None, cut=1.5, scale='width'
)
ax2 = sns.stripplot(
    x='Pred_Score', y='Diagnosis', hue='Diagnosis', data=df_v0,
    palette=palette_dict, order=order, legend=False,
    orient='h', size=5, jitter=0.12, alpha=0.85,  # Larger dots, darker
    edgecolor='white', linewidth=0.4, zorder=1       
)
sns.boxplot(
    x='Pred_Score', y='Diagnosis', data=df_v0,
    color='black', order=order,
    orient='h', width=0.15, showfliers=False,
    boxprops={'facecolor': 'none', 'zorder': 20},                     
    medianprops={'linewidth': 3, 'color': 'black', 'zorder': 21},     
    whiskerprops={'linewidth': 1.5, 'color': 'black'},
    capprops={'linewidth': 1.5, 'color': 'black'}
)

baselines = []
for coll in ax2.collections:
    if isinstance(coll, collections.PolyCollection):
        for path in coll.get_paths():
            path.vertices[:, 0] = np.clip(path.vertices[:, 0], 0, None)
            baselines.append(np.max(path.vertices[:, 1]))
if baselines:
    ax2.hlines(y=baselines, xmin=0, xmax=15, color='#4c4c4c', linewidth=2.0, zorder=0)

ax2.set_title("")
ax2.set_xlabel("DCS", fontsize=16, fontweight='bold')
ax2.set_ylabel("")
ax2.set_xlim(-0.5, 15)                        # Match violin range
ax2.set_xticks(np.arange(0, 15.1, 2.5))       # Same tick interval
sns.despine(top=True, right=True, left=False, bottom=False)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
ax2.tick_params(axis='y', length=0)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "exp3_v0_raincloud_merged.pdf"), bbox_inches='tight')