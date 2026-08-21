import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import ptitprince as pt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# ==========================================
# 0. Load data and extract V0 baselines
# ==========================================
# Assuming inference.py was just run for fold prediction
FOLD_TO_ANALYZE = 3
csv_path = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', f'fold_{FOLD_TO_ANALYZE}', f'predictions_fold_{FOLD_TO_ANALYZE}.csv')

print(f"Reading predictions from: {csv_path}")
df_all = pd.read_csv(csv_path)

# Core operation: extract V0 at Pandas level
# Sort by patient and time point, keep only first record per patient (keep='first')
df_v0 = df_all.sort_values(by=['Subject', 'Time_Point']).drop_duplicates(subset=['Subject'], keep='first').copy()
diag_map = {1: 'CN', 2: 'MCI', 3: 'AD', '1': 'CN', '2': 'MCI', '3': 'AD'}
df_v0['Diagnosis'] = df_v0['Diagnosis'].map(diag_map)

print(f"Total longitudinal records: {len(df_all)}")
print(f"Extracted Baseline (V0) subjects: {len(df_v0)}")

# Save pure V0 data as an independent CSV file
v0_output_path = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', f'fold_{FOLD_TO_ANALYZE}', f'v0_predictions_fold_{FOLD_TO_ANALYZE}.csv')
df_v0.to_csv(v0_output_path, index=False)
print(f"V0 Baseline table successfully saved to: {v0_output_path}")

# ==========================================================
# Use list instead of dict, ptitprince supports lists best
# ==========================================================
order = ['CN', 'MCI', 'AD']
palette_dict = {'CN': '#2ecc71', 'MCI': '#f39c12', 'AD': '#e74c3c'}

# ==========================================
# 1. Statistical analysis (Kruskal-Wallis, Mann-Whitney U, FDR, Effect Size)
# ==========================================
print("\n" + "="*40)
print(f"=== Experiment 1: V0 Statistical Analysis Report ===")
print("="*40)

# Use mapped 'CN', 'MCI', 'AD' to extract scores
group_CN = df_v0[df_v0['Diagnosis'] == 'CN']['Pred_Score']
group_MCI = df_v0[df_v0['Diagnosis'] == 'MCI']['Pred_Score']
group_AD = df_v0[df_v0['Diagnosis'] == 'AD']['Pred_Score']

# 1.1 Global Kruskal-Wallis H Test
h_stat, p_kw = stats.kruskal(group_CN, group_MCI, group_AD)
print(f"Kruskal-Wallis H-statistic: {h_stat:.4f}, p-value: {p_kw:.2e}")

# 1.2 Pairwise Mann-Whitney U Tests
# Use mapped group names for comparisons
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
    print(f"Stage {name1} vs Stage {name2} | U: {u_stat:.1f} | p-value: {p_val:.2e} | Effect Size (r): {abs(r_size):.4f}")

reject, pvals_corrected, _, _ = multipletests(p_values_uncorrected, alpha=0.05, method='fdr_bh')
print("\n--- Pairwise Comparison (FDR Corrected) ---")
for idx, (name1, name2, _, _) in enumerate(comparisons):
    p_val_corr = pvals_corrected[idx]
    print(f"Stage {name1} vs Stage {name2} | Corrected p-value = {p_val_corr:.2e} (approx. {p_val_corr:.6f})")

# ==========================================
# 2. Violin plot with mean/std
# ==========================================
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

ax1 = sns.violinplot(
    x="Diagnosis", y="Pred_Score", hue="Diagnosis", data=df_v0, 
    order=order, palette=palette_dict, inner=None, linewidth=1.5, alpha=0.6, legend=False
)

sns.pointplot(
    x="Diagnosis", y="Pred_Score", data=df_v0, order=order, color="black", 
    errorbar="sd", capsize=0.1, marker="D", linestyle="none", ax=ax1
)

plt.title(f"Distribution of Inferred Position Score $\mu$ (Fold {FOLD_TO_ANALYZE})", fontsize=14, pad=15)
plt.ylabel("Inferred Position Score $\mu$", fontsize=12)
plt.xlabel("Clinical Stage") 
plt.tight_layout()
save_path_violin = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', f'fold_{FOLD_TO_ANALYZE}', 'new_plots', f'exp1_v0_violin_fold_{FOLD_TO_ANALYZE}.png')
os.makedirs(os.path.dirname(save_path_violin), exist_ok=True)
plt.savefig(save_path_violin, dpi=300)
print(f"\nViolin plot saved.")

# ==========================================
# 3. Raincloud plot of continuous distribution
# ==========================================
plt.figure(figsize=(12, 8))
sns.set_theme(style="ticks")

# 1. Draw “cloud” (Half Violin)
pt.half_violinplot(
    x='Pred_Score', y='Diagnosis', data=df_v0,
    palette=['#2ecc71', '#f39c12', '#e74c3c'], order=order,
    orient='h', width=0.7, inner=None, cut=1.5, scale='width'
)

# 2. Draw “raindrops” (Stripplot)
ax = sns.stripplot(
    x='Pred_Score', y='Diagnosis', hue='Diagnosis', data=df_v0,
    palette=palette_dict, order=order, legend=False,
    orient='h', size=4, jitter=0.1, alpha=0.6
)

# 3. Draw “umbrella” (Boxplot)
sns.boxplot(
    x='Pred_Score', y='Diagnosis', data=df_v0,
    color='black', order=order,
    orient='h', width=0.15, showfliers=False,
    boxprops={'facecolor': 'none', 'zorder': 10}, 
    medianprops={'linewidth': 3, 'color': 'black'},
    whiskerprops={'linewidth': 1.5, 'color': 'black'},
    capprops={'linewidth': 1.5, 'color': 'black'}
)

# Dynamically extract true baselines and extend horizon lines
baselines = []
for coll in ax.collections:
    if isinstance(coll, collections.PolyCollection):
        for path in coll.get_paths():
            path.vertices[:, 0] = np.clip(path.vertices[:, 0], 0, None)
            baseline_y = np.max(path.vertices[:, 1])
            baselines.append(baseline_y)

# Visual optimization: extend horizon lines
if baselines:
    ax.hlines(y=baselines, xmin=0, xmax=8.5, color='#4c4c4c', linewidth=2.0, zorder=0)

plt.xlim(-0.5, 8.5)
sns.despine(top=True, right=True, left=False, bottom=False)

plt.title(f"Intra-Stage Continuous Variation (Fold {FOLD_TO_ANALYZE})", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("Inferred Position Score $\mu$", fontsize=16, fontweight='bold')
plt.ylabel("Clinical Stage", fontsize=16, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

ax.tick_params(axis='y', length=0)
plt.tight_layout()

plot_path = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', f'fold_{FOLD_TO_ANALYZE}', 'new_plots', f'exp3_v0_raincloud_fold_{FOLD_TO_ANALYZE}.png')
plt.savefig(plot_path, dpi=300)
print(f"Raincloud plot saved to: {plot_path}")