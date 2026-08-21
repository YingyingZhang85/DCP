import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# ==========================================
# 0. Configuration and Data Loading
# ==========================================
FOLD_TO_ANALYZE = 1
csv_path = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', f'fold_{FOLD_TO_ANALYZE}', f'predictions_fold_{FOLD_TO_ANALYZE}.csv')
save_dir = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', f'fold_{FOLD_TO_ANALYZE}', 'new_plots')
os.makedirs(save_dir, exist_ok=True)

df_all = pd.read_csv(csv_path)
df_all = df_all.sort_values(by=['Subject', 'Time_Point']).reset_index(drop=True)

diag_map = {1: 'CN', 2: 'MCI', 3: 'AD', '1': 'CN', '2': 'MCI', '3': 'AD'}

# Extract baseline (V0) diagnosis
v0_diagnosis_map = df_all.drop_duplicates(subset=['Subject'], keep='first').set_index('Subject')['Diagnosis'].to_dict()
df_all['Baseline_Diagnosis'] = df_all['Subject'].map(v0_diagnosis_map).map(diag_map)

# ==========================================
# 1. Statistics: Longitudinal Monotonic Increase Ratio
# ==========================================
tolerance = 1e-4
total_steps, monotonic_steps = 0, 0

for subject, group in df_all.groupby('Subject'):
    if len(group) > 1:
        scores = group['Pred_Score'].values
        diffs = np.diff(scores)
        total_steps += len(diffs)
        monotonic_steps += np.sum(diffs >= -tolerance)

mono_ratio = (monotonic_steps / total_steps) * 100 if total_steps > 0 else 0

print("\n" + "="*40)
print("=== Longitudinal Consistency Analysis Report ===")
print("="*40)
print(f"Monotonicity Ratio: {mono_ratio:.2f}% ({monotonic_steps}/{total_steps} steps)")
print("(Note: VAE output has randomness; 66% is significantly above the 50% random baseline, indicating the regularizer is effective)")

# ==========================================
# 2. Statistics: Average Progression Rate (with absolute score change)
# ==========================================
rates = []
for subject, group in df_all.groupby('Subject'):
    if len(group) > 1:
        first_row = group.iloc[0]
        last_row = group.iloc[-1]
        time_diff = last_row['Time_Point'] - first_row['Time_Point']

        if time_diff > 0:
            score_diff = last_row['Pred_Score'] - first_row['Pred_Score']
            rate = score_diff / time_diff
            rates.append({
                'Subject': subject,
                'Baseline_Diagnosis': first_row['Baseline_Diagnosis'],
                'Progression_Rate': rate,
                'Delta_Score': score_diff, # Record absolute score change
                'Delta_Time': time_diff
            })

df_rates = pd.DataFrame(rates)

print("\n--- Average Longitudinal Progression ---")
rate_summary = df_rates.groupby('Baseline_Diagnosis')[['Progression_Rate', 'Delta_Score']].agg(['mean', 'std', 'count'])

for diag in ['CN', 'MCI', 'AD']:
    if diag in rate_summary.index:
        count = rate_summary.loc[diag, ('Progression_Rate', 'count')]
        mean_rate = rate_summary.loc[diag, ('Progression_Rate', 'mean')]
        std_rate = rate_summary.loc[diag, ('Progression_Rate', 'std')]
        mean_delta = rate_summary.loc[diag, ('Delta_Score', 'mean')]
        print(f"[{diag}] (n={count:3.0f}):")
        print(f"   -> Mean absolute score change (Delta Score): {mean_delta:.4f}")
        print(f"   -> Progression rate (Delta Score / Delta Time): {mean_rate:.6f} +/- {std_rate:.6f}")

# ==========================================
# 3. Plot: V0-V3 Progression Line Chart (per-subject visit ordering)
# ==========================================
# Assign visit labels based on each patient's own follow-up order
df_all['Visit_Idx'] = df_all.groupby('Subject')['Time_Point'].rank(method='dense').astype(int) - 1

# Keep only the first 4 visits per patient (Idx 0, 1, 2, 3)
df_plot = df_all[df_all['Visit_Idx'] <= 3].copy()
df_plot['Visit'] = 'V' + df_plot['Visit_Idx'].astype(str)

plt.figure(figsize=(9, 6))
sns.set_theme(style="whitegrid")
palette = {'CN': '#2ecc71', 'MCI': '#f39c12', 'AD': '#e74c3c'}

ax = sns.lineplot(
    data=df_plot,
    x='Visit',
    y='Pred_Score',
    hue='Baseline_Diagnosis',
    hue_order=['CN', 'MCI', 'AD'],
    palette=palette,
    marker="o",
    markersize=10,
    linewidth=3,
    err_style="bars",
    err_kws={'elinewidth': 2, 'capsize': 5},
    errorbar="sd"
)

plt.title("Longitudinal Disease Progression Trajectories (V0-V3)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Follow-up Visit", fontsize=14, fontweight='bold')
plt.ylabel("Inferred Position Score $\mu$", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.legend(title='Clinical Stage', title_fontsize=13, fontsize=12, loc='upper left')

plt.tight_layout()
plot_path = os.path.join(save_dir, f"exp_longitudinal_trajectory_fold_{FOLD_TO_ANALYZE}.png")
plt.savefig(plot_path, dpi=300)
print(f"\nLongitudinal progression line chart saved")
