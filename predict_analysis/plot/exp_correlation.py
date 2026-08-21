import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.ticker as ticker
import numpy as np
import matplotlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def set_times():
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 0. Configuration and Data Loading (predictions + clinical metadata)
# ==========================================
FOLD_TO_ANALYZE = 2

pred_csv_path = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', f'fold_{FOLD_TO_ANALYZE}', f'predictions_fold_{FOLD_TO_ANALYZE}.csv')

meta_csv_path = os.path.join(PROJECT_ROOT, 'datasets', 'csv', '5_folds', 'dataset_5_folds_DTI.csv')

save_dir = os.path.join(PROJECT_ROOT, 'outputs', 'experimental_results', 'predictions', f'fold_{FOLD_TO_ANALYZE}', 'new_plots')
os.makedirs(save_dir, exist_ok=True)

print(f"Reading and merging data for Fold {FOLD_TO_ANALYZE}...")

df_pred = pd.read_csv(pred_csv_path)
df_meta = pd.read_csv(meta_csv_path)

# Standardize column names for merging
if 'Subject (PTID)' in df_meta.columns:
    df_meta = df_meta.rename(columns={'Subject (PTID)': 'Subject'})

# Left join clinical data to predictions on Subject and Acq Date
df_merged = pd.merge(df_pred, df_meta, on=['Subject', 'Acq Date'], how='left')

# Map clinical labels
diag_map = {1: 'CN', 2: 'MCI', 3: 'AD', '1': 'CN', '2': 'MCI', '3': 'AD'}
df_merged['Diagnosis'] = df_merged['Diagnosis'].map(diag_map)

print(f"Data merge complete. Total records: {len(df_merged)}")

# ==========================================
# General plotting function definition
# ==========================================
def plot_scatter_correlation(df, y_col, y_label, save_dir, prefix, text_loc, legend_loc):
    if y_col not in df.columns:
        print(f"Warning: column '{y_col}' not found, skipping {prefix.upper()} chart.")
        return

    # Remove rows containing NaN
    df_plot = df.dropna(subset=['Pred_Score', y_col]).copy()
    if len(df_plot) == 0: return

    plt.figure(figsize=(8, 6), dpi=300)
    sns.set_theme(style="whitegrid", font_scale=1.2)
    set_times()  # Set Times New Roman font and math font

    # Scatter plot
    ax = sns.scatterplot(
        x="Pred_Score", y=y_col, hue="Diagnosis", data=df_plot,
        hue_order=['CN', 'MCI', 'AD'],
        palette={'CN': '#2ecc71', 'MCI': '#f39c12', 'AD': '#e74c3c'},
        s=60, alpha=0.7, edgecolor="w"
    )

    # Regression line
    sns.regplot(
        x="Pred_Score", y=y_col, data=df_plot,
        scatter=False, color="gray", line_kws={"linewidth": 2.5, "linestyle": "--"},
        ax=ax
    )

    # X-axis: range 0-15, interval 2.5
    ax.set_xlim(0, 15)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2.5))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    for label in ax.get_xticklabels():
        label.set_fontweight('bold'); label.set_fontsize(14); label.set_color('black')

    # Tick marks and clean grid
    ax.tick_params(axis='x', which='major', length=5, width=1.2, color='black')
    ax.grid(True, which='major', linestyle='-', linewidth=0.8, color='#e0e0e0', alpha=0.8)

    # Fine-tune Y-axis
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=10))
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(14)

    # Compute correlation with both metrics
    spearman_corr, spearman_p = stats.spearmanr(df_plot['Pred_Score'], df_plot[y_col])
    pearson_corr, pearson_p = stats.pearsonr(df_plot['Pred_Score'], df_plot[y_col])

    print(f"\n[MERGED - {prefix.upper()} vs Inferred Score]")
    print(f"  -> Pearson  (linear) : r = {pearson_corr:.4f}, p-value = {pearson_p:.4e}")
    print(f"  -> Spearman (rank)   : rho = {spearman_corr:.4f}, p-value = {spearman_p:.4e}")

    # Text box
    textstr = (f"Pearson r = {pearson_corr:.3f} (p = {pearson_p:.2e})\n"
               f"Spearman $\\rho$ = {spearman_corr:.3f} (p = {spearman_p:.2e})")

    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
    text_obj = ax.text(text_loc[0], text_loc[1], textstr, transform=ax.transAxes, fontsize=13,
                       verticalalignment='top', bbox=props)
    text_obj.set_fontweight('bold')

    plt.xlabel("DCS", fontweight='bold', fontsize=16)
    plt.ylabel(y_label, fontweight='bold', fontsize=16)
    ax.set_title("")                                            # Remove top title

    # Legend: remove "Clinical Stage" title, add frame
    plt.legend(title=None, loc=legend_loc, frameon=True, fontsize=12)
    plt.setp(ax.get_legend().get_texts(), fontweight='bold')

    sns.despine()

    out_pdf = os.path.join(save_dir, f'exp_corr_{prefix}_fold_{FOLD_TO_ANALYZE}.pdf')
    plt.tight_layout()
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Merged {prefix.upper()} scatter plot saved to: {out_pdf}")


print("\n" + "="*40)
print(f"=== Generating clinical correlation charts for Fold {FOLD_TO_ANALYZE} ===")
print("="*40)

# ==========================================
# 1. Plot MMSE (negative correlation, text upper-right, legend lower-left)
# ==========================================
# Auto-compatible with column name MMSE or MMSE_Score
plot_scatter_correlation(
    df = df_merged,
    y_col = 'MMSCORE',            # Specify column name as 'MMSCORE'
    y_label = "MMSE",
    save_dir = save_dir,
    prefix = 'mmse',
    text_loc = (0.65, 0.95),      # Upper-left is high-score healthy region, avoid overlap
    legend_loc = 'lower left'
)

# ==========================================
# 2. Plot CDR-SB (positive correlation, text upper-left, legend lower-right)
# ==========================================
plot_scatter_correlation(
    df = df_merged,
    y_col = 'CDRSB',
    y_label = "CDR-SB",
    save_dir = save_dir,
    prefix = 'cdr',
    text_loc = (0.05, 0.95),
    legend_loc = 'lower right'
)

# ==========================================
# 3. Plot ADAS11 (positive correlation, text upper-left, legend lower-right)
# ==========================================
# plot_scatter_correlation(
#     df = df_merged,
#     y_col = 'ADAS11',
#     y_label = "ADAS-Cog 11 Score",
#     save_dir = save_dir,
#     prefix = 'adas11',
#     text_loc = (0.05, 0.95),
#     legend_loc = 'lower right'
# )

# ==========================================
# 4. Plot ADAS13 (positive correlation, text upper-left, legend lower-right)
# ==========================================
plot_scatter_correlation(
    df = df_merged,
    y_col = 'ADAS13',
    y_label = "ADAS-Cog",
    save_dir = save_dir,
    prefix = 'adas13',
    text_loc = (0.05, 0.95),
    legend_loc = 'lower right'
)

# 5. GDS (higher score indicates more severe depression symptoms)
plot_scatter_correlation(
    df = df_merged,
    y_col = 'GDTOTAL',            # Ensure this matches the actual column name in your dataset
    y_label = "GDS",
    save_dir = save_dir,
    prefix = 'gds',               # Output filename will be exp_corr_gds_fold_x.pdf
    text_loc = (0.05, 0.95),      # Avoid upper-right high score area
    legend_loc = 'lower right'
)

# 6. FAQ (higher score indicates more severity)
plot_scatter_correlation(
    df = df_merged,
    y_col = 'FAQTOTAL',
    y_label = "FAQ",
    save_dir = save_dir,
    prefix = 'faq',
    text_loc = (0.05, 0.95),
    legend_loc = 'lower right'
)

# 7. NPI (higher score indicates more severity)
plot_scatter_correlation(
    df = df_merged,
    y_col = 'NPITOTAL',
    y_label = "NPI",
    save_dir = save_dir,
    prefix = 'npi',
    text_loc = (0.05, 0.95),
    legend_loc = 'lower right'
)

print("\nCorrelation analysis complete.")
