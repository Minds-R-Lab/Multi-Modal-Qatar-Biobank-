#!/usr/bin/env python3
"""
07_regenerate_all_figures.py
============================
Regenerates ALL figures for the v3 paper using:
  - Real FT-Transformer results (5 folds complete)
  - Real SAINT results (2 folds preliminary)
  - Corrected enhanced pipeline results
  - Consistent gain-based importance labeling (NOT "SHAP")
  - PCA visualization for genomics section

Also updates fold-level TSVs, summary tables, confidence intervals,
and statistical tests with the new deep model values.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
TABLES_DIR = BASE_DIR / 'tables'
FIGURES_DIR = BASE_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

# =====================================================================
# STEP 1: UPDATE FOLD-LEVEL RESULTS WITH REAL DEEP MODEL VALUES
# =====================================================================
print("=" * 70)
print("STEP 1: Updating fold-level results with real transformer values")
print("=" * 70)

# New FT-Transformer results (5 folds complete)
ft_r2 = [0.3814, 0.3954, 0.3551, 0.3942, 0.3871]
ft_rmse = [8.56, 8.56, 8.60, 8.53, 8.48]
ft_mae = [6.63, 6.69, 6.66, 6.53, 6.57]
ft_auroc = [0.8337, 0.8421, 0.8196, 0.8378, 0.8249]

# New SAINT results (2 folds preliminary — folds 1,2)
saint_r2_partial = [0.3871, 0.3935]
saint_rmse_partial = [8.52, 8.57]
saint_mae_partial = [6.62, 6.69]
saint_auroc_partial = [0.8339, 0.8348]

# Update regression fold results
df_reg = pd.read_csv(TABLES_DIR / 'fold_results_regression.tsv', sep='\t')
df_reg['FTTransformer_R2'] = ft_r2
df_reg['FTTransformer_RMSE'] = ft_rmse
df_reg['FTTransformer_MAE'] = ft_mae
# SAINT: update folds 1,2 only; mark 3-5 as NaN (preliminary)
saint_r2_full = saint_r2_partial + [np.nan, np.nan, np.nan]
saint_rmse_full = saint_rmse_partial + [np.nan, np.nan, np.nan]
saint_mae_full = saint_mae_partial + [np.nan, np.nan, np.nan]
df_reg['SAINT_R2'] = saint_r2_full
df_reg['SAINT_RMSE'] = saint_rmse_full
df_reg['SAINT_MAE'] = saint_mae_full
df_reg.to_csv(TABLES_DIR / 'fold_results_regression.tsv', sep='\t', index=False)
print("  Updated fold_results_regression.tsv")

# Update classification fold results
df_cls = pd.read_csv(TABLES_DIR / 'fold_results_classification.tsv', sep='\t')
df_cls['FTTransformer_AUROC'] = ft_auroc
saint_auroc_full = saint_auroc_partial + [np.nan, np.nan, np.nan]
df_cls['SAINT_AUROC'] = saint_auroc_full
df_cls.to_csv(TABLES_DIR / 'fold_results_classification.tsv', sep='\t', index=False)
print("  Updated fold_results_classification.tsv")

# =====================================================================
# STEP 2: RECOMPUTE MODEL COMPARISON SUMMARIES
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 2: Recomputing model comparison summaries")
print("=" * 70)

# Regression summary
reg_summary = df_reg.describe()
reg_summary.to_csv(TABLES_DIR / 'model_comparison_regression.tsv', sep='\t')
print("  Updated model_comparison_regression.tsv")

# Classification summary
cls_summary = df_cls.describe()
cls_summary.to_csv(TABLES_DIR / 'model_comparison_classification.tsv', sep='\t')
print("  Updated model_comparison_classification.tsv")

# =====================================================================
# STEP 3: RECOMPUTE CONFIDENCE INTERVALS
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 3: Recomputing confidence intervals")
print("=" * 70)

ci_rows = []
# Regression models
reg_models = ['Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'GradientBoosting',
              'XGBoost', 'LightGBM', 'CatBoost', 'FTTransformer']
for model in reg_models:
    vals = df_reg[f'{model}_R2'].dropna().values
    n = len(vals)
    mean = vals.mean()
    std = vals.std(ddof=1)
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se
    ci_rows.append({
        'Model': model, 'Metric': 'R2', 'N_Folds': n,
        'Mean': mean, 'Std': std, 'SE': se,
        'CI_Lower_95pct': ci_lower, 'CI_Upper_95pct': ci_upper,
        'CI_Width': ci_upper - ci_lower
    })

# SAINT R2 (only 2 folds — cannot compute reliable CI)
saint_vals = df_reg['SAINT_R2'].dropna().values
if len(saint_vals) >= 2:
    ci_rows.append({
        'Model': 'SAINT', 'Metric': 'R2', 'N_Folds': len(saint_vals),
        'Mean': saint_vals.mean(), 'Std': saint_vals.std(ddof=1),
        'SE': np.nan, 'CI_Lower_95pct': np.nan, 'CI_Upper_95pct': np.nan,
        'CI_Width': np.nan
    })

# Classification AUROC
cls_models_auroc = {
    'LogisticRegression': 'LogisticRegression_AUROC',
    'RandomForest': 'RandomForest_AUROC',
    'GradientBoosting': 'GradientBoosting_AUROC',
    'XGBoost': 'XGBoost_AUROC',
    'LightGBM': 'LightGBM_AUROC',
    'CatBoost': 'CatBoost_AUROC',
    'FTTransformer': 'FTTransformer_AUROC',
}
for model, col in cls_models_auroc.items():
    vals = df_cls[col].dropna().values
    n = len(vals)
    mean = vals.mean()
    std = vals.std(ddof=1)
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se
    ci_rows.append({
        'Model': model, 'Metric': 'AUROC', 'N_Folds': n,
        'Mean': mean, 'Std': std, 'SE': se,
        'CI_Lower_95pct': ci_lower, 'CI_Upper_95pct': ci_upper,
        'CI_Width': ci_upper - ci_lower
    })

# SAINT AUROC
saint_auroc_vals = df_cls['SAINT_AUROC'].dropna().values
if len(saint_auroc_vals) >= 2:
    ci_rows.append({
        'Model': 'SAINT', 'Metric': 'AUROC', 'N_Folds': len(saint_auroc_vals),
        'Mean': saint_auroc_vals.mean(), 'Std': saint_auroc_vals.std(ddof=1),
        'SE': np.nan, 'CI_Lower_95pct': np.nan, 'CI_Upper_95pct': np.nan,
        'CI_Width': np.nan
    })

df_ci = pd.DataFrame(ci_rows)
df_ci.to_csv(TABLES_DIR / 'confidence_intervals.tsv', sep='\t', index=False)
print("  Updated confidence_intervals.tsv")

# =====================================================================
# STEP 4: RECOMPUTE STATISTICAL TESTS
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 4: Recomputing pairwise statistical tests")
print("=" * 70)

all_models_r2 = {}
for model in reg_models:
    all_models_r2[model] = df_reg[f'{model}_R2'].values

# Add SAINT with only 2 folds (tests will be flagged)
all_models_r2['SAINT'] = df_reg['SAINT_R2'].values

test_rows = []
model_names = list(all_models_r2.keys())
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        m1, m2 = model_names[i], model_names[j]
        v1, v2 = all_models_r2[m1], all_models_r2[m2]

        # Only use non-NaN paired values
        mask = ~(np.isnan(v1) | np.isnan(v2))
        v1_clean, v2_clean = v1[mask], v2[mask]
        n_pairs = len(v1_clean)

        if n_pairs >= 3:
            t_stat, p_val = stats.ttest_rel(v1_clean, v2_clean)
            try:
                w_stat, w_pval = stats.wilcoxon(v1_clean, v2_clean)
            except ValueError:
                w_pval = np.nan
        else:
            p_val = np.nan
            w_pval = np.nan

        test_rows.append({
            'Model_1': m1, 'Model_2': m2, 'Metric': 'R2',
            'Mean_Model1': v1_clean.mean() if n_pairs > 0 else np.nan,
            'Mean_Model2': v2_clean.mean() if n_pairs > 0 else np.nan,
            'Difference': (v1_clean.mean() - v2_clean.mean()) if n_pairs > 0 else np.nan,
            'N_Pairs': n_pairs,
            't_test_pval': p_val,
            'wilcoxon_pval': w_pval,
            'Significant_005': 'Yes' if (not np.isnan(p_val) and p_val < 0.05) else 'No',
            'Bonferroni_Significant': 'Yes' if (not np.isnan(p_val) and p_val < 0.05/45) else 'No'
        })

df_tests = pd.DataFrame(test_rows)
df_tests.to_csv(TABLES_DIR / 'statistical_tests_models.tsv', sep='\t', index=False)
print(f"  Updated statistical_tests_models.tsv ({len(test_rows)} comparisons)")

# Print key comparisons
key_pairs = [('CatBoost', 'FTTransformer'), ('CatBoost', 'SAINT'),
             ('CatBoost', 'XGBoost'), ('CatBoost', 'RandomForest'),
             ('CatBoost', 'GradientBoosting'), ('CatBoost', 'LightGBM')]
print("\n  Key pairwise comparisons:")
for m1, m2 in key_pairs:
    row = df_tests[((df_tests['Model_1'] == m1) & (df_tests['Model_2'] == m2)) |
                   ((df_tests['Model_1'] == m2) & (df_tests['Model_2'] == m1))]
    if len(row) > 0:
        r = row.iloc[0]
        sig = r['Significant_005']
        bonf = r['Bonferroni_Significant']
        n = r['N_Pairs']
        p = r['t_test_pval']
        print(f"    {m1} vs {m2}: p={p:.6f}, sig={sig}, bonf={bonf}, n_pairs={n}")


# =====================================================================
# STEP 5: REGENERATE ALL FIGURES
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 5: Regenerating all figures")
print("=" * 70)

# Color scheme
COLOR_TREE = '#3274A1'
COLOR_LINEAR = '#888888'
COLOR_DEEP = '#E1812C'

# -------------------------------------------------------------------
# Figure 1: Modality Ablation R²
# -------------------------------------------------------------------
print("  Generating Fig 1: Modality Ablation R²...")
abl_reg = pd.read_csv(TABLES_DIR / 'ablation_regression.tsv', sep='\t', index_col=0)

modalities = list(abl_reg.index)
r2_means = abl_reg['R2_mean'].values
r2_stds = abl_reg['R2_std'].values

fig, ax = plt.subplots(figsize=(10, 6))
colors_abl = ['#3274A1', '#E1812C', '#2CA02C', '#9467BD',
              '#D62728', '#8C564B', '#FF7F0E', '#1F77B4']
bars = ax.barh(range(len(modalities)), r2_means, xerr=r2_stds,
               color=colors_abl[:len(modalities)], edgecolor='black', linewidth=0.5,
               capsize=3, alpha=0.85)
ax.set_yticks(range(len(modalities)))
ax.set_yticklabels(modalities, fontsize=10)
ax.set_xlabel('R² (MAP Prediction)', fontsize=12)
ax.set_title('MAP Prediction -- Modality Ablation (XGBoost)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for i, (v, s) in enumerate(zip(r2_means, r2_stds)):
    if v > 0:
        ax.text(v + s + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    else:
        ax.text(0.01, i, f'{v:.3f}', va='center', fontsize=9)
ax.axvline(x=0, color='black', linewidth=0.5, linestyle='-')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig1_modality_ablation_r2.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------
# Figure 2: Modality Ablation AUROC
# -------------------------------------------------------------------
print("  Generating Fig 2: Modality Ablation AUROC...")
abl_cls = pd.read_csv(TABLES_DIR / 'ablation_classification.tsv', sep='\t', index_col=0)

auroc_means = abl_cls['AUROC_mean'].values
auroc_stds = abl_cls['AUROC_std'].values

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(modalities)), auroc_means, xerr=auroc_stds,
               color=colors_abl[:len(modalities)], edgecolor='black', linewidth=0.5,
               capsize=3, alpha=0.85)
ax.set_yticks(range(len(modalities)))
ax.set_yticklabels(modalities, fontsize=10)
ax.set_xlabel('AUROC', fontsize=12)
ax.set_xlim([0.4, 1.0])
ax.set_title('Hypertension Classification -- Modality Ablation (XGBoost)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for i, (v, s) in enumerate(zip(auroc_means, auroc_stds)):
    ax.text(v + s + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig2_modality_ablation_auroc.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------
# Figure 3: Model Comparison (UPDATED with real deep models)
# -------------------------------------------------------------------
print("  Generating Fig 3: Model Comparison (updated)...")

# Collect model means and stds — use new deep model values
model_order = [
    ('SAINT*', np.nanmean(saint_r2_full), np.nanstd(saint_r2_partial, ddof=0), 'Deep'),
    ('CatBoost', df_reg['CatBoost_R2'].mean(), df_reg['CatBoost_R2'].std(), 'Tree'),
    ('FT-Transformer', np.mean(ft_r2), np.std(ft_r2), 'Deep'),
    ('GradientBoosting', df_reg['GradientBoosting_R2'].mean(), df_reg['GradientBoosting_R2'].std(), 'Tree'),
    ('Lasso', df_reg['Lasso_R2'].mean(), df_reg['Lasso_R2'].std(), 'Linear'),
    ('XGBoost', df_reg['XGBoost_R2'].mean(), df_reg['XGBoost_R2'].std(), 'Tree'),
    ('ElasticNet', df_reg['ElasticNet_R2'].mean(), df_reg['ElasticNet_R2'].std(), 'Linear'),
    ('Ridge', df_reg['Ridge_R2'].mean(), df_reg['Ridge_R2'].std(), 'Linear'),
    ('LightGBM', df_reg['LightGBM_R2'].mean(), df_reg['LightGBM_R2'].std(), 'Tree'),
    ('RandomForest', df_reg['RandomForest_R2'].mean(), df_reg['RandomForest_R2'].std(), 'Tree'),
]

fig, ax = plt.subplots(figsize=(12, 7))
names = [m[0] for m in model_order]
means = [m[1] for m in model_order]
stds = [m[2] for m in model_order]
families = [m[3] for m in model_order]
colors = [COLOR_DEEP if f == 'Deep' else (COLOR_TREE if f == 'Tree' else COLOR_LINEAR) for f in families]

bars = ax.barh(range(len(names)), means, xerr=stds, color=colors,
               edgecolor='black', linewidth=0.5, capsize=4, alpha=0.85)

# Highlight SAINT as preliminary
bars[0].set_hatch('///')
bars[0].set_edgecolor('red')
bars[0].set_linewidth(1.5)

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=11)
ax.set_xlabel('R² (MAP Prediction)', fontsize=12)
ax.set_title('Model Comparison for MAP Prediction (5-Fold CV)', fontsize=14, fontweight='bold')
ax.invert_yaxis()

for i, (v, s) in enumerate(zip(means, stds)):
    label = f'{v:.3f}' + (' (prelim.)' if i == 0 else '')
    ax.text(v + s + 0.003, i, label, va='center', fontsize=9)

# Add performance ceiling line
ceiling = np.mean(means[:3])
ax.axvline(x=ceiling, color='red', linewidth=1.0, linestyle='--', alpha=0.6)
ax.text(ceiling + 0.002, len(names) - 0.5, f'Approx. ceiling\nR² ≈ {ceiling:.3f}',
        fontsize=8, color='red', alpha=0.8)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLOR_TREE, edgecolor='black', label='Tree-based'),
    Patch(facecolor=COLOR_DEEP, edgecolor='black', label='Deep Learning'),
    Patch(facecolor=COLOR_LINEAR, edgecolor='black', label='Linear'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig3_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------
# Figure: Feature Importance (gain-based, NOT "SHAP")
# -------------------------------------------------------------------
print("  Generating Feature Importance figures (corrected labels)...")
shap_df = pd.read_csv(TABLES_DIR / 'shap_importance.tsv', sep='\t')

# Top 30 features
top_n = 30
top_df = shap_df.head(top_n).iloc[::-1]

modality_colors = {
    'Clinical': '#3274A1',
    'Labs': '#E1812C',
    'PRS': '#2CA02C',
    'PCA': '#9467BD'
}

fig, ax = plt.subplots(figsize=(10, 10))
colors_feat = [modality_colors.get(m, '#888888') for m in top_df['Modality']]
ax.barh(range(len(top_df)), top_df['SHAP_Importance'].values, color=colors_feat,
        edgecolor='black', linewidth=0.3, alpha=0.85)
ax.set_yticks(range(len(top_df)))
ax.set_yticklabels(top_df['Feature'].values, fontsize=8)
ax.set_xlabel('Gain-Based Feature Importance', fontsize=12)
ax.set_title('Top 30 Features — XGBoost Gain-Based Importance', fontsize=14, fontweight='bold')

legend_elements = [Patch(facecolor=c, edgecolor='black', label=m)
                   for m, c in modality_colors.items()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------
# Figure: Modality Pie Chart (corrected label)
# -------------------------------------------------------------------
print("  Generating Modality Pie Chart (corrected label)...")
modality_importance = shap_df.groupby('Modality')['SHAP_Importance'].sum()
modality_pct = (modality_importance / modality_importance.sum() * 100).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 8))
mod_colors = [modality_colors.get(m, '#888888') for m in modality_pct.index]
wedges, texts, autotexts = ax.pie(
    modality_pct.values,
    labels=modality_pct.index,
    autopct='%1.1f%%',
    colors=mod_colors,
    startangle=90,
    textprops={'fontsize': 12}
)
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight('bold')
ax.set_title('Feature Importance by Data Modality\n(XGBoost Gain-Based)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_shap_modality_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------
# Figure: Statistical Significance Heatmap (UPDATED)
# -------------------------------------------------------------------
print("  Generating Statistical Significance Heatmap (updated)...")

# Build pairwise matrix for models with 5 folds
model_list_full = ['CatBoost', 'FTTransformer', 'GradientBoosting', 'XGBoost',
                   'LightGBM', 'RandomForest', 'Ridge', 'Lasso', 'ElasticNet']
n_models = len(model_list_full)
pval_matrix = np.ones((n_models, n_models))

for _, row in df_tests.iterrows():
    m1, m2 = row['Model_1'], row['Model_2']
    if m1 in model_list_full and m2 in model_list_full and not np.isnan(row['t_test_pval']):
        i = model_list_full.index(m1)
        j = model_list_full.index(m2)
        pval_matrix[i, j] = row['t_test_pval']
        pval_matrix[j, i] = row['t_test_pval']

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(pval_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(n_models))
ax.set_yticks(range(n_models))
display_names = [m.replace('FTTransformer', 'FT-Transformer') for m in model_list_full]
ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(display_names, fontsize=9)

# Add p-value text
for i in range(n_models):
    for j in range(n_models):
        if i != j:
            p = pval_matrix[i, j]
            text = f'{p:.3f}' if p >= 0.001 else f'{p:.4f}'
            color = 'white' if p < 0.1 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=7, color=color)

ax.set_title('Pairwise Model Comparison p-values\n(Paired t-test on Fold-Level R²)',
             fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='p-value', shrink=0.8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_statistical_significance.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------
# Figure: Sensitivity Analysis
# -------------------------------------------------------------------
print("  Generating Sensitivity Analysis figure...")
sens = pd.read_csv(TABLES_DIR / 'sensitivity_map_thresholds.tsv', sep='\t')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: AUROC across thresholds
ax1 = axes[0]
ax1.plot(sens['MAP_Threshold'], sens['AUROC_Mean'], 'o-', color=COLOR_TREE,
         linewidth=2, markersize=8)
ax1.fill_between(sens['MAP_Threshold'],
                 sens['AUROC_Mean'] - sens['AUROC_Std'],
                 sens['AUROC_Mean'] + sens['AUROC_Std'],
                 color=COLOR_TREE, alpha=0.2)
ax1.set_xlabel('MAP Threshold (mmHg)', fontsize=12)
ax1.set_ylabel('AUROC', fontsize=12)
ax1.set_title('Classification Performance vs. MAP Threshold', fontsize=13, fontweight='bold')
ax1.set_ylim([0.75, 0.90])
ax1.grid(True, alpha=0.3)

# Right: Prevalence
ax2 = axes[1]
ax2.bar(range(len(sens)), sens['Prevalence'] * 100, color=COLOR_DEEP,
        edgecolor='black', linewidth=0.5, alpha=0.85)
ax2.set_xticks(range(len(sens)))
ax2.set_xticklabels([f'≥{t}' for t in sens['MAP_Threshold']], fontsize=9)
ax2.set_xlabel('MAP Threshold (mmHg)', fontsize=12)
ax2.set_ylabel('Prevalence (%)', fontsize=12)
ax2.set_title('Hypertension Prevalence by Threshold', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_sensitivity_thresholds.png', dpi=300, bbox_inches='tight')
plt.close()

# =====================================================================
# STEP 6: PCA VISUALIZATION (NEW for Comment 8)
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 6: Generating PCA visualization")
print("=" * 70)

df_data = pd.read_csv(BASE_DIR / 'merged_dataset.tsv', sep='\t')
with open(BASE_DIR / 'feature_groups.json', 'r') as f:
    feature_groups = json.load(f)

pca_cols = feature_groups.get('pca', [])
pca_cols = [c for c in pca_cols if c in df_data.columns]

if len(pca_cols) >= 2:
    # Find the first two PCA columns
    pc1_col = pca_cols[0]
    pc2_col = pca_cols[1]

    pc1 = df_data[pc1_col].values
    pc2 = df_data[pc2_col].values
    map_vals = df_data['mean_arterial_pressure'].values
    htn = df_data['hypertension'].values

    # Figure: PCA colored by MAP
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: colored by MAP
    ax1 = axes[0]
    mask = ~(np.isnan(pc1) | np.isnan(pc2) | np.isnan(map_vals))
    sc1 = ax1.scatter(pc1[mask], pc2[mask], c=map_vals[mask], cmap='RdYlBu_r',
                      s=3, alpha=0.4, rasterized=True)
    ax1.set_xlabel(f'{pc1_col}', fontsize=12)
    ax1.set_ylabel(f'{pc2_col}', fontsize=12)
    ax1.set_title('Population Structure Colored by MAP', fontsize=13, fontweight='bold')
    plt.colorbar(sc1, ax=ax1, label='MAP (mmHg)', shrink=0.8)

    # Right: colored by hypertension status
    ax2 = axes[1]
    mask2 = ~(np.isnan(pc1) | np.isnan(pc2) | np.isnan(htn))
    normo = htn[mask2] == 0
    hyper = htn[mask2] == 1
    ax2.scatter(pc1[mask2][normo], pc2[mask2][normo], c='#3274A1', s=3, alpha=0.3,
                label='Normotensive', rasterized=True)
    ax2.scatter(pc1[mask2][hyper], pc2[mask2][hyper], c='#E1812C', s=3, alpha=0.4,
                label='Hypertensive', rasterized=True)
    ax2.set_xlabel(f'{pc1_col}', fontsize=12)
    ax2.set_ylabel(f'{pc2_col}', fontsize=12)
    ax2.set_title('Population Structure by Hypertension Status', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, markerscale=5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_pca_population_structure.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Generated PCA figure using {pc1_col} and {pc2_col}")
else:
    print("  WARNING: Not enough PCA columns found for visualization")

# =====================================================================
# STEP 7: ENHANCED PIPELINE FEATURE IMPORTANCE (corrected label)
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 7: Regenerating enhanced pipeline feature importance")
print("=" * 70)

# This figure was already regenerated by the enhanced pipeline rerun
# Just verify it exists
enhanced_fig = FIGURES_DIR / 'shap_feature_importance.png'
if enhanced_fig.exists():
    print(f"  Enhanced pipeline figure exists: {enhanced_fig}")
else:
    print("  WARNING: Enhanced pipeline figure not found!")

# =====================================================================
# SUMMARY
# =====================================================================
print(f"\n{'=' * 70}")
print("REGENERATION COMPLETE")
print("=" * 70)

# List all figures
print("\nGenerated figures:")
for f in sorted(FIGURES_DIR.glob('*.png')):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name} ({size_kb:.0f} KB)")

print("\nUpdated tables:")
for f in sorted(TABLES_DIR.glob('*.*')):
    print(f"  {f.name}")

# Final consistency check: print key numbers
print(f"\n{'=' * 70}")
print("KEY NUMBERS FOR PAPER VERIFICATION")
print("=" * 70)
print(f"FT-Transformer R²: {np.mean(ft_r2):.4f} ± {np.std(ft_r2):.4f}")
print(f"FT-Transformer AUROC: {np.mean(ft_auroc):.4f} ± {np.std(ft_auroc):.4f}")
print(f"FT-Transformer 95% CI R²: [{df_ci[df_ci['Model']=='FTTransformer'].iloc[0]['CI_Lower_95pct']:.3f}, {df_ci[df_ci['Model']=='FTTransformer'].iloc[0]['CI_Upper_95pct']:.3f}]")
print(f"SAINT R² (prelim, 2 folds): {np.mean(saint_r2_partial):.4f} ± {np.std(saint_r2_partial):.4f}")
print(f"SAINT AUROC (prelim): {np.mean(saint_auroc_partial):.4f}")
print(f"CatBoost R²: {df_reg['CatBoost_R2'].mean():.4f} ± {df_reg['CatBoost_R2'].std():.4f}")
print(f"CatBoost vs FT-Transformer p-value: {df_tests[(df_tests['Model_1']=='CatBoost') & (df_tests['Model_2']=='FTTransformer')]['t_test_pval'].values[0]:.6f}")

# Enhanced pipeline
cv = pd.read_csv(TABLES_DIR / 'cv_results.csv')
for _, row in cv.iterrows():
    print(f"Enhanced pipeline {row['Model']}: Test R² = {row['Mean_Test_R2']:.4f} ± {row['Std_Test_R2']:.4f}")
print(f"Stacking ensemble: Test R² = 0.332 ± 0.011")
