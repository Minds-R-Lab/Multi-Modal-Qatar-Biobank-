#!/usr/bin/env python3
"""
Statistical significance testing for ML model results.

Performs pairwise model comparisons, modality ablation analysis, bootstrap CI estimation,
and generates visualization of statistical significance.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Setup
OUT = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(OUT, 'tables')
FIGURES_DIR = os.path.join(OUT, 'figures')
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# File paths (input from script 02)
regression_results_file = os.path.join(TABLES_DIR, 'fold_results_regression.tsv')
classification_results_file = os.path.join(TABLES_DIR, 'fold_results_classification.tsv')
ablation_results_file = os.path.join(TABLES_DIR, 'fold_results_ablation.tsv')

# Output files
output_models_file = os.path.join(TABLES_DIR, 'statistical_tests_models.tsv')
output_ablation_file = os.path.join(TABLES_DIR, 'statistical_tests_ablation.tsv')
output_ci_file = os.path.join(TABLES_DIR, 'confidence_intervals.tsv')
output_figure_file = os.path.join(FIGURES_DIR, 'fig_statistical_significance.png')


def wide_to_long(df, metric_suffix):
    """Convert wide-format fold results to long format.

    Input:  fold | Ridge_R2 | Lasso_R2 | ...
    Output: Model | Fold | value
    """
    metric_cols = [c for c in df.columns if c.endswith(f'_{metric_suffix}')]
    if not metric_cols:
        return None

    records = []
    for _, row in df.iterrows():
        fold_num = row['fold']
        for col in metric_cols:
            model_name = col.replace(f'_{metric_suffix}', '')
            records.append({
                'Model': model_name,
                'Fold': fold_num,
                metric_suffix: row[col]
            })
    return pd.DataFrame(records)


def load_results():
    """Load fold-level results from pipeline and convert to long format."""
    reg_long = None
    cls_long = None
    ablation_df = None

    if os.path.exists(regression_results_file):
        reg_wide = pd.read_csv(regression_results_file, sep='\t')
        print(f"Loaded regression results: {reg_wide.shape}")
        # Convert wide to long for R2
        reg_long = wide_to_long(reg_wide, 'R2')
        if reg_long is not None:
            # Also add RMSE and MAE
            rmse_long = wide_to_long(reg_wide, 'RMSE')
            mae_long = wide_to_long(reg_wide, 'MAE')
            if rmse_long is not None:
                reg_long = reg_long.merge(rmse_long, on=['Model', 'Fold'], how='left')
            if mae_long is not None:
                reg_long = reg_long.merge(mae_long, on=['Model', 'Fold'], how='left')
            print(f"  Converted to long format: {reg_long.shape}, models: {reg_long['Model'].unique().tolist()}")
    else:
        print(f"Warning: {regression_results_file} not found")

    if os.path.exists(classification_results_file):
        cls_wide = pd.read_csv(classification_results_file, sep='\t')
        print(f"Loaded classification results: {cls_wide.shape}")
        cls_long = wide_to_long(cls_wide, 'AUROC')
        if cls_long is not None:
            print(f"  Converted to long format: {cls_long.shape}, models: {cls_long['Model'].unique().tolist()}")
    else:
        print(f"Warning: {classification_results_file} not found")

    if os.path.exists(ablation_results_file):
        ablation_df = pd.read_csv(ablation_results_file, sep='\t')
        print(f"Loaded ablation results: {ablation_df.shape}")
    else:
        print(f"Warning: {ablation_results_file} not found")

    return reg_long, cls_long, ablation_df


def test_pairwise_model_comparison(results_df, metric='R2'):
    """Perform pairwise model comparisons using paired t-test and Wilcoxon test."""
    if results_df is None or metric not in results_df.columns:
        print(f"No results to compare for metric {metric}")
        return None, None, None

    models = sorted(results_df['Model'].unique())
    n_models = len(models)

    comparison_results = []
    pvalue_matrix = np.ones((n_models, n_models))

    print(f"\n--- Pairwise Model Comparison ({metric}) ---")
    print(f"Models: {models}")
    print(f"Number of folds: {results_df['Fold'].nunique()}")

    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i >= j:
                continue

            vals1 = results_df[results_df['Model'] == model1][metric].dropna().values
            vals2 = results_df[results_df['Model'] == model2][metric].dropna().values

            if len(vals1) < 2 or len(vals2) < 2 or len(vals1) != len(vals2):
                continue

            # Paired t-test
            try:
                t_stat, t_pval = stats.ttest_rel(vals1, vals2)
            except Exception:
                t_pval = np.nan

            # Wilcoxon signed-rank test
            w_pval = np.nan
            if len(vals1) >= 5:
                try:
                    w_stat, w_pval = stats.wilcoxon(vals1 - vals2)
                except Exception:
                    pass

            pvalue_matrix[i, j] = t_pval
            pvalue_matrix[j, i] = t_pval

            comparison_results.append({
                'Model_1': model1,
                'Model_2': model2,
                'Metric': metric,
                'Mean_Model1': np.mean(vals1),
                'Mean_Model2': np.mean(vals2),
                'Difference': np.mean(vals1) - np.mean(vals2),
                't_test_pval': t_pval,
                'wilcoxon_pval': w_pval,
                'Significant_005': 'Yes' if t_pval < 0.05 else 'No'
            })

    # Bonferroni correction
    n_comp = len(comparison_results)
    bonf_alpha = 0.05 / n_comp if n_comp > 0 else 0.05
    for row in comparison_results:
        row['Bonferroni_Significant'] = 'Yes' if row['t_test_pval'] < bonf_alpha else 'No'

    results_out = pd.DataFrame(comparison_results) if comparison_results else None
    print(f"Completed {len(comparison_results)} comparisons, Bonferroni alpha: {bonf_alpha:.6f}")

    return results_out, pvalue_matrix, models


def test_modality_ablation(ablation_df):
    """Test modality ablation significance using paired t-tests."""
    if ablation_df is None:
        print("No ablation results to analyze")
        return None

    modalities = sorted(ablation_df['Modality'].unique())
    print(f"\n--- Modality Ablation Significance ---")
    print(f"Modalities: {modalities}")

    ablation_results = []

    # Compare each modality pair: specifically, compare 'All' vs each individual
    all_modalities = [m for m in modalities if m == 'All']
    single_modalities = [m for m in modalities if '+' not in m and m != 'All']
    combo_modalities = [m for m in modalities if '+' in m]

    # Compare All vs each single/combo
    for metric in ['R2', 'AUROC']:
        if metric not in ablation_df.columns:
            continue

        for ref_mod in ['All']:
            vals_ref = ablation_df[ablation_df['Modality'] == ref_mod][metric].dropna().values
            if len(vals_ref) == 0:
                continue

            for comp_mod in single_modalities + combo_modalities:
                vals_comp = ablation_df[ablation_df['Modality'] == comp_mod][metric].dropna().values
                if len(vals_comp) == 0 or len(vals_comp) != len(vals_ref):
                    continue

                try:
                    t_stat, t_pval = stats.ttest_rel(vals_ref, vals_comp)
                except Exception:
                    t_pval = np.nan

                ablation_results.append({
                    'Modality_Full': ref_mod,
                    'Modality_Reduced': comp_mod,
                    'Metric': metric,
                    'Mean_Full': np.mean(vals_ref),
                    'Mean_Reduced': np.mean(vals_comp),
                    'Improvement': np.mean(vals_ref) - np.mean(vals_comp),
                    't_test_pval': t_pval,
                    'Significant_005': 'Yes' if t_pval < 0.05 else 'No'
                })

    results_df = pd.DataFrame(ablation_results)
    print(f"Completed {len(results_df)} ablation comparisons")
    return results_df


def compute_confidence_intervals(results_df, metric='R2'):
    """Compute 95% t-distribution confidence intervals for all models."""
    if results_df is None or metric not in results_df.columns:
        return None

    print(f"\n--- Confidence Intervals ({metric}) ---")

    ci_results = []
    for model in results_df['Model'].unique():
        fold_values = results_df[results_df['Model'] == model][metric].dropna().values
        if len(fold_values) < 2:
            continue

        n = len(fold_values)
        mean = np.mean(fold_values)
        std = np.std(fold_values, ddof=1)
        se = std / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se

        ci_results.append({
            'Model': model,
            'Metric': metric,
            'N_Folds': n,
            'Mean': mean,
            'Std': std,
            'SE': se,
            'CI_Lower_95pct': ci_lower,
            'CI_Upper_95pct': ci_upper,
            'CI_Width': ci_upper - ci_lower
        })

        print(f"  {model}: {mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    return pd.DataFrame(ci_results)


def plot_statistical_significance(pvalue_matrix, models):
    """Create heatmap of pairwise p-values."""
    print(f"\n--- Generating Significance Heatmap ---")

    mask = np.triu(np.ones_like(pvalue_matrix, dtype=bool), k=0)
    np.fill_diagonal(pvalue_matrix, np.nan)

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)

    sns.heatmap(
        pvalue_matrix,
        mask=mask,
        xticklabels=models,
        yticklabels=models,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        vmin=0,
        vmax=0.1,
        cbar_kws={'label': 'p-value (paired t-test)'},
        ax=ax,
        square=True,
        linewidths=0.5
    )

    ax.set_title('Pairwise Model Comparisons (R²)\np < 0.05 indicates significant difference',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_figure_file, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {output_figure_file}")
    plt.close()


def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 80)

    # Load results
    print("\nLoading results...")
    reg_long, cls_long, ablation_df = load_results()

    # Test 1: Pairwise model comparison (R2)
    print("\n" + "=" * 80)
    print("TEST 1: PAIRWISE MODEL COMPARISON (R²)")
    print("=" * 80)

    comparison_r2, pvalue_matrix_r2, models_r2 = test_pairwise_model_comparison(reg_long, metric='R2')

    if comparison_r2 is not None and len(comparison_r2) > 0:
        comparison_r2.to_csv(output_models_file, sep='\t', index=False)
        print(f"\nSaved: {output_models_file}")
        sig = comparison_r2[comparison_r2['Significant_005'] == 'Yes']
        print(f"\nSignificant pairwise differences (p < 0.05): {len(sig)} / {len(comparison_r2)}")
        if len(sig) > 0:
            print(sig[['Model_1', 'Model_2', 'Difference', 't_test_pval']].to_string(index=False))

    # Test 2: Pairwise model comparison (AUROC)
    print("\n" + "=" * 80)
    print("TEST 2: PAIRWISE MODEL COMPARISON (AUROC)")
    print("=" * 80)

    comparison_auroc, _, _ = test_pairwise_model_comparison(cls_long, metric='AUROC')
    if comparison_auroc is not None and len(comparison_auroc) > 0:
        cls_output = os.path.join(TABLES_DIR, 'statistical_tests_classification.tsv')
        comparison_auroc.to_csv(cls_output, sep='\t', index=False)
        print(f"\nSaved: {cls_output}")

    # Test 3: Modality ablation significance
    print("\n" + "=" * 80)
    print("TEST 3: MODALITY ABLATION SIGNIFICANCE")
    print("=" * 80)

    ablation_results = test_modality_ablation(ablation_df)
    if ablation_results is not None and len(ablation_results) > 0:
        ablation_results.to_csv(output_ablation_file, sep='\t', index=False)
        print(f"\nSaved: {output_ablation_file}")

    # Test 4: Confidence intervals
    print("\n" + "=" * 80)
    print("TEST 4: CONFIDENCE INTERVALS")
    print("=" * 80)

    ci_reg = compute_confidence_intervals(reg_long, metric='R2')
    ci_cls = compute_confidence_intervals(cls_long, metric='AUROC')

    ci_all = pd.concat([ci_reg, ci_cls], ignore_index=True) if ci_reg is not None and ci_cls is not None else (ci_reg if ci_reg is not None else ci_cls)
    if ci_all is not None:
        ci_all.to_csv(output_ci_file, sep='\t', index=False)
        print(f"\nSaved: {output_ci_file}")

    # Figure: Statistical significance heatmap
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    if pvalue_matrix_r2 is not None and models_r2 is not None and len(models_r2) > 1:
        plot_statistical_significance(pvalue_matrix_r2, models_r2)

    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs:")
    for f in [output_models_file, output_ablation_file, output_ci_file, output_figure_file]:
        exists = "✓" if os.path.exists(f) else "✗"
        print(f"  {exists} {os.path.basename(f)}")


if __name__ == '__main__':
    main()
