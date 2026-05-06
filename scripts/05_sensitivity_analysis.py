#!/usr/bin/env python3
"""
Sensitivity analysis on hypertension classification threshold.

Tests multiple MAP (Mean Arterial Pressure) thresholds to determine optimal
classification threshold for hypertension definition.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Setup
OUT = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(OUT, 'tables')
FIGURES_DIR = os.path.join(OUT, 'figures')
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

SEED = 42
K_FOLDS = 5
np.random.seed(SEED)

# File paths
dataset_file = os.path.join(OUT, 'merged_dataset.tsv')
feature_groups_file = os.path.join(OUT, 'feature_groups.json')
output_results_file = os.path.join(TABLES_DIR, 'sensitivity_map_thresholds.tsv')
output_figure_file = os.path.join(FIGURES_DIR, 'fig_sensitivity_thresholds.png')

# MAP thresholds to test (in mmHg)
# MAP = DBP + 1/3(SBP - DBP)
# Approximate: MAP >= 93.3 ~ SBP/DBP = 120/80, MAP >= 96.7 ~ 130/80,
#              MAP >= 100 ~ 140/85, MAP >= 103.3 ~ 140/90, etc.
MAP_THRESHOLDS = [93.3, 96.7, 100.0, 103.3, 107.0, 110.0]


def load_model():
    """Load best available tree classifier with fallback chain."""
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1,
                                  random_state=SEED, verbose=0), 'CatBoost'
    except ImportError:
        pass
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             random_state=SEED, verbosity=0, n_jobs=-1), 'XGBoost'
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              random_state=SEED, verbose=-1, n_jobs=-1), 'LightGBM'
    except ImportError:
        pass
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                      random_state=SEED), 'GradientBoosting'


def main():
    print("=" * 80)
    print("SENSITIVITY ANALYSIS: MAP THRESHOLD FOR HYPERTENSION")
    print("=" * 80)

    # Load data
    df = pd.read_csv(dataset_file, sep='\t')
    with open(feature_groups_file, 'r') as f:
        feature_groups = json.load(f)

    print(f"Dataset: {df.shape[0]} participants × {df.shape[1]} columns")

    # Get leak-free features
    features = feature_groups.get('all_leak_free', [])
    features = [f for f in features if f in df.columns]
    print(f"Leak-free features: {len(features)}")

    # Use mean_arterial_pressure directly (already computed in data prep)
    map_col = 'mean_arterial_pressure'
    if map_col not in df.columns:
        print(f"ERROR: {map_col} not in dataset")
        return

    map_vals = df[map_col].values

    # Prepare feature matrix
    X = df[features].values.astype(np.float32)

    # Get model
    model_template, model_name = load_model()
    print(f"Using model: {model_name}")

    # Results
    results = []

    for threshold in MAP_THRESHOLDS:
        print(f"\n--- MAP >= {threshold:.1f} mmHg ---")

        # Define binary target at this threshold
        y = (map_vals >= threshold).astype(int)
        prevalence = y.mean()
        print(f"  Prevalence: {prevalence:.1%} ({y.sum()}/{len(y)})")

        if y.sum() < 50 or (len(y) - y.sum()) < 50:
            print(f"  SKIPPING: too few cases or controls")
            continue

        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
        fold_metrics = {'auroc': [], 'accuracy': [], 'sensitivity': [], 'specificity': []}

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Preprocess
            preprocessor = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)

            # Train fresh model
            model, _ = load_model()
            model.fit(X_train_proc, y_train)

            # Predict
            y_pred_proba = model.predict_proba(X_test_proc)[:, 1]
            y_pred = model.predict(X_test_proc)

            # Metrics
            auroc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            fold_metrics['auroc'].append(auroc)
            fold_metrics['accuracy'].append(accuracy)
            fold_metrics['sensitivity'].append(sensitivity)
            fold_metrics['specificity'].append(specificity)

        results.append({
            'MAP_Threshold': threshold,
            'Prevalence': prevalence,
            'N_Cases': int(y.sum()),
            'N_Controls': int(len(y) - y.sum()),
            'AUROC_Mean': np.mean(fold_metrics['auroc']),
            'AUROC_Std': np.std(fold_metrics['auroc']),
            'Accuracy_Mean': np.mean(fold_metrics['accuracy']),
            'Accuracy_Std': np.std(fold_metrics['accuracy']),
            'Sensitivity_Mean': np.mean(fold_metrics['sensitivity']),
            'Sensitivity_Std': np.std(fold_metrics['sensitivity']),
            'Specificity_Mean': np.mean(fold_metrics['specificity']),
            'Specificity_Std': np.std(fold_metrics['specificity']),
        })

        print(f"  AUROC:       {results[-1]['AUROC_Mean']:.4f} ± {results[-1]['AUROC_Std']:.4f}")
        print(f"  Accuracy:    {results[-1]['Accuracy_Mean']:.4f} ± {results[-1]['Accuracy_Std']:.4f}")
        print(f"  Sensitivity: {results[-1]['Sensitivity_Mean']:.4f} ± {results[-1]['Sensitivity_Std']:.4f}")
        print(f"  Specificity: {results[-1]['Specificity_Mean']:.4f} ± {results[-1]['Specificity_Std']:.4f}")

    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(output_results_file, sep='\t', index=False)
    print(f"\nSaved: {output_results_file}")

    # Generate figure
    print("\nGenerating sensitivity figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: AUROC vs threshold
    ax1.errorbar(results_df['MAP_Threshold'], results_df['AUROC_Mean'],
                 yerr=results_df['AUROC_Std'], fmt='o-', capsize=5, capthick=2,
                 markersize=8, linewidth=2, color='#2E86AB', label='AUROC')

    # Mark current and ACC/AHA thresholds
    ax1.axvline(107.0, color='red', linestyle='--', lw=1.5, label='Current (107)')
    ax1.axvline(96.7, color='green', linestyle='--', lw=1.5, label='ACC/AHA Stage 1 (96.7)')

    ax1.set_xlabel('MAP Threshold (mmHg)', fontsize=12)
    ax1.set_ylabel('AUROC', fontsize=12)
    ax1.set_title(f'Classification Performance vs MAP Threshold ({model_name})',
                  fontsize=13, fontweight='bold')
    ax1.set_ylim([0.5, 1.0])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Right panel: Prevalence
    ax2.bar(results_df['MAP_Threshold'], results_df['Prevalence'] * 100,
            width=2, color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.axvline(107.0, color='red', linestyle='--', lw=1.5, label='Current (107)')
    ax2.axvline(96.7, color='green', linestyle='--', lw=1.5, label='ACC/AHA Stage 1 (96.7)')

    ax2.set_xlabel('MAP Threshold (mmHg)', fontsize=12)
    ax2.set_ylabel('Prevalence (%)', fontsize=12)
    ax2.set_title('Hypertension Prevalence vs MAP Threshold', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_figure_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_figure_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 80)
    best_idx = results_df['AUROC_Mean'].idxmax()
    best = results_df.loc[best_idx]
    print(f"\nBest threshold: MAP >= {best['MAP_Threshold']:.1f} mmHg")
    print(f"  AUROC = {best['AUROC_Mean']:.4f}, Prevalence = {best['Prevalence']:.1%}")

    print(f"\nAll results:")
    print(results_df[['MAP_Threshold', 'Prevalence', 'AUROC_Mean', 'Accuracy_Mean']].to_string(index=False))


if __name__ == '__main__':
    main()
