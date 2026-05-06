#!/usr/bin/env python3
"""
Full ML Pipeline for Blood Pressure Prediction - MultiModal Genomics
Uses ONLY leak-free features (no BP variability)
Benchmarks 8 classical/tree-based models, ablation studies, and SHAP analysis.

Note: FT-Transformer and SAINT are trained separately in 06_deep_models_gpu.py
with full-scale architectures (d_model=192, 8 heads, 3 layers). The results
reported in the paper for those two models come from script 06, not this script.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, roc_curve, confusion_matrix

warnings.filterwarnings('ignore')

# Try-except imports for optional libraries
HAS_XGBOOST = False
HAS_LIGHTGBM = False
HAS_CATBOOST = False
HAS_SHAP = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    print("[WARNING] XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    print("[WARNING] LightGBM not available")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    print("[WARNING] CatBoost not available")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    print("[WARNING] SHAP not available")

# Setup paths
OUT = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(OUT, 'tables')
FIGURES_DIR = os.path.join(OUT, 'figures')

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Constants
TARGET_REG = 'mean_arterial_pressure'
TARGET_CLS = 'hypertension'
SEED = 42
K_FOLDS = 5

print(f"\n{'='*80}")
print("BLOOD PRESSURE PREDICTION - FULL ML PIPELINE")
print(f"{'='*80}")
print(f"Working directory: {OUT}")
print(f"Seed: {SEED}, K-Folds: {K_FOLDS}")
print(f"{'='*80}\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_preprocessor():
    """Return preprocessing pipeline: SimpleImputer(median) + StandardScaler"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def load_data():
    """Load merged_dataset.tsv and feature_groups.json"""
    df_path = os.path.join(OUT, 'merged_dataset.tsv')
    fg_path = os.path.join(OUT, 'feature_groups.json')

    print(f"Loading data from {df_path}")
    df = pd.read_csv(df_path, sep='\t')

    print(f"Loading feature groups from {fg_path}")
    with open(fg_path, 'r') as f:
        feature_groups = json.load(f)

    print(f"Dataset shape: {df.shape}")
    print(f"Feature groups available: {list(feature_groups.keys())}")

    return df, feature_groups



# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_all_models(df, feature_cols, target_col, task='regression', k=5):
    """
    Run k-fold cross-validation with all available models.
    Returns per-fold results (not just means).
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING ALL MODELS - {task.upper()}")
    print(f"Target: {target_col}, Features: {len(feature_cols)}")
    print(f"{'='*80}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1}/{k} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Preprocessing
        preprocessor = get_preprocessor()
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)

        # Model results for this fold
        fold_data = {'fold': fold + 1}

        # ===== LINEAR MODELS =====
        if task == 'regression':
            # Ridge
            print("  Training Ridge...", end=' ', flush=True)
            ridge = Ridge(alpha=10)
            ridge.fit(X_train_scaled, y_train)
            y_pred = ridge.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            fold_data['Ridge_R2'] = r2
            fold_data['Ridge_RMSE'] = rmse
            fold_data['Ridge_MAE'] = mae
            print(f"R2={r2:.4f}")

            # Lasso
            print("  Training Lasso...", end=' ', flush=True)
            lasso = Lasso(alpha=0.01, max_iter=10000)
            lasso.fit(X_train_scaled, y_train)
            y_pred = lasso.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            fold_data['Lasso_R2'] = r2
            fold_data['Lasso_RMSE'] = rmse
            fold_data['Lasso_MAE'] = mae
            print(f"R2={r2:.4f}")

            # ElasticNet
            print("  Training ElasticNet...", end=' ', flush=True)
            en = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
            en.fit(X_train_scaled, y_train)
            y_pred = en.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            fold_data['ElasticNet_R2'] = r2
            fold_data['ElasticNet_RMSE'] = rmse
            fold_data['ElasticNet_MAE'] = mae
            print(f"R2={r2:.4f}")

        else:  # classification
            # Logistic Regression
            print("  Training LogisticRegression...", end=' ', flush=True)
            lr = LogisticRegression(max_iter=10000, random_state=SEED)
            lr.fit(X_train_scaled, y_train.astype(int))
            y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
            auroc = roc_auc_score(y_test.astype(int), y_pred_proba)
            fold_data['LogisticRegression_AUROC'] = auroc
            print(f"AUROC={auroc:.4f}")

        # ===== TREE MODELS =====
        if task == 'regression':
            # RandomForest
            print("  Training RandomForest (Regression)...", end=' ', flush=True)
            rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=SEED, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            y_pred = rf.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            fold_data['RandomForest_R2'] = r2
            fold_data['RandomForest_RMSE'] = rmse
            fold_data['RandomForest_MAE'] = mae
            print(f"R2={r2:.4f}")

            # GradientBoosting
            print("  Training GradientBoosting (Regression)...", end=' ', flush=True)
            gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=SEED)
            gb.fit(X_train_scaled, y_train)
            y_pred = gb.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            fold_data['GradientBoosting_R2'] = r2
            fold_data['GradientBoosting_RMSE'] = rmse
            fold_data['GradientBoosting_MAE'] = mae
            print(f"R2={r2:.4f}")

        else:  # classification
            # RandomForest
            print("  Training RandomForest (Classification)...", end=' ', flush=True)
            rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=SEED, n_jobs=-1)
            rf.fit(X_train_scaled, y_train.astype(int))
            y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
            auroc = roc_auc_score(y_test.astype(int), y_pred_proba)
            fold_data['RandomForest_AUROC'] = auroc
            print(f"AUROC={auroc:.4f}")

            # GradientBoosting
            print("  Training GradientBoosting (Classification)...", end=' ', flush=True)
            gb = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=SEED)
            gb.fit(X_train_scaled, y_train.astype(int))
            y_pred_proba = gb.predict_proba(X_test_scaled)[:, 1]
            auroc = roc_auc_score(y_test.astype(int), y_pred_proba)
            fold_data['GradientBoosting_AUROC'] = auroc
            print(f"AUROC={auroc:.4f}")

        # ===== XGBOOST =====
        if HAS_XGBOOST:
            if task == 'regression':
                print("  Training XGBoost (Regression)...", end=' ', flush=True)
                xgb_model = xgb.XGBRegressor(
                    n_estimators=500, max_depth=6, learning_rate=0.1,
                    random_state=SEED, n_jobs=-1, verbosity=0
                )
                xgb_model.fit(X_train_scaled, y_train)
                y_pred = xgb_model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                fold_data['XGBoost_R2'] = r2
                fold_data['XGBoost_RMSE'] = rmse
                fold_data['XGBoost_MAE'] = mae
                print(f"R2={r2:.4f}")
            else:
                print("  Training XGBoost (Classification)...", end=' ', flush=True)
                xgb_model = xgb.XGBClassifier(
                    n_estimators=500, max_depth=6, learning_rate=0.1,
                    random_state=SEED, n_jobs=-1, verbosity=0
                )
                xgb_model.fit(X_train_scaled, y_train.astype(int))
                y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
                auroc = roc_auc_score(y_test.astype(int), y_pred_proba)
                fold_data['XGBoost_AUROC'] = auroc
                print(f"AUROC={auroc:.4f}")

        # ===== LIGHTGBM =====
        if HAS_LIGHTGBM:
            if task == 'regression':
                print("  Training LightGBM (Regression)...", end=' ', flush=True)
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=500, max_depth=6, learning_rate=0.1,
                    random_state=SEED, n_jobs=-1, verbose=-1
                )
                lgb_model.fit(X_train_scaled, y_train)
                y_pred = lgb_model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                fold_data['LightGBM_R2'] = r2
                fold_data['LightGBM_RMSE'] = rmse
                fold_data['LightGBM_MAE'] = mae
                print(f"R2={r2:.4f}")
            else:
                print("  Training LightGBM (Classification)...", end=' ', flush=True)
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=500, max_depth=6, learning_rate=0.1,
                    random_state=SEED, n_jobs=-1, verbose=-1
                )
                lgb_model.fit(X_train_scaled, y_train.astype(int))
                y_pred_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
                auroc = roc_auc_score(y_test.astype(int), y_pred_proba)
                fold_data['LightGBM_AUROC'] = auroc
                print(f"AUROC={auroc:.4f}")

        # ===== CATBOOST =====
        if HAS_CATBOOST:
            if task == 'regression':
                print("  Training CatBoost (Regression)...", end=' ', flush=True)
                cb_model = cb.CatBoostRegressor(
                    iterations=500, max_depth=6, learning_rate=0.1,
                    random_state=SEED, verbose=False
                )
                cb_model.fit(X_train_scaled, y_train)
                y_pred = cb_model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                fold_data['CatBoost_R2'] = r2
                fold_data['CatBoost_RMSE'] = rmse
                fold_data['CatBoost_MAE'] = mae
                print(f"R2={r2:.4f}")
            else:
                print("  Training CatBoost (Classification)...", end=' ', flush=True)
                cb_model = cb.CatBoostClassifier(
                    iterations=500, max_depth=6, learning_rate=0.1,
                    random_state=SEED, verbose=False
                )
                cb_model.fit(X_train_scaled, y_train.astype(int))
                y_pred_proba = cb_model.predict_proba(X_test_scaled)[:, 1]
                auroc = roc_auc_score(y_test.astype(int), y_pred_proba)
                fold_data['CatBoost_AUROC'] = auroc
                print(f"AUROC={auroc:.4f}")

        # Note: FT-Transformer and SAINT are trained in 06_deep_models_gpu.py
        # with the full-scale architecture used in the paper.

        fold_results.append(fold_data)

    return pd.DataFrame(fold_results)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    # Load data
    df, feature_groups = load_data()

    # Verify leak-free features exist
    if 'all_leak_free' not in feature_groups:
        print("[ERROR] 'all_leak_free' feature group not found!")
        return

    leak_free_features = feature_groups['all_leak_free']
    print(f"\nUsing {len(leak_free_features)} leak-free features")
    print(f"Features: {leak_free_features[:5]}...")

    # ========================================================================
    # EXPERIMENT 1: Model Comparison (All leak-free features)
    # ========================================================================
    print(f"\n{'='*80}")
    print("EXPERIMENT 1: MODEL COMPARISON (ALL LEAK-FREE FEATURES)")
    print(f"{'='*80}")

    # Regression
    print("\n[REGRESSION] Evaluating all models...")
    fold_results_reg = evaluate_all_models(
        df, leak_free_features, TARGET_REG, task='regression', k=K_FOLDS
    )

    # Save fold-level results
    fold_results_reg.to_csv(os.path.join(TABLES_DIR, 'fold_results_regression.tsv'), sep='\t', index=False)
    print(f"Saved fold-level regression results to {os.path.join(TABLES_DIR, 'fold_results_regression.tsv')}")

    # Calculate summaries
    model_cols_reg = [col for col in fold_results_reg.columns if '_R2' in col or '_RMSE' in col or '_MAE' in col]
    summary_reg = fold_results_reg[model_cols_reg].describe()
    summary_reg.to_csv(os.path.join(TABLES_DIR, 'model_comparison_regression.tsv'), sep='\t')
    print(f"Saved regression comparison to {os.path.join(TABLES_DIR, 'model_comparison_regression.tsv')}")

    # Classification
    print("\n[CLASSIFICATION] Evaluating all models...")
    fold_results_cls = evaluate_all_models(
        df, leak_free_features, TARGET_CLS, task='classification', k=K_FOLDS
    )

    # Save fold-level results
    fold_results_cls.to_csv(os.path.join(TABLES_DIR, 'fold_results_classification.tsv'), sep='\t', index=False)
    print(f"Saved fold-level classification results to {os.path.join(TABLES_DIR, 'fold_results_classification.tsv')}")

    # Calculate summaries
    model_cols_cls = [col for col in fold_results_cls.columns if '_AUROC' in col]
    summary_cls = fold_results_cls[model_cols_cls].describe()
    summary_cls.to_csv(os.path.join(TABLES_DIR, 'model_comparison_classification.tsv'), sep='\t')
    print(f"Saved classification comparison to {os.path.join(TABLES_DIR, 'model_comparison_classification.tsv')}")

    # ========================================================================
    # EXPERIMENT 2: Modality Ablation (using XGBoost, or best available)
    # ========================================================================
    print(f"\n{'='*80}")
    print("EXPERIMENT 2: MODALITY ABLATION")
    print(f"{'='*80}")

    # Determine best available model for ablation
    ablation_model_name = 'XGBoost' if HAS_XGBOOST else ('LightGBM' if HAS_LIGHTGBM else 'GradientBoosting')
    print(f"Using {ablation_model_name} for ablation")

    ablation_configs = {
        'Clinical': feature_groups.get('clinical_leak_free', []),
        'Labs': feature_groups.get('labs', []),
        'PRS': feature_groups.get('prs', []),
        'PCA': feature_groups.get('pca', []),
        'Clinical+Labs': feature_groups.get('clinical_leak_free', []) + feature_groups.get('labs', []),
        'PRS+PCA': feature_groups.get('prs', []) + feature_groups.get('pca', []),
        'Clinical+Labs+PRS': feature_groups.get('clinical_leak_free', []) + feature_groups.get('labs', []) + feature_groups.get('prs', []),
        'All': leak_free_features
    }

    ablation_fold_results = []
    ablation_summary_reg = {}
    ablation_summary_cls = {}

    for config_name, feature_list in ablation_configs.items():
        if len(feature_list) == 0:
            print(f"\nSkipping {config_name} - no features available")
            continue

        print(f"\nAblation: {config_name} ({len(feature_list)} features)")

        # Regression
        X = df[feature_list].values.astype(np.float32)
        y_reg = df[TARGET_REG].values.astype(np.float32)

        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
        config_results_reg = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_reg[train_idx], y_reg[test_idx]

            preprocessor = get_preprocessor()
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            if ablation_model_name == 'XGBoost' and HAS_XGBOOST:
                model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=SEED, n_jobs=-1, verbosity=0)
            elif ablation_model_name == 'LightGBM' and HAS_LIGHTGBM:
                model = lgb.LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=SEED, n_jobs=-1, verbose=-1)
            else:
                model = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=SEED)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            config_results_reg.append({
                'Modality': config_name,
                'Fold': fold + 1,
                'R2': r2,
                'RMSE': rmse
            })

        ablation_summary_reg[config_name] = {
            'R2_mean': np.mean([r['R2'] for r in config_results_reg]),
            'R2_std': np.std([r['R2'] for r in config_results_reg]),
            'RMSE_mean': np.mean([r['RMSE'] for r in config_results_reg]),
            'RMSE_std': np.std([r['RMSE'] for r in config_results_reg])
        }

        ablation_fold_results.extend(config_results_reg)

        # Classification
        y_cls = df[TARGET_CLS].values.astype(int)
        config_results_cls = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_cls[train_idx], y_cls[test_idx]

            preprocessor = get_preprocessor()
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            if ablation_model_name == 'XGBoost' and HAS_XGBOOST:
                model = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=SEED, n_jobs=-1, verbosity=0)
            elif ablation_model_name == 'LightGBM' and HAS_LIGHTGBM:
                model = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=SEED, n_jobs=-1, verbose=-1)
            else:
                model = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=SEED)

            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            auroc = roc_auc_score(y_test, y_pred_proba)

            config_results_cls.append({
                'Modality': config_name,
                'Fold': fold + 1,
                'AUROC': auroc
            })

        ablation_summary_cls[config_name] = {
            'AUROC_mean': np.mean([r['AUROC'] for r in config_results_cls]),
            'AUROC_std': np.std([r['AUROC'] for r in config_results_cls])
        }

        ablation_fold_results.extend(config_results_cls)

    # Save ablation results
    ablation_fold_df = pd.DataFrame(ablation_fold_results)
    ablation_fold_df.to_csv(os.path.join(TABLES_DIR, 'fold_results_ablation.tsv'), sep='\t', index=False)
    print(f"\nSaved fold-level ablation results to {os.path.join(TABLES_DIR, 'fold_results_ablation.tsv')}")

    # Save summaries
    summary_reg_df = pd.DataFrame(ablation_summary_reg).T
    summary_reg_df.to_csv(os.path.join(TABLES_DIR, 'ablation_regression.tsv'), sep='\t')
    print(f"Saved regression ablation summary to {os.path.join(TABLES_DIR, 'ablation_regression.tsv')}")

    summary_cls_df = pd.DataFrame(ablation_summary_cls).T
    summary_cls_df.to_csv(os.path.join(TABLES_DIR, 'ablation_classification.tsv'), sep='\t')
    print(f"Saved classification ablation summary to {os.path.join(TABLES_DIR, 'ablation_classification.tsv')}")

    # ========================================================================
    # EXPERIMENT 3: SHAP Analysis
    # ========================================================================
    if HAS_SHAP and HAS_XGBOOST:
        print(f"\n{'='*80}")
        print("EXPERIMENT 3: SHAP ANALYSIS")
        print(f"{'='*80}")

        X = df[leak_free_features].values.astype(np.float32)
        y = df[TARGET_REG].values.astype(np.float32)

        # Preprocess
        preprocessor = get_preprocessor()
        X_scaled = preprocessor.fit_transform(X)

        # Train XGBoost on full data
        print("\nTraining XGBoost on full data for SHAP analysis...")
        xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=SEED, n_jobs=-1, verbosity=0)
        xgb_model.fit(X_scaled, y)

        # SHAP TreeExplainer on subsample
        print("Computing SHAP values...")
        sample_indices = np.random.choice(X_scaled.shape[0], size=min(2000, X_scaled.shape[0]), replace=False)
        X_subsample = X_scaled[sample_indices]

        try:
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_subsample)
            shap_importance = np.mean(np.abs(shap_values), axis=0)
        except (ValueError, Exception) as e:
            print(f"  TreeExplainer failed ({e}), using feature_importances_ instead")
            shap_values = None
            shap_importance = xgb_model.feature_importances_

        # Assign modality
        def get_modality(feature_name):
            if feature_name.startswith('LAB_'):
                return 'Labs'
            elif feature_name.startswith('PRS_'):
                return 'PRS'
            elif feature_name.startswith('PC'):
                return 'PCA'
            else:
                return 'Clinical'

        modality_counts = defaultdict(float)
        for feat_name, importance in zip(leak_free_features, shap_importance):
            modality = get_modality(feat_name)
            modality_counts[modality] += importance

        # Save SHAP importance
        shap_df = pd.DataFrame({
            'Feature': leak_free_features,
            'SHAP_Importance': shap_importance,
            'Modality': [get_modality(f) for f in leak_free_features]
        }).sort_values('SHAP_Importance', ascending=False)

        shap_df.to_csv(os.path.join(TABLES_DIR, 'shap_importance.tsv'), sep='\t', index=False)
        print(f"Saved SHAP importance to {os.path.join(TABLES_DIR, 'shap_importance.tsv')}")

        # Generate SHAP summary plot (top 30 features)
        print("Generating SHAP summary plot...")
        top_features_idx = np.argsort(shap_importance)[-30:]
        top_feature_names = [leak_free_features[i] for i in top_features_idx]

        if shap_values is not None:
            top_shap_values = shap_values[:, top_features_idx]
            X_subsample_top = X_subsample[:, top_features_idx]
            plt.figure(figsize=(12, 10))
            shap.summary_plot(top_shap_values, X_subsample_top, feature_names=top_feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'fig_shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # Fallback: horizontal bar chart of feature importances
            plt.figure(figsize=(12, 10))
            plt.barh(range(len(top_features_idx)), shap_importance[top_features_idx])
            plt.yticks(range(len(top_features_idx)), top_feature_names)
            plt.xlabel('Feature Importance')
            plt.title('Top 30 Features by XGBoost Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'fig_shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
        print(f"Saved SHAP summary plot to {os.path.join(FIGURES_DIR, 'fig_shap_summary.png')}")

        # Generate modality pie chart
        print("Generating modality pie chart...")
        fig, ax = plt.subplots(figsize=(8, 8))
        modalities = list(modality_counts.keys())
        values = list(modality_counts.values())
        colors = {'Clinical': '#1f77b4', 'Labs': '#ff7f0e', 'PRS': '#2ca02c', 'PCA': '#d62728'}
        color_list = [colors.get(m, '#808080') for m in modalities]

        ax.pie(values, labels=modalities, autopct='%1.1f%%', colors=color_list, startangle=90)
        ax.set_title('SHAP Importance by Modality', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig_shap_modality_pie.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved modality pie chart to {os.path.join(FIGURES_DIR, 'fig_shap_modality_pie.png')}")

    else:
        print(f"\n{'='*80}")
        print("SKIPPING EXPERIMENT 3: SHAP Analysis (SHAP or XGBoost not available)")
        print(f"{'='*80}")

    # ========================================================================
    # FIGURE GENERATION
    # ========================================================================
    print(f"\n{'='*80}")
    print("GENERATING FIGURES")
    print(f"{'='*80}")

    # Figure 1: Modality Ablation R²
    if not ablation_summary_reg.empty if isinstance(ablation_summary_reg, pd.DataFrame) else len(ablation_summary_reg) > 0:
        print("\nGenerating Fig 1: Modality Ablation (R²)...")

        modalities = list(ablation_summary_reg.keys())
        r2_means = [ablation_summary_reg[m]['R2_mean'] for m in modalities]
        r2_stds = [ablation_summary_reg[m]['R2_std'] for m in modalities]

        # Color by type
        colors = []
        for m in modalities:
            if m == 'All':
                colors.append('red')
            elif '+' in m:
                colors.append('blue')
            else:
                colors.append('gray')

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(modalities))
        ax.barh(y_pos, r2_means, xerr=r2_stds, color=colors, alpha=0.7, capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(modalities)
        ax.set_xlabel('R²', fontsize=12)
        ax.set_title('MAP Prediction — Modality Ablation (XGBoost)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        # Add value labels
        for i, (v, std) in enumerate(zip(r2_means, r2_stds)):
            ax.text(v + std + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig1_modality_ablation_r2.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Fig 1 to {os.path.join(FIGURES_DIR, 'fig1_modality_ablation_r2.png')}")

    # Figure 2: Modality Ablation AUROC
    if not ablation_summary_cls.empty if isinstance(ablation_summary_cls, pd.DataFrame) else len(ablation_summary_cls) > 0:
        print("Generating Fig 2: Modality Ablation (AUROC)...")

        modalities = list(ablation_summary_cls.keys())
        auroc_means = [ablation_summary_cls[m]['AUROC_mean'] for m in modalities]
        auroc_stds = [ablation_summary_cls[m]['AUROC_std'] for m in modalities]

        colors = []
        for m in modalities:
            if m == 'All':
                colors.append('red')
            elif '+' in m:
                colors.append('blue')
            else:
                colors.append('gray')

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(modalities))
        ax.barh(y_pos, auroc_means, xerr=auroc_stds, color=colors, alpha=0.7, capsize=5)
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Random')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(modalities)
        ax.set_xlabel('AUROC', fontsize=12)
        ax.set_xlim([0.4, 1.0])
        ax.set_title('Hypertension Classification — Modality Ablation (XGBoost)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.legend()

        for i, (v, std) in enumerate(zip(auroc_means, auroc_stds)):
            ax.text(v + std + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig2_modality_ablation_auroc.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Fig 2 to {os.path.join(FIGURES_DIR, 'fig2_modality_ablation_auroc.png')}")

    # Figure 3: Model Comparison (R²)
    print("Generating Fig 3: Model Comparison (R²)...")

    r2_cols = [col for col in fold_results_reg.columns if '_R2' in col]
    model_names = [col.replace('_R2', '') for col in r2_cols]
    r2_means = [fold_results_reg[col].mean() for col in r2_cols]
    r2_stds = [fold_results_reg[col].std() for col in r2_cols]

    # Color by family
    colors = []
    families = []
    for model in model_names:
        if model in ['Ridge', 'Lasso', 'ElasticNet']:
            colors.append('gray')
            families.append('Linear')
        else:  # Tree-based and gradient boosting
            colors.append('blue')
            families.append('Tree')

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(len(model_names))
    ax.barh(y_pos, r2_means, xerr=r2_stds, color=colors, alpha=0.7, capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.set_xlabel('R²', fontsize=12)
    ax.set_title('MAP Prediction — All Models Comparison', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    for i, (v, std) in enumerate(zip(r2_means, r2_stds)):
        ax.text(v + std + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.7, label='Linear'),
        Patch(facecolor='blue', alpha=0.7, label='Tree / Boosting')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Fig 3 to {os.path.join(FIGURES_DIR, 'fig3_model_comparison.png')}")

    # Figure 4: Predicted vs Actual (using best model)
    print("Generating Fig 4: Predicted vs Actual...")

    # Train best model (CatBoost if available, else XGBoost)
    X = df[leak_free_features].values.astype(np.float32)
    y = df[TARGET_REG].values.astype(np.float32)

    preprocessor = get_preprocessor()
    X_scaled = preprocessor.fit_transform(X)

    if HAS_CATBOOST:
        best_model = cb.CatBoostRegressor(iterations=500, max_depth=6, learning_rate=0.1, random_state=SEED, verbose=False)
    elif HAS_XGBOOST:
        best_model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=SEED, n_jobs=-1, verbosity=0)
    else:
        best_model = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=SEED)

    best_model.fit(X_scaled, y)
    y_pred = best_model.predict(X_scaled)

    r2_final = r2_score(y, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y, y_pred, alpha=0.5, s=30)

    # Diagonal line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    ax.set_xlabel('Actual MAP (mmHg)', fontsize=12)
    ax.set_ylabel('Predicted MAP (mmHg)', fontsize=12)
    ax.set_title('MAP Prediction: Predicted vs Actual', fontsize=14, fontweight='bold')
    ax.text(0.05, 0.95, f'R² = {r2_final:.4f}', transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_predicted_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Fig 4 to {os.path.join(FIGURES_DIR, 'fig6_predicted_vs_actual.png')}")

    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved to:")
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
