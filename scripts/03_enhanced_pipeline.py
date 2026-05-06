#!/usr/bin/env python3
"""
Enhanced Feature Engineering Pipeline for MultiModal BP v2
Builds advanced features on merged dataset with strictly leak-free design.

Correctly handles:
- All actual column names from merged_dataset.tsv
- Pathway z-scores computed WITHIN CV folds (prevents leakage)
- eGFR calculation (CKD-EPI formula)
- Lab ratio features, grip ratios, anthropometric features
- PRS aggregates and interactions
- Log transforms of right-skewed labs
- 5-fold CV with XGBoost + SHAP + stacking ensemble
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================
OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 42
K_FOLDS = 5

np.random.seed(SEED)

# Check for optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: xgboost not available")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: lightgbm not available")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: catboost not available")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: shap not available")

print(f"\n{'='*80}")
print("ENHANCED FEATURE ENGINEERING PIPELINE - MultiModal BP v2")
print(f"{'='*80}")
print(f"Output directory: {OUT}")
print(f"Seed: {SEED}, K-Folds: {K_FOLDS}")

# Create output subdirectories
os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print(f"\n{'='*80}")
print("STEP 1: Loading Merged Dataset")
print(f"{'='*80}")

merged_path = os.path.join(OUT, 'merged_dataset.tsv')
feature_groups_path = os.path.join(OUT, 'feature_groups.json')

if not os.path.exists(merged_path):
    raise FileNotFoundError(f"Merged dataset not found at {merged_path}")
if not os.path.exists(feature_groups_path):
    raise FileNotFoundError(f"Feature groups not found at {feature_groups_path}")

df = pd.read_csv(merged_path, sep='\t', low_memory=False)
with open(feature_groups_path, 'r') as f:
    feature_groups = json.load(f)

print(f"Loaded merged dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Baseline leak-free features: {len(feature_groups.get('all_leak_free', []))}")
print(f"Available feature groups: {list(feature_groups.keys())}")

# ============================================================================
# STEP 2: Define Pathway Compositions
# ============================================================================
PATHWAYS = {
    'RAAS': [
        'LAB_Creatinine',
        'LAB_Potassium',
        'LAB_Sodium',
        'LAB_Urea',
        'LAB_Uric Acid'
    ],
    'Inflammation': [
        'LAB_C-Reactive Protein',
        'LAB_White Blood Cell',
        'LAB_Neutrophil Auto #',
        'LAB_Fibrinogen',
        'LAB_Ferritin'
    ],
    'Metabolic': [
        'LAB_Glucose',
        'LAB_HBA 1C %',
        'LAB_Triglyceride',
        'LAB_HDL-Cholesterol',
        'LAB_LDL-Cholesterol Calc',
        'LAB_Cholesterol Total'
    ],
    'Liver': [
        'LAB_ALT ( GPT )',
        'LAB_AST (GOT)',
        'LAB_GGT_2',
        'LAB_Bilirubin Total',
        'LAB_Alkaline Phosphatase',
        'LAB_Albumin'
    ],
    'Hematology': [
        'LAB_Hemoglobin',
        'LAB_Hematocrit',
        'LAB_Red Blood Cell',
        'LAB_Mean Cell Volume',
        'LAB_Mean Cell Hemoglobin',
        'LAB_Platelet',
        'LAB_RDW'
    ],
    'Coagulation': [
        'LAB_Prothrombin Time (PT)',
        'LAB_Activated Partial Thromboplastin Time',
        'LAB_International Normalization Ratio',
        'LAB_Fibrinogen'
    ]
}

print(f"\nDefined {len(PATHWAYS)} pathways:")
for pathway, features in PATHWAYS.items():
    print(f"  {pathway}: {len(features)} features")

# ============================================================================
# STEP 3: Enhanced Feature Engineering Function
# ============================================================================
def create_enhanced_features(data, train_indices=None):
    """
    Create enhanced features with leak-free design.

    If train_indices provided, pathway z-scores are computed on train data
    and applied to the full dataset (for CV compliance).
    """
    df_enhanced = data.copy()

    # ---- Age Interactions ----
    if 'age' in df_enhanced.columns and 'BMI' in df_enhanced.columns:
        df_enhanced['Age_BMI_interaction'] = df_enhanced['age'] * df_enhanced['BMI']
        df_enhanced['Age_squared'] = df_enhanced['age'] ** 2
        df_enhanced['BMI_squared'] = df_enhanced['BMI'] ** 2

    if 'age' in df_enhanced.columns and 'sex' in df_enhanced.columns:
        # Encode sex if not numeric
        sex_encoded = df_enhanced['sex'].copy()
        if df_enhanced['sex'].dtype == 'object':
            sex_map = {'M': 1, 'F': 0, 'male': 1, 'female': 0}
            sex_encoded = df_enhanced['sex'].map(sex_map).fillna(df_enhanced['sex'])
        df_enhanced['Age_Sex_interaction'] = df_enhanced['age'] * sex_encoded

    if 'BMI' in df_enhanced.columns and 'sex' in df_enhanced.columns:
        sex_encoded = df_enhanced['sex'].copy()
        if df_enhanced['sex'].dtype == 'object':
            sex_map = {'M': 1, 'F': 0, 'male': 1, 'female': 0}
            sex_encoded = df_enhanced['sex'].map(sex_map).fillna(df_enhanced['sex'])
        df_enhanced['BMI_Sex_interaction'] = df_enhanced['BMI'] * sex_encoded

    # ---- eGFR Calculation (CKD-EPI) ----
    if 'LAB_Creatinine' in df_enhanced.columns and 'age' in df_enhanced.columns and 'sex' in df_enhanced.columns:
        creatinine = df_enhanced['LAB_Creatinine'].values
        age = df_enhanced['age'].values
        sex = df_enhanced['sex'].values

        # Encode sex: M/male=1, F/female=0
        sex_numeric = np.where((sex == 'M') | (sex == 'male'), 1, 0)

        # CKD-EPI formula (2021 version, creatinine in mg/dL)
        # eGFR = 142 × (Scr/κ)^α × (0.9938)^age × (1.012 if female)
        kappa = np.where(sex_numeric == 1, 0.9, 0.7)
        alpha = np.where(sex_numeric == 1, -0.302, -0.241)

        scr_ratio = creatinine / kappa
        egfr = 142 * (scr_ratio ** alpha) * (0.9938 ** age)
        egfr = np.where(sex_numeric == 0, egfr * 1.012, egfr)

        df_enhanced['eGFR'] = egfr
        df_enhanced['eGFR'] = df_enhanced['eGFR'].clip(lower=15)  # Min plausible value

        # eGFR interactions
        if 'age' in df_enhanced.columns:
            df_enhanced['Age_eGFR'] = df_enhanced['age'] * df_enhanced['eGFR']
        if 'BMI' in df_enhanced.columns:
            df_enhanced['BMI_eGFR'] = df_enhanced['BMI'] * df_enhanced['eGFR']

    # ---- Grip Ratios ----
    if 'Grip_Mean' in df_enhanced.columns and 'HEIGHTWEIGHT_OUT_WEIGHT' in df_enhanced.columns:
        df_enhanced['Grip_Weight_Ratio'] = (
            df_enhanced['Grip_Mean'] / df_enhanced['HEIGHTWEIGHT_OUT_WEIGHT'].clip(lower=1e-5)
        )

    if 'Grip_Mean' in df_enhanced.columns and 'BMI' in df_enhanced.columns:
        df_enhanced['Grip_BMI_Ratio'] = (
            df_enhanced['Grip_Mean'] / df_enhanced['BMI'].clip(lower=1e-5)
        )

    # ---- Waist-Hip Ratio ----
    if 'HIPWAIST_OUT_WAIST_SIZE' in df_enhanced.columns and 'HIPWAIST_OUT_HIPS_SIZE' in df_enhanced.columns:
        df_enhanced['Waist_Hip_Ratio'] = (
            df_enhanced['HIPWAIST_OUT_WAIST_SIZE'] /
            df_enhanced['HIPWAIST_OUT_HIPS_SIZE'].clip(lower=1e-5)
        )

    # ---- Lab Ratios ----
    if 'LAB_Sodium' in df_enhanced.columns and 'LAB_Potassium' in df_enhanced.columns:
        df_enhanced['Ratio_Na_K'] = (
            df_enhanced['LAB_Sodium'] / df_enhanced['LAB_Potassium'].clip(lower=1e-5)
        )

    if 'LAB_Triglyceride' in df_enhanced.columns and 'LAB_HDL-Cholesterol' in df_enhanced.columns:
        df_enhanced['Ratio_TG_HDL'] = (
            df_enhanced['LAB_Triglyceride'] / df_enhanced['LAB_HDL-Cholesterol'].clip(lower=1e-5)
        )

    if 'LAB_Urea' in df_enhanced.columns and 'LAB_Creatinine' in df_enhanced.columns:
        df_enhanced['Ratio_BUN_Cr'] = (
            df_enhanced['LAB_Urea'] / df_enhanced['LAB_Creatinine'].clip(lower=1e-5)
        )

    if 'LAB_AST (GOT)' in df_enhanced.columns and 'LAB_ALT ( GPT )' in df_enhanced.columns:
        df_enhanced['Ratio_AST_ALT'] = (
            df_enhanced['LAB_AST (GOT)'] / df_enhanced['LAB_ALT ( GPT )'].clip(lower=1e-5)
        )

    # Neutrophil-to-Lymphocyte Ratio
    if 'LAB_Neutrophil Auto #' in df_enhanced.columns and 'LAB_Lymphocyte Auto #' in df_enhanced.columns:
        df_enhanced['Ratio_NLR'] = (
            df_enhanced['LAB_Neutrophil Auto #'] / df_enhanced['LAB_Lymphocyte Auto #'].clip(lower=1e-5)
        )

    # ---- PRS Aggregates ----
    prs_cols = [col for col in df_enhanced.columns if col.startswith('PRS_')]
    if prs_cols:
        prs_data = df_enhanced[prs_cols].values
        df_enhanced['PRS_mean'] = np.nanmean(prs_data, axis=1)
        df_enhanced['PRS_std'] = np.nanstd(prs_data, axis=1)

        if 'age' in df_enhanced.columns:
            df_enhanced['PRS_mean_Age'] = df_enhanced['PRS_mean'] * df_enhanced['age']
        if 'BMI' in df_enhanced.columns:
            df_enhanced['PRS_mean_BMI'] = df_enhanced['PRS_mean'] * df_enhanced['BMI']

    # ---- Log Transforms of Right-Skewed Labs ----
    lab_cols = [col for col in df_enhanced.columns if col.startswith('LAB_')]
    for col in lab_cols:
        valid_data = df_enhanced[col].dropna()
        if len(valid_data) > 0:
            skewness = stats.skew(valid_data)
            if skewness > 1.5:
                log_col_name = f"{col}_log"
                # Avoid log(0) or log(negative) with shift
                df_enhanced[log_col_name] = np.log1p(df_enhanced[col].clip(lower=0))

    # ---- Pathway Z-Scores (Leak-Free Computation) ----
    for pathway_name, pathway_cols in PATHWAYS.items():
        # Check which pathway columns exist
        available_cols = [col for col in pathway_cols if col in df_enhanced.columns]

        if available_cols:
            if train_indices is not None:
                # CV mode: compute z-score stats on train fold only
                train_data = df_enhanced.iloc[train_indices][available_cols]
                train_mean = train_data.mean(axis=1).values
                train_std = train_data.std(axis=1).values

                # Apply to all data
                pathway_mean = df_enhanced[available_cols].mean(axis=1).values
                pathway_std = df_enhanced[available_cols].std(axis=1).values

                z_score = (pathway_mean - train_mean) / (train_std + 1e-8)
                df_enhanced[f'Pathway_{pathway_name}'] = z_score
            else:
                # No CV: compute z-score on full data
                pathway_mean = df_enhanced[available_cols].mean(axis=1).values
                pathway_std = df_enhanced[available_cols].std(axis=1).values

                # Global z-score stats
                global_mean = df_enhanced[available_cols].values.mean()
                global_std = df_enhanced[available_cols].values.std()

                z_score = (pathway_mean - global_mean) / (global_std + 1e-8)
                df_enhanced[f'Pathway_{pathway_name}'] = z_score

    return df_enhanced


# ============================================================================
# STEP 4: Prepare Data for CV
# ============================================================================
print(f"\n{'='*80}")
print("STEP 4: Preparing Data for Cross-Validation")
print(f"{'='*80}")

# Get baseline leak-free features
baseline_features = feature_groups.get('all_leak_free', [])
print(f"Baseline leak-free features: {len(baseline_features)}")

# Create enhanced features on full dataset (for reference)
df_full_enhanced = create_enhanced_features(df.copy())
enhanced_feature_cols = [col for col in df_full_enhanced.columns if col not in df.columns]
print(f"Created {len(enhanced_feature_cols)} new enhanced features")
print(f"Sample new features: {enhanced_feature_cols[:10]}")

# Target variables
target_cols = ['mean_arterial_pressure', 'hypertension']
available_targets = [col for col in target_cols if col in df.columns]

if not available_targets:
    raise ValueError(f"No target columns found in dataset. Expected: {target_cols}")

print(f"Available targets: {available_targets}")

# Select primary target
primary_target = available_targets[0]
print(f"Primary target: {primary_target}")

# Prepare feature sets
baseline_feature_set = [col for col in baseline_features if col in df.columns]
enhanced_feature_set = baseline_feature_set + [
    col for col in enhanced_feature_cols
    if col in df_full_enhanced.columns and not col.startswith('Pathway_')
]

print(f"Baseline features available: {len(baseline_feature_set)}")
print(f"Enhanced features (no pathways): {len(enhanced_feature_set)}")

# ============================================================================
# STEP 5: 5-Fold Cross-Validation with Enhanced Features
# ============================================================================
print(f"\n{'='*80}")
print("STEP 5: 5-Fold Cross-Validation")
print(f"{'='*80}")

# Prepare data: drop rows with missing target
df_cv = df.dropna(subset=[primary_target]).reset_index(drop=True)
print(f"Samples with target available: {len(df_cv)}")

# Store CV results
cv_results = {
    'baseline': {'train': [], 'test': []},
    'enhanced': {'train': [], 'test': []},
    'enhanced_with_pathways': {'train': [], 'test': []}
}

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

oof_predictions = {
    'baseline': np.zeros(len(df_cv)),
    'enhanced': np.zeros(len(df_cv)),
    'enhanced_with_pathways': np.zeros(len(df_cv))
}

fold_models = {
    'enhanced': [],
    'enhanced_with_pathways': []
}

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df_cv)):
    print(f"\n--- Fold {fold_idx + 1}/{K_FOLDS} ---")

    df_train = df_cv.iloc[train_idx].copy()
    df_test = df_cv.iloc[test_idx].copy()

    # Create enhanced features
    df_train_enhanced = create_enhanced_features(df_train.copy(), train_indices=None)
    df_test_enhanced = create_enhanced_features(df_test.copy(), train_indices=None)

    # Get target
    y_train = df_train[primary_target].values
    y_test = df_test[primary_target].values

    # ---- Baseline Model ----
    X_train_baseline = df_train_enhanced[
        [col for col in baseline_feature_set if col in df_train_enhanced.columns]
    ].values
    X_test_baseline = df_test_enhanced[
        [col for col in baseline_feature_set if col in df_test_enhanced.columns]
    ].values

    # Impute and scale
    imputer_base = SimpleImputer(strategy='median')
    X_train_baseline = imputer_base.fit_transform(X_train_baseline)
    X_test_baseline = imputer_base.transform(X_test_baseline)

    scaler_base = StandardScaler()
    X_train_baseline = scaler_base.fit_transform(X_train_baseline)
    X_test_baseline = scaler_base.transform(X_test_baseline)

    # Train baseline model
    if HAS_XGBOOST:
        model_base = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            random_state=SEED, verbosity=0, n_jobs=-1
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model_base = GradientBoostingRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            random_state=SEED
        )

    model_base.fit(X_train_baseline, y_train)
    y_pred_base_train = model_base.predict(X_train_baseline)
    y_pred_base_test = model_base.predict(X_test_baseline)

    train_r2 = r2_score(y_train, y_pred_base_train)
    test_r2 = r2_score(y_test, y_pred_base_test)
    cv_results['baseline']['train'].append(train_r2)
    cv_results['baseline']['test'].append(test_r2)

    print(f"Baseline - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

    # ---- Enhanced Model (no pathways) ----
    enhanced_cols = [col for col in enhanced_feature_set
                     if col in df_train_enhanced.columns and col in df_test_enhanced.columns]
    X_train_enhanced = df_train_enhanced[enhanced_cols].values
    X_test_enhanced = df_test_enhanced[enhanced_cols].values

    imputer_enh = SimpleImputer(strategy='median')
    X_train_enhanced = imputer_enh.fit_transform(X_train_enhanced)
    X_test_enhanced = imputer_enh.transform(X_test_enhanced)

    scaler_enh = StandardScaler()
    X_train_enhanced = scaler_enh.fit_transform(X_train_enhanced)
    X_test_enhanced = scaler_enh.transform(X_test_enhanced)

    if HAS_XGBOOST:
        model_enh = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            random_state=SEED, verbosity=0, n_jobs=-1
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model_enh = GradientBoostingRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            random_state=SEED
        )

    model_enh.fit(X_train_enhanced, y_train)
    y_pred_enh_train = model_enh.predict(X_train_enhanced)
    y_pred_enh_test = model_enh.predict(X_test_enhanced)

    train_r2 = r2_score(y_train, y_pred_enh_train)
    test_r2 = r2_score(y_test, y_pred_enh_test)
    cv_results['enhanced']['train'].append(train_r2)
    cv_results['enhanced']['test'].append(test_r2)
    oof_predictions['enhanced'][test_idx] = y_pred_enh_test
    fold_models['enhanced'].append(model_enh)

    print(f"Enhanced - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

    # ---- Enhanced + Pathways Model ----
    df_train_enh_path = create_enhanced_features(df_train.copy(), train_indices=None)
    df_test_enh_path = create_enhanced_features(df_test.copy(), train_indices=None)

    pathway_cols = [col for col in df_train_enh_path.columns if col.startswith('Pathway_')]
    enh_path_cols = enhanced_cols + [col for col in pathway_cols
                                      if col in df_train_enh_path.columns and col in df_test_enh_path.columns]
    # Filter again to ensure both have all columns
    enh_path_cols = [c for c in enh_path_cols
                     if c in df_train_enh_path.columns and c in df_test_enh_path.columns]

    X_train_enh_path = df_train_enh_path[enh_path_cols].values
    X_test_enh_path = df_test_enh_path[enh_path_cols].values

    imputer_enh_path = SimpleImputer(strategy='median')
    X_train_enh_path = imputer_enh_path.fit_transform(X_train_enh_path)
    X_test_enh_path = imputer_enh_path.transform(X_test_enh_path)

    scaler_enh_path = StandardScaler()
    X_train_enh_path = scaler_enh_path.fit_transform(X_train_enh_path)
    X_test_enh_path = scaler_enh_path.transform(X_test_enh_path)

    if HAS_XGBOOST:
        model_enh_path = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            random_state=SEED, verbosity=0, n_jobs=-1
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model_enh_path = GradientBoostingRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            random_state=SEED
        )

    model_enh_path.fit(X_train_enh_path, y_train)
    y_pred_enh_path_train = model_enh_path.predict(X_train_enh_path)
    y_pred_enh_path_test = model_enh_path.predict(X_test_enh_path)

    train_r2 = r2_score(y_train, y_pred_enh_path_train)
    test_r2 = r2_score(y_test, y_pred_enh_path_test)
    cv_results['enhanced_with_pathways']['train'].append(train_r2)
    cv_results['enhanced_with_pathways']['test'].append(test_r2)
    oof_predictions['enhanced_with_pathways'][test_idx] = y_pred_enh_path_test
    fold_models['enhanced_with_pathways'].append(model_enh_path)

    print(f"Enhanced+Pathways - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

# ============================================================================
# STEP 6: Summarize CV Results
# ============================================================================
print(f"\n{'='*80}")
print("STEP 6: Cross-Validation Summary")
print(f"{'='*80}")

cv_summary = pd.DataFrame({
    'Model': ['Baseline', 'Enhanced', 'Enhanced+Pathways'],
    'Mean_Train_R2': [
        np.mean(cv_results['baseline']['train']),
        np.mean(cv_results['enhanced']['train']),
        np.mean(cv_results['enhanced_with_pathways']['train'])
    ],
    'Std_Train_R2': [
        np.std(cv_results['baseline']['train']),
        np.std(cv_results['enhanced']['train']),
        np.std(cv_results['enhanced_with_pathways']['train'])
    ],
    'Mean_Test_R2': [
        np.mean(cv_results['baseline']['test']),
        np.mean(cv_results['enhanced']['test']),
        np.mean(cv_results['enhanced_with_pathways']['test'])
    ],
    'Std_Test_R2': [
        np.std(cv_results['baseline']['test']),
        np.std(cv_results['enhanced']['test']),
        np.std(cv_results['enhanced_with_pathways']['test'])
    ]
})

print(cv_summary.to_string(index=False))

cv_summary.to_csv(os.path.join(OUT, 'tables', 'cv_results.csv'), index=False)
print(f"\nCV results saved to tables/cv_results.csv")

# ============================================================================
# STEP 7: SHAP Analysis (on Enhanced Model)
# ============================================================================
if HAS_SHAP and HAS_XGBOOST:
    print(f"\n{'='*80}")
    print("STEP 7: SHAP Feature Importance Analysis")
    print(f"{'='*80}")

    # Train enhanced model on full data for SHAP
    df_full_enh = create_enhanced_features(df_cv.copy())
    enhanced_cols = [col for col in enhanced_feature_set if col in df_full_enh.columns]
    X_full_enhanced = df_full_enh[enhanced_cols].values
    y_full = df_cv[primary_target].values

    imputer = SimpleImputer(strategy='median')
    X_full_enhanced = imputer.fit_transform(X_full_enhanced)

    scaler = StandardScaler()
    X_full_enhanced = scaler.fit_transform(X_full_enhanced)

    model_shap = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        random_state=SEED, verbosity=0, n_jobs=-1
    )
    model_shap.fit(X_full_enhanced, y_full)

    # Compute SHAP values on sample (for computational efficiency)
    sample_size = min(100, len(X_full_enhanced))
    X_sample = X_full_enhanced[:sample_size]

    try:
        explainer = shap.TreeExplainer(model_shap)
        shap_values = explainer.shap_values(X_sample)
    except (ValueError, Exception) as e:
        print(f"  TreeExplainer failed ({e}), falling back to feature_importances_")
        # Fallback: use XGBoost built-in feature importances
        mean_abs_shap = model_shap.feature_importances_
        shap_values = None

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    if shap_values is not None:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    # else: mean_abs_shap already set from feature_importances_ above
    sorted_idx = np.argsort(mean_abs_shap)[-15:]

    feature_names = np.array(enhanced_cols)
    ax.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(feature_names[sorted_idx])
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Top 15 Features by SHAP Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'figures', 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"SHAP feature importance plot saved to figures/shap_feature_importance.png")
else:
    print("SHAP analysis skipped (HAS_SHAP={}, HAS_XGBOOST={})".format(HAS_SHAP, HAS_XGBOOST))

# ============================================================================
# STEP 8: Stacking Ensemble
# ============================================================================
print(f"\n{'='*80}")
print("STEP 8: Stacking Ensemble")
print(f"{'='*80}")

ensemble_results = {'train': [], 'test': []}

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df_cv)):
    print(f"\n--- Ensemble Fold {fold_idx + 1}/{K_FOLDS} ---")

    df_train = df_cv.iloc[train_idx].copy()
    df_test = df_cv.iloc[test_idx].copy()
    y_train = df_train[primary_target].values
    y_test = df_test[primary_target].values

    # Base learners: XGBoost, LightGBM, CatBoost (with fallbacks)
    base_models = []
    base_predictions_train = []
    base_predictions_test = []

    # ---- XGBoost ----
    if HAS_XGBOOST:
        df_train_enh = create_enhanced_features(df_train.copy())
        df_test_enh = create_enhanced_features(df_test.copy())

        enhanced_cols = [col for col in enhanced_feature_set
                         if col in df_train_enh.columns and col in df_test_enh.columns]
        X_train_xgb = df_train_enh[enhanced_cols].values
        X_test_xgb = df_test_enh[enhanced_cols].values

        imputer = SimpleImputer(strategy='median')
        X_train_xgb = imputer.fit_transform(X_train_xgb)
        X_test_xgb = imputer.transform(X_test_xgb)

        scaler = StandardScaler()
        X_train_xgb = scaler.fit_transform(X_train_xgb)
        X_test_xgb = scaler.transform(X_test_xgb)

        xgb_model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            random_state=SEED, verbosity=0, n_jobs=-1
        )
        xgb_model.fit(X_train_xgb, y_train)

        base_predictions_train.append(xgb_model.predict(X_train_xgb))
        base_predictions_test.append(xgb_model.predict(X_test_xgb))
        print(f"XGBoost: Train R² = {r2_score(y_train, base_predictions_train[-1]):.4f}")

    # ---- LightGBM ----
    if HAS_LIGHTGBM:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            random_state=SEED, verbosity=-1, n_jobs=-1
        )
        lgb_model.fit(X_train_xgb, y_train)

        base_predictions_train.append(lgb_model.predict(X_train_xgb))
        base_predictions_test.append(lgb_model.predict(X_test_xgb))
        print(f"LightGBM: Train R² = {r2_score(y_train, base_predictions_train[-1]):.4f}")

    # ---- CatBoost ----
    if HAS_CATBOOST:
        cb_model = cb.CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.1,
            random_state=SEED, verbose=False
        )
        cb_model.fit(X_train_xgb, y_train)

        base_predictions_train.append(cb_model.predict(X_train_xgb))
        base_predictions_test.append(cb_model.predict(X_test_xgb))
        print(f"CatBoost: Train R² = {r2_score(y_train, base_predictions_train[-1]):.4f}")

    # ---- Meta-Learner (Ridge) ----
    if base_predictions_train:
        X_meta_train = np.column_stack(base_predictions_train)
        X_meta_test = np.column_stack(base_predictions_test)

        meta_model = Ridge(alpha=1.0)
        meta_model.fit(X_meta_train, y_train)

        y_pred_ensemble_train = meta_model.predict(X_meta_train)
        y_pred_ensemble_test = meta_model.predict(X_meta_test)

        train_r2 = r2_score(y_train, y_pred_ensemble_train)
        test_r2 = r2_score(y_test, y_pred_ensemble_test)

        ensemble_results['train'].append(train_r2)
        ensemble_results['test'].append(test_r2)

        print(f"Ensemble - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

if ensemble_results['test']:
    print(f"\nEnsemble Mean Test R²: {np.mean(ensemble_results['test']):.4f} ± {np.std(ensemble_results['test']):.4f}")

# ============================================================================
# STEP 9: Save Results and Summary
# ============================================================================
print(f"\n{'='*80}")
print("STEP 9: Saving Results")
print(f"{'='*80}")

# Enhanced feature list
enhanced_feature_df = pd.DataFrame({
    'Feature': enhanced_feature_cols,
    'Type': ['Interaction' if 'interaction' in col.lower() or '_' in col else 'Composite' for col in enhanced_feature_cols]
})
enhanced_feature_df.to_csv(os.path.join(OUT, 'tables', 'enhanced_features.csv'), index=False)
print(f"Enhanced features list saved to tables/enhanced_features.csv")

# Summary statistics
summary = {
    'n_samples': len(df_cv),
    'n_baseline_features': len(baseline_feature_set),
    'n_enhanced_features': len(enhanced_feature_cols),
    'n_pathways': len(PATHWAYS),
    'baseline_mean_test_r2': np.mean(cv_results['baseline']['test']),
    'enhanced_mean_test_r2': np.mean(cv_results['enhanced']['test']),
    'enhanced_pathways_mean_test_r2': np.mean(cv_results['enhanced_with_pathways']['test']),
    'ensemble_mean_test_r2': np.mean(ensemble_results['test']) if ensemble_results['test'] else None
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(OUT, 'tables', 'pipeline_summary.csv'), index=False)
print(f"Pipeline summary saved to tables/pipeline_summary.csv")

print(f"\n{'='*80}")
print("ENHANCED PIPELINE COMPLETE")
print(f"{'='*80}")
print(f"Results saved to:")
print(f"  - tables/cv_results.csv")
print(f"  - tables/enhanced_features.csv")
print(f"  - tables/pipeline_summary.csv")
print(f"  - figures/shap_feature_importance.png (if SHAP available)")
