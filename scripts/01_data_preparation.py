#!/usr/bin/env python3
"""
Multi-Modal Blood Pressure Prediction — Data Preparation Pipeline (v2 CORRECTED)
==================================================================================
CRITICAL FIXES from v1:
  1. BP-derived variability features (SBP_std, DBP_std, PR_std, SBP_range,
     DBP_range, Pulse_Pressure) are EXCLUDED from the leak-free feature set.
  2. Feature counts are accurate (75 labs with >70% missing threshold).
  3. Feature groups are correctly defined and verified.

Merges 4 data modalities:
  1. Clinical measurements (anthropometrics, demographics)
  2. Clinical labs (~75 blood tests after filtering)
  3. Polygenic Risk Scores (12 PRS)
  4. Ancestry Principal Components (10 PCs)

Outputs:
  - merged_dataset.tsv
  - feature_groups.json
  - data_summary.json
"""

import pandas as pd
import numpy as np
import json
import os

# =====================================================
# Path Setup
# =====================================================
OUT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(OUT), 'QBB_data')
if not os.path.isdir(BASE):
    BASE = '/mnt/e/genomics/QBB_data'

print("=" * 70)
print("MULTI-MODAL BP PREDICTION — DATA PREPARATION (v2 CORRECTED)")
print("=" * 70)
print(f"\nData path: {BASE}")
print(f"Output path: {OUT}")

os.makedirs(f'{OUT}/figures', exist_ok=True)
os.makedirs(f'{OUT}/tables', exist_ok=True)

# =====================================================
# STEP 1: Load all data modalities
# =====================================================
print("\n" + "=" * 70)
print("STEP 1: Loading all data modalities")
print("=" * 70)

# --- 1. Clinical Measurements ---
print("\n[1/4] Loading clinical measurements...")
meas = pd.read_csv(f'{BASE}/processed_14k_measurments.tsv', sep='\t')
meas = meas.rename(columns={'IID': 'sid'})
print(f"  Measurements: {meas.shape[0]} participants, {meas.shape[1]} columns")

# Select useful measurement features
meas_features = [
    'BMI', 'BP_OUT_CALC_AVG_DIASTOLIC_BP', 'BP_OUT_CALC_AVG_SYSTOLIC_BP',
    'BP_OUT_CALC_AVG_PULSE_RATE',
    'BP_OUT_DIASTOLIC_BP_1', 'BP_OUT_DIASTOLIC_BP_2', 'BP_OUT_DIASTOLIC_BP_3',
    'BP_OUT_SYSTOLIC_BP_1', 'BP_OUT_SYSTOLIC_BP_2', 'BP_OUT_SYSTOLIC_BP_3',
    'BP_OUT_PULSE_RATE_1', 'BP_OUT_PULSE_RATE_2', 'BP_OUT_PULSE_RATE_3',
    'HANDGRIP_OUT_LEFT', 'HANDGRIP_OUT_RIGHT',
    'HEIGHTWEIGHT_OUT_SITTING_HEIGHT', 'HEIGHTWEIGHT_OUT_STANDING_HEIGHT',
    'HEIGHTWEIGHT_OUT_WEIGHT',
    'HIPWAIST_OUT_CALC_WAIST_TO_HIP_RATIO', 'HIPWAIST_OUT_HIPS_SIZE',
    'HIPWAIST_OUT_WAIST_SIZE',
    'mean_arterial_pressure', 'hypertension',
    'sex', 'age'
]
meas_avail = [c for c in meas_features if c in meas.columns]
meas_df = meas[['sid'] + meas_avail].copy()
for col in meas_avail:
    meas_df[col] = pd.to_numeric(meas_df[col], errors='coerce')
print(f"  Selected {len(meas_avail)} measurement features")

# --- 2. Clinical Lab Results ---
print("\n[2/4] Loading clinical lab results...")
labs = pd.read_csv(f'{BASE}/processed_14k_lab_results.tsv', sep='\t')
labs = labs.rename(columns={'encparticipantid': 'sid'})
print(f"  Labs: {labs.shape[0]} participants, {labs.shape[1]} columns")

lab_features = [c for c in labs.columns if c != 'sid']
for col in lab_features:
    labs[col] = labs[col].replace('.', np.nan)
    labs[col] = pd.to_numeric(labs[col], errors='coerce')

# Drop columns with >70% missing (this is the ACTUAL threshold)
missing_pct = labs[lab_features].isnull().mean()
keep_labs = missing_pct[missing_pct < 0.70].index.tolist()
drop_labs = missing_pct[missing_pct >= 0.70].index.tolist()
labs_df = labs[['sid'] + keep_labs].copy()
labs_df.columns = ['sid'] + [f'LAB_{c}' for c in keep_labs]
print(f"  Kept {len(keep_labs)} lab features (dropped {len(drop_labs)} with >70% missing)")

# --- 3. Polygenic Risk Scores ---
print("\n[3/4] Loading PRS scores...")
prs_files = sorted([f for f in os.listdir(f'{BASE}/PRS') if f.endswith('.txt')])
prs_dfs = []
for f in prs_files:
    pgs_id = f.replace('QBB_smaller_onlysnps.', '').replace('.txt', '')
    df = pd.read_csv(f'{BASE}/PRS/{f}', sep='\t')
    df = df.rename(columns={'#IID': 'prs_iid', 'SCORE1_AVG': f'PRS_{pgs_id}'})
    df['sid'] = df['prs_iid'].str.split('_').str[0]
    df = df.drop_duplicates(subset='sid', keep='first')
    prs_dfs.append(df[['sid', f'PRS_{pgs_id}']])
    print(f"  Loaded {pgs_id}: {len(df)} participants")

prs_df = prs_dfs[0]
for df in prs_dfs[1:]:
    prs_df = prs_df.merge(df, on='sid', how='inner')
print(f"  Combined PRS: {prs_df.shape[0]} participants, {prs_df.shape[1]-1} scores")

# --- 4. Principal Components ---
print("\n[4/4] Loading PCA components...")
pca = pd.read_csv(f'{BASE}/pca/14kpca_onlysnps_eigen.eigenvec.txt', sep='\t')
pca = pca.rename(columns={'#FID': 'sid'})
pc_cols = [c for c in pca.columns if c.startswith('PC')]
pca_df = pca[['sid'] + pc_cols].copy()
print(f"  PCA: {pca_df.shape[0]} participants, {len(pc_cols)} PCs")

# =====================================================
# STEP 2: Merge all modalities
# =====================================================
print("\n" + "=" * 70)
print("STEP 2: Merging all modalities")
print("=" * 70)

merged = meas_df.copy()
print(f"\n  Base (measurements): {len(merged)} participants")

merged = merged.merge(labs_df, on='sid', how='left')
print(f"  + Labs: {len(merged)} participants")

merged = merged.merge(prs_df, on='sid', how='inner')
print(f"  + PRS: {len(merged)} participants")

merged = merged.merge(pca_df, on='sid', how='inner')
print(f"  + PCA: {len(merged)} participants")

# =====================================================
# STEP 3: Feature engineering
# =====================================================
print("\n" + "=" * 70)
print("STEP 3: Feature engineering & leak-free group definition")
print("=" * 70)

target_continuous = 'mean_arterial_pressure'
target_binary = 'hypertension'

# --- BP variability features (THESE ARE BP-DERIVED AND WILL BE EXCLUDED FROM PRIMARY ANALYSIS) ---
sbp_cols = ['BP_OUT_SYSTOLIC_BP_1', 'BP_OUT_SYSTOLIC_BP_2', 'BP_OUT_SYSTOLIC_BP_3']
dbp_cols = ['BP_OUT_DIASTOLIC_BP_1', 'BP_OUT_DIASTOLIC_BP_2', 'BP_OUT_DIASTOLIC_BP_3']
pr_cols = ['BP_OUT_PULSE_RATE_1', 'BP_OUT_PULSE_RATE_2', 'BP_OUT_PULSE_RATE_3']

if all(c in merged.columns for c in sbp_cols):
    merged['SBP_std'] = merged[sbp_cols].std(axis=1)
    merged['DBP_std'] = merged[dbp_cols].std(axis=1)
    merged['PR_std'] = merged[pr_cols].std(axis=1)
    merged['SBP_range'] = merged[sbp_cols].max(axis=1) - merged[sbp_cols].min(axis=1)
    merged['DBP_range'] = merged[dbp_cols].max(axis=1) - merged[dbp_cols].min(axis=1)
    merged['Pulse_Pressure'] = merged['BP_OUT_CALC_AVG_SYSTOLIC_BP'] - merged['BP_OUT_CALC_AVG_DIASTOLIC_BP']
    print("  Created BP variability features (SBP_std, DBP_std, PR_std, ranges, pulse pressure)")
    print("  *** NOTE: These are BP-DERIVED and will be EXCLUDED from leak-free sets ***")

# --- Anthropometric ratios (SAFE — not derived from BP) ---
if 'HIPWAIST_OUT_WAIST_SIZE' in merged.columns and 'HEIGHTWEIGHT_OUT_STANDING_HEIGHT' in merged.columns:
    merged['Waist_Height_Ratio'] = merged['HIPWAIST_OUT_WAIST_SIZE'] / merged['HEIGHTWEIGHT_OUT_STANDING_HEIGHT']
    print("  Created waist-to-height ratio (SAFE)")

if 'HANDGRIP_OUT_LEFT' in merged.columns and 'HANDGRIP_OUT_RIGHT' in merged.columns:
    merged['Grip_Asymmetry'] = (merged['HANDGRIP_OUT_RIGHT'] - merged['HANDGRIP_OUT_LEFT']).abs()
    merged['Grip_Mean'] = (merged['HANDGRIP_OUT_RIGHT'] + merged['HANDGRIP_OUT_LEFT']) / 2
    print("  Created grip strength features (SAFE)")

# =====================================================
# STEP 4: Define feature groups
# =====================================================
all_features = [c for c in merged.columns if c not in ['sid', target_continuous, target_binary]]

# Modality groups
lab_cols_final = [c for c in all_features if c.startswith('LAB_')]
prs_cols_final = [c for c in all_features if c.startswith('PRS_')]
pc_cols_final = [c for c in all_features if c.startswith('PC')]

# BP reading columns (raw readings — definitely leaked)
bp_reading_cols = [c for c in all_features if 'BP_OUT' in c]

# BP-derived engineered features (derived from SBP/DBP readings — also leaked)
bp_derived_cols = ['SBP_std', 'DBP_std', 'PR_std', 'SBP_range', 'DBP_range', 'Pulse_Pressure']
bp_derived_cols = [c for c in bp_derived_cols if c in all_features]

# ALL columns to exclude from leak-free analysis
bp_leak_set = set(bp_reading_cols + bp_derived_cols)

# Clinical features: everything that's not labs, PRS, PCA, or BP-leaked
clinical_leak_free = [c for c in all_features
                      if c not in lab_cols_final
                      and c not in prs_cols_final
                      and c not in pc_cols_final
                      and c not in bp_leak_set]

# Combined leak-free set
all_leak_free = clinical_leak_free + lab_cols_final + prs_cols_final + pc_cols_final

# Secondary analysis set (with BP variability)
all_with_bp_variability = clinical_leak_free + bp_derived_cols + lab_cols_final + prs_cols_final + pc_cols_final

# =====================================================
# SAFETY ASSERTION — the CRITICAL fix
# =====================================================
print("\n  *** SAFETY ASSERTION ***")
for feat in clinical_leak_free:
    assert 'BP_OUT' not in feat, f"LEAK: {feat} in clinical_leak_free!"
    assert feat not in bp_derived_cols, f"LEAK: {feat} in clinical_leak_free!"

for feat in all_leak_free:
    assert 'BP_OUT' not in feat, f"LEAK: {feat} in all_leak_free!"
    assert feat not in bp_derived_cols, f"LEAK: {feat} in all_leak_free!"

# Extra paranoia: check for any feature that looks BP-related
bp_suspects = [c for c in all_leak_free
               if any(x in c.lower() for x in ['systolic', 'diastolic', 'pulse_pressure'])]
assert len(bp_suspects) == 0, f"LEAK: BP-related features found: {bp_suspects}"

print("  PASSED: No BP-related features in leak-free sets")

print(f"\n  Feature group counts:")
print(f"    Clinical (leak-free): {len(clinical_leak_free)}")
print(f"    Labs:                 {len(lab_cols_final)}")
print(f"    PRS:                  {len(prs_cols_final)}")
print(f"    PCA:                  {len(pc_cols_final)}")
print(f"    ---")
print(f"    Total leak-free:      {len(all_leak_free)}")
print(f"    BP readings excluded: {len(bp_reading_cols)}")
print(f"    BP derived excluded:  {len(bp_derived_cols)}")
print(f"\n  Clinical leak-free features:")
for f in clinical_leak_free:
    print(f"    - {f}")

# =====================================================
# STEP 5: Save outputs
# =====================================================
print("\n" + "=" * 70)
print("STEP 5: Saving outputs")
print("=" * 70)

# Drop rows with missing targets
before = len(merged)
merged = merged.dropna(subset=[target_continuous])
print(f"  Dropped {before - len(merged)} rows with missing MAP")
before = len(merged)
merged = merged.dropna(subset=[target_binary])
print(f"  Dropped {before - len(merged)} rows with missing hypertension")

# Feature groups dict
feature_groups = {
    'bp_readings': bp_reading_cols,
    'bp_derived': bp_derived_cols,
    'clinical_leak_free': clinical_leak_free,
    'labs': lab_cols_final,
    'prs': prs_cols_final,
    'pca': pc_cols_final,
    'all_leak_free': all_leak_free,
    'all_with_bp_variability': all_with_bp_variability,
}

# Data summary
summary = {
    'n_participants': int(merged.shape[0]),
    'n_clinical_leak_free': len(clinical_leak_free),
    'n_labs': len(lab_cols_final),
    'n_prs': len(prs_cols_final),
    'n_pca': len(pc_cols_final),
    'n_total_leak_free': len(all_leak_free),
    'n_bp_derived_excluded': len(bp_derived_cols),
    'target_MAP_mean': float(merged[target_continuous].mean()),
    'target_MAP_std': float(merged[target_continuous].std()),
    'target_HTN_prevalence': float(merged[target_binary].mean()),
    'missing_pct_labs': float(merged[lab_cols_final].isnull().mean().mean() * 100),
    'missing_pct_clinical': float(merged[clinical_leak_free].isnull().mean().mean() * 100),
    'missing_pct_prs': float(merged[prs_cols_final].isnull().mean().mean() * 100),
    'missing_pct_pca': float(merged[pc_cols_final].isnull().mean().mean() * 100),
    'lab_missing_threshold': '70%',
}

merged.to_csv(f'{OUT}/merged_dataset.tsv', sep='\t', index=False)
print(f"  Saved merged_dataset.tsv: {merged.shape[0]} × {merged.shape[1]}")

with open(f'{OUT}/feature_groups.json', 'w') as f:
    json.dump(feature_groups, f, indent=2)
print(f"  Saved feature_groups.json")

with open(f'{OUT}/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved data_summary.json")

# Final summary
print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETE")
print("=" * 70)
print(f"\n  Dataset: {merged.shape[0]:,} participants × {merged.shape[1]} columns")
print(f"  MAP: {summary['target_MAP_mean']:.1f} ± {summary['target_MAP_std']:.1f} mmHg")
print(f"  HTN prevalence: {summary['target_HTN_prevalence']:.1%}")
print(f"\n  Leak-free features: {len(all_leak_free)}")
print(f"    Clinical: {len(clinical_leak_free)}")
print(f"    Labs: {len(lab_cols_final)} (threshold: >70% missing dropped)")
print(f"    PRS: {len(prs_cols_final)}")
print(f"    PCA: {len(pc_cols_final)}")
print(f"\n  Lab missingness: {summary['missing_pct_labs']:.1f}% average")
print(f"  Clinical missingness: {summary['missing_pct_clinical']:.1f}% average")
print(f"\n  EXCLUDED from primary analysis:")
print(f"    - {len(bp_reading_cols)} raw BP readings")
print(f"    - {len(bp_derived_cols)} BP-derived features: {bp_derived_cols}")
print("\n✓ Data preparation complete!")
