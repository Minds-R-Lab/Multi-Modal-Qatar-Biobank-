# Multi-Modal Machine Learning for Blood Pressure Prediction in 14,383 Qatar Biobank Participants: Integrating Clinical, Laboratory, Polygenic Risk, and Population-Structure Data

[![Paper](https://img.shields.io/badge/Journal-Artificial%20Intelligence%20in%20Medicine-blue)](https://www.sciencedirect.com/journal/artificial-intelligence-in-medicine)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code and results for the paper:

> **Multi-Modal Machine Learning for Blood Pressure Prediction in 14,383 Qatar Biobank Participants: Integrating Clinical, Laboratory, Polygenic Risk, and Population-Structure Data**
>
> Mohamed A. Mabrok, Rana Aldisi, Hatem Zayed*
>
> *Submitted to Artificial Intelligence in Medicine*

---


## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── config/
│   ├── data_summary.json           # Dataset statistics (N, feature counts, missingness)
│   └── feature_groups.json         # Feature-to-modality mapping
├── scripts/
│   ├── 01_data_preparation.py      # Data loading, QC, feature grouping
│   ├── 02_full_ml_pipeline.py      # 8-model benchmark (5-fold CV) + SHAP importance
│   ├── 03_enhanced_pipeline.py     # Feature engineering + stacking ensemble
│   ├── 04_statistical_tests.py     # Pairwise model comparisons + confidence intervals
│   ├── 05_sensitivity_analysis.py  # MAP-threshold sensitivity for classification
│   ├── 06_deep_models_gpu.py       # FT-Transformer + SAINT (GPU implementation)
│   └── 07_regenerate_all_figures.py # Regenerate all publication figures
└── results/
    ├── tables/
    │   ├── fold_results_regression.tsv       # Per-fold R², RMSE, MAE for all 10 models
    │   ├── fold_results_classification.tsv   # Per-fold AUROC for classification
    │   ├── fold_results_ablation.tsv         # Per-fold ablation results (8 configurations)
    │   ├── model_comparison_regression.tsv   # Summary statistics for regression
    │   ├── model_comparison_classification.tsv # Summary statistics for classification
    │   ├── confidence_intervals.tsv          # 95% CIs for all models
    │   ├── statistical_tests_models.tsv      # Pairwise p-values (t-test, Wilcoxon, Bonferroni)
    │   ├── statistical_tests_ablation.tsv    # Ablation pairwise comparisons
    │   ├── statistical_tests_classification.tsv # Classification pairwise comparisons
    │   ├── ablation_regression.tsv           # Ablation summary (R², RMSE)
    │   ├── ablation_classification.tsv       # Ablation summary (AUROC)
    │   ├── shap_importance.tsv               # Gain-based feature importance (111 features)
    │   ├── cv_results.csv                    # Enhanced pipeline cross-validation results
    │   ├── enhanced_features.csv             # Engineered feature definitions
    │   ├── pipeline_summary.csv              # Baseline vs. enhanced vs. stacking summary
    │   └── sensitivity_map_thresholds.tsv    # Sensitivity analysis across MAP thresholds
    └── figures/
        ├── fig1_modality_ablation_r2.png     # Ablation analysis (R²)
        ├── fig2_modality_ablation_auroc.png  # Ablation analysis (AUROC)
        ├── fig3_model_comparison.png         # 10-model comparison bar chart
        ├── fig_shap_summary.png              # Top 20 feature importance (horizontal bar)
        ├── fig_shap_modality_pie.png         # Modality contribution (pie + bar)
        ├── fig6_predicted_vs_actual.png      # Predicted vs. actual MAP scatter
        ├── fig_sensitivity_thresholds.png    # Sensitivity analysis across thresholds
        ├── fig_statistical_significance.png  # Pairwise p-value heatmap
        ├── fig_pca_population_structure.png  # PCA population structure visualization
        └── shap_feature_importance.png       # Full SHAP importance bar chart
```

---


