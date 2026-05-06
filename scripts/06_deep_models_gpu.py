#!/usr/bin/env python3
"""
06_deep_models_gpu.py
=====================
Trains REAL FT-Transformer and SAINT models on the QBB blood pressure dataset
using PyTorch with GPU/CUDA support.

This script replaces the MLP-proxy results in the main pipeline with genuine
deep tabular architectures. Results are saved in the same format as the main
pipeline so they can be merged directly.

Requirements:
    pip install torch pandas numpy scikit-learn

Usage:
    python 06_deep_models_gpu.py

Output:
    tables/deep_model_fold_results_regression.tsv
    tables/deep_model_fold_results_classification.tsv
    tables/deep_model_summary.tsv
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION — must match the main pipeline exactly
# =====================================================================
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / 'merged_dataset.tsv'
FEATURE_GROUPS_FILE = BASE_DIR / 'feature_groups.json'
TABLES_DIR = BASE_DIR / 'tables'
TABLES_DIR.mkdir(exist_ok=True)

K_FOLDS = 5
SEED = 42
TARGET_REG = 'mean_arterial_pressure'
TARGET_CLS = 'hypertension'

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# FT-TRANSFORMER ARCHITECTURE (Gorishniy et al., 2021)
# =====================================================================
class NumericalEmbedding(nn.Module):
    """Embeds each numerical feature into a d-dimensional vector.

    Following the FT-Transformer paper: each feature x_j is mapped to
    e_j = x_j * w_j + b_j, where w_j, b_j are learnable d-dimensional vectors.
    """
    def __init__(self, n_features, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        # Xavier-style initialization
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (batch, n_features) -> (batch, n_features, d_model)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer (Gorishniy et al., 2021).
    Each numerical feature is projected to a d-dimensional token,
    a [CLS] token is prepended, and a standard Transformer encoder
    processes the sequence. The [CLS] output is used for prediction.
    """
    def __init__(self, n_features, d_model=192, n_heads=8, n_layers=3,
                 d_ff=768, dropout=0.2, output_dim=1, task='regression'):
        super().__init__()
        self.task = task
        self.n_features = n_features
        self.d_model = d_model

        # Feature tokenizer: each feature -> d_model vector
        self.feature_embedding = NumericalEmbedding(n_features, d_model)

        # [CLS] token (learnable)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer encoder with pre-norm (better for tabular)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final layer norm + prediction head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, output_dim)
        )

        # Initialize head
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, n_features)
        batch_size = x.size(0)

        # Tokenize features: (batch, n_features, d_model)
        tokens = self.feature_embedding(x)

        # Prepend [CLS] token: (batch, 1+n_features, d_model)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Transformer
        tokens = self.transformer(tokens)

        # Use [CLS] output for prediction
        cls_out = self.norm(tokens[:, 0, :])
        out = self.head(cls_out)

        if self.task == 'classification':
            out = torch.sigmoid(out)

        return out.squeeze(-1)


# =====================================================================
# SAINT ARCHITECTURE (Somepalli et al., 2021)
# =====================================================================
class SAINTAttention(nn.Module):
    """Self-attention across feature tokens (column-wise attention)."""
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = x + residual

        # Pre-norm feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ff1(x) + residual
        return x


class IntersampleAttention(nn.Module):
    """Attention across samples (rows) for each feature position.

    This is the key innovation of SAINT: for each feature token position,
    attend across all samples in the batch to capture inter-sample relationships.
    """
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, n_tokens, d_model) -> transpose to (n_tokens, batch, d_model)
        x = x.transpose(0, 1)  # (n_tokens, batch, d_model)
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x) + residual
        x = x.transpose(0, 1)  # back to (batch, n_tokens, d_model)
        return x


class SAINT(nn.Module):
    """
    SAINT: Self-Attention and Intersample Attention Transformer
    (Somepalli et al., 2021).

    Alternates between:
      1) Self-attention across features (column attention)
      2) Intersample attention across rows (row attention)
    """
    def __init__(self, n_features, d_model=192, n_heads=8, n_layers=3,
                 d_ff=768, dropout=0.2, output_dim=1, task='regression'):
        super().__init__()
        self.task = task

        # Feature embedding
        self.feature_embedding = NumericalEmbedding(n_features, d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Alternating self-attention and intersample attention layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'self_attn': SAINTAttention(d_model, n_heads, dropout),
                'intersample_attn': IntersampleAttention(d_model, n_heads, dropout),
            }))

        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, output_dim)
        )

        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.size(0)

        # Tokenize
        tokens = self.feature_embedding(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Alternating attention layers
        for layer in self.layers:
            tokens = layer['self_attn'](tokens)
            tokens = layer['intersample_attn'](tokens)

        # [CLS] output
        cls_out = self.norm(tokens[:, 0, :])
        out = self.head(cls_out)

        if self.task == 'classification':
            out = torch.sigmoid(out)

        return out.squeeze(-1)


# =====================================================================
# TRAINING UTILITIES
# =====================================================================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_deep_model(model, X_train, y_train, X_val, y_val,
                     task='regression', epochs=200, batch_size=256,
                     lr=1e-3, weight_decay=1e-5):
    """
    Train a deep model with early stopping and cosine annealing.

    IMPORTANT: For regression, BOTH y_train and y_val must be on the
    SAME scale (either both scaled or both original). The caller is
    responsible for ensuring this.
    """

    model = model.to(DEVICE)

    # Create tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)

    if task == 'regression':
        y_train_t = torch.FloatTensor(y_train).to(DEVICE)
        y_val_t = torch.FloatTensor(y_val).to(DEVICE)
        criterion = nn.MSELoss()
    else:
        y_train_t = torch.FloatTensor(y_train).to(DEVICE)
        y_val_t = torch.FloatTensor(y_val).to(DEVICE)
        criterion = nn.BCELoss()

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    early_stopping = EarlyStopping(patience=25)
    best_state = None
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"    Early stopping at epoch {epoch+1} (best val_loss: {best_val_loss:.4f})")
            break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, train_loss: {train_loss/n_batches:.4f}, val_loss: {val_loss:.4f}")

    else:
        print(f"    Completed all {epochs} epochs (best val_loss: {best_val_loss:.4f})")

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    return model


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("=" * 70)
    print("DEEP TABULAR MODELS -- FT-Transformer & SAINT (GPU)")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/4] Loading data...")
    df = pd.read_csv(DATA_FILE, sep='\t')
    with open(FEATURE_GROUPS_FILE, 'r') as f:
        feature_groups = json.load(f)

    feature_cols = feature_groups['all_leak_free']
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"  Samples: {len(df)}")
    print(f"  Leak-free features: {len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    y_reg = df[TARGET_REG].values.astype(np.float32)
    y_cls = df[TARGET_CLS].values.astype(np.float32)

    # --- Cross-validation (must match main pipeline exactly) ---
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    # Model configs
    model_configs = {
        'FTTransformer': {
            'class': FTTransformer,
            'params': {
                'd_model': 192,
                'n_heads': 8,
                'n_layers': 3,
                'd_ff': 768,       # 4x d_model (standard)
                'dropout': 0.2,
            },
            'train_params': {
                'epochs': 200,
                'batch_size': 256,
                'lr': 1e-3,        # Higher LR with cosine annealing
                'weight_decay': 1e-5,
            }
        },
        'SAINT': {
            'class': SAINT,
            'params': {
                'd_model': 192,
                'n_heads': 8,
                'n_layers': 3,
                'd_ff': 768,
                'dropout': 0.2,
            },
            'train_params': {
                'epochs': 200,
                'batch_size': 256,
                'lr': 1e-3,
                'weight_decay': 1e-5,
            }
        }
    }

    # Storage
    reg_results = {'fold': list(range(1, K_FOLDS + 1))}
    cls_results = {'fold': list(range(1, K_FOLDS + 1))}

    for model_name, config in model_configs.items():
        reg_results[f'{model_name}_R2'] = []
        reg_results[f'{model_name}_RMSE'] = []
        reg_results[f'{model_name}_MAE'] = []
        cls_results[f'{model_name}_AUROC'] = []

    # --- Run cross-validation ---
    print(f"\n[2/4] Running {K_FOLDS}-fold cross-validation...")

    for model_name, config in model_configs.items():
        print(f"\n{'='*50}")
        print(f"  Model: {model_name}")
        print(f"{'='*50}")

        ModelClass = config['class']
        model_params = config['params']
        train_params = config['train_params']

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"\n  Fold {fold_idx + 1}/{K_FOLDS}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train_reg, y_test_reg = y_reg[train_idx], y_reg[test_idx]
            y_train_cls, y_test_cls = y_cls[train_idx], y_cls[test_idx]

            # Preprocessing (same as main pipeline)
            imputer = SimpleImputer(strategy='median')
            scaler = StandardScaler()

            X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
            X_test_proc = scaler.transform(imputer.transform(X_test))

            # Scale target for regression (helps deep models converge)
            target_scaler = StandardScaler()
            y_train_scaled = target_scaler.fit_transform(
                y_train_reg.reshape(-1, 1)
            ).ravel()
            # FIX: Scale validation targets with SAME scaler so loss is comparable
            y_test_scaled = target_scaler.transform(
                y_test_reg.reshape(-1, 1)
            ).ravel()

            n_features = X_train_proc.shape[1]

            # --- REGRESSION ---
            print(f"    Training {model_name} (regression)...")

            model_reg = ModelClass(
                n_features=n_features,
                output_dim=1,
                task='regression',
                **model_params
            )

            # FIX: Both train and val targets on same scale
            model_reg = train_deep_model(
                model_reg, X_train_proc, y_train_scaled,
                X_test_proc, y_test_scaled,
                task='regression',
                **train_params
            )

            # Get predictions and inverse-transform to original scale
            model_reg.eval()
            with torch.no_grad():
                X_test_t = torch.FloatTensor(X_test_proc).to(DEVICE)
                y_pred_scaled = model_reg(X_test_t).cpu().numpy()
            y_pred_reg = target_scaler.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()

            r2 = r2_score(y_test_reg, y_pred_reg)
            rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
            mae = mean_absolute_error(y_test_reg, y_pred_reg)

            reg_results[f'{model_name}_R2'].append(r2)
            reg_results[f'{model_name}_RMSE'].append(rmse)
            reg_results[f'{model_name}_MAE'].append(mae)
            print(f"    Regression -- R2: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

            # --- CLASSIFICATION ---
            print(f"    Training {model_name} (classification)...")
            model_cls = ModelClass(
                n_features=n_features,
                output_dim=1,
                task='classification',
                **model_params
            )

            model_cls = train_deep_model(
                model_cls, X_train_proc, y_train_cls,
                X_test_proc, y_test_cls,
                task='classification',
                **train_params
            )

            # Get predictions
            model_cls.eval()
            with torch.no_grad():
                y_pred_proba = model_cls(X_test_t).cpu().numpy()

            y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
            auroc = roc_auc_score(y_test_cls, y_pred_proba)
            cls_results[f'{model_name}_AUROC'].append(auroc)
            print(f"    Classification -- AUROC: {auroc:.4f}")

            # Free GPU memory
            del model_reg, model_cls
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

    # --- Save results ---
    print(f"\n[3/4] Saving results...")

    df_reg = pd.DataFrame(reg_results)
    df_cls = pd.DataFrame(cls_results)

    df_reg.to_csv(TABLES_DIR / 'deep_model_fold_results_regression.tsv', sep='\t', index=False)
    df_cls.to_csv(TABLES_DIR / 'deep_model_fold_results_classification.tsv', sep='\t', index=False)

    # Summary
    print(f"\n[4/4] Summary of results")
    print("=" * 70)
    summary_rows = []
    for model_name in model_configs:
        r2_vals = reg_results[f'{model_name}_R2']
        rmse_vals = reg_results[f'{model_name}_RMSE']
        mae_vals = reg_results[f'{model_name}_MAE']
        auroc_vals = cls_results[f'{model_name}_AUROC']

        row = {
            'Model': model_name,
            'R2_mean': np.mean(r2_vals),
            'R2_std': np.std(r2_vals),
            'RMSE_mean': np.mean(rmse_vals),
            'RMSE_std': np.std(rmse_vals),
            'MAE_mean': np.mean(mae_vals),
            'MAE_std': np.std(mae_vals),
            'AUROC_mean': np.mean(auroc_vals),
            'AUROC_std': np.std(auroc_vals),
        }
        summary_rows.append(row)

        print(f"\n  {model_name}:")
        print(f"    R2:    {row['R2_mean']:.4f} +/- {row['R2_std']:.4f}")
        print(f"    RMSE:  {row['RMSE_mean']:.2f} +/- {row['RMSE_std']:.2f}")
        print(f"    MAE:   {row['MAE_mean']:.2f} +/- {row['MAE_std']:.2f}")
        print(f"    AUROC: {row['AUROC_mean']:.4f} +/- {row['AUROC_std']:.4f}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(TABLES_DIR / 'deep_model_summary.tsv', sep='\t', index=False)

    # --- Instructions for merging results ---
    print("\n" + "=" * 70)
    print("DONE! Results saved to:")
    print(f"  {TABLES_DIR / 'deep_model_fold_results_regression.tsv'}")
    print(f"  {TABLES_DIR / 'deep_model_fold_results_classification.tsv'}")
    print(f"  {TABLES_DIR / 'deep_model_summary.tsv'}")
    print("=" * 70)


if __name__ == '__main__':
    main