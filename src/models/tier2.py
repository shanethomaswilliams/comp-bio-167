"""
Tier 2 ensemble methods for CAFA-5: learned flat combiners (stacking).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import time

from src.models.tier1 import EnsembleMethod
import torch

def rows_to_emb_matrix(protein_ids, embeddings, return_mask=False):
    if hasattr(embeddings, 'embedding_matrix'):
        result = embeddings.embedding_matrix(protein_ids, return_mask=return_mask)
        if return_mask:
            emb, mask = result
            return np.asarray(emb, dtype=np.float32), np.asarray(mask, dtype=bool)
        return np.asarray(result, dtype=np.float32)

    if isinstance(embeddings, dict):
        dim = next((len(v) for v in embeddings.values()), None)
        if dim is None:
            raise ValueError("rows_to_emb_matrix: embeddings dict is empty")
        out  = np.zeros((len(protein_ids), dim), dtype=np.float32)
        mask = np.zeros(len(protein_ids), dtype=bool)
        for i, pid in enumerate(protein_ids):
            v = embeddings.get(pid)
            if v is not None:
                out[i]  = v
                mask[i] = True
        if return_mask:
            return out, mask
        return out

    if callable(embeddings):
        mat = np.asarray(embeddings(protein_ids), dtype=np.float32)
        if return_mask:
            return mat, np.ones(len(protein_ids), dtype=bool)
        return mat

    raise TypeError(f"unsupported embeddings type: {type(embeddings)}")

class Tier2Method(EnsembleMethod):
    name: str = "tier2_base"

    def __init__(self, use_embeddings: bool = False, emb_pca: int | None = None, train_subsample: float = 1.0):
        super().__init__()
        self.use_embeddings = use_embeddings
        self.emb_pca = emb_pca
        self.train_subsample = train_subsample  # New parameter for subsampling!
        self.conf_cols_ = None
        self._pca = None        
        self._emb_dim_raw = None
        self._emb_dim_out = None

        base = type(self).name
        if use_embeddings:
            suffix = f"_emb_pca{emb_pca}" if emb_pca else "_emb"
            self.name = base + suffix
        else:
            self.name = base

    def fit(self, train_data, train_labels, ia_weights, embeddings=None):
        if 'label' not in train_data.columns:
            raise ValueError(f"{self.name}: train_data must have a 'label' column.")
        
        # --- NEW: Protein-Level Subsampling ---
        if self.train_subsample < 1.0:
            unique_pids = train_data['protein_id'].unique()
            rng = np.random.default_rng(42) # Fixed seed for stable CV
            n_keep = max(1, int(len(unique_pids) * self.train_subsample))
            keep_pids = set(rng.choice(unique_pids, n_keep, replace=False))
            
            orig_len = len(train_data)
            train_data = train_data[train_data['protein_id'].isin(keep_pids)].reset_index(drop=True)
            print(f"\n    -> [Subsample] Kept {n_keep:,} proteins. Rows reduced: {orig_len:,} -> {len(train_data):,}")

        self.conf_cols_ = self._conf_cols(train_data)
        X = self._build_X(train_data, embeddings, fit_pca=True)
        y = train_data['label'].to_numpy(dtype=np.float32)
        self._fit_model(X, y)
        return self

    def predict(self, test_data, embeddings=None):
        if self.conf_cols_ is None:
            raise RuntimeError(f"{self.name} must be fit before predict")
        
        conf_cols = self._conf_cols(test_data)
        missing = set(self.conf_cols_) - set(conf_cols)
        if missing:
            raise ValueError(f"{self.name}: test_data missing conf columns {missing}")
            
        X = self._build_X(test_data, embeddings, fit_pca=False)
        conf = self._predict_proba(X)
        out = test_data[['protein_id', 'GO_term']].copy()
        out['confidence'] = conf
        return out

    def _build_X(self, df, embeddings, fit_pca: bool):
        X_conf = df[self.conf_cols_].to_numpy(dtype=np.float32)
        if not self.use_embeddings:
            return X_conf

        X_emb = rows_to_emb_matrix(df['protein_id'].values, embeddings)
        if self._emb_dim_raw is None:
            self._emb_dim_raw = X_emb.shape[1]

        if self.emb_pca is not None:
            if fit_pca:
                from sklearn.decomposition import PCA
                unique_pids = df['protein_id'].unique()
                X_unique = rows_to_emb_matrix(unique_pids, embeddings)
                self._pca = PCA(n_components=self.emb_pca, random_state=42)
                self._pca.fit(X_unique)
                self._emb_dim_out = self.emb_pca
            X_emb = self._pca.transform(X_emb).astype(np.float32)
        else:
            self._emb_dim_out = self._emb_dim_raw

        return np.hstack([X_conf, X_emb])

    def _fit_model(self, X, y): raise NotImplementedError
    def _predict_proba(self, X): raise NotImplementedError


class WeightedMeanMethod(Tier2Method):
    name = "weighted_mean"
    def __init__(self, use_embeddings=False, emb_pca=None, train_subsample=1.0, **kwargs):
        super().__init__(use_embeddings=use_embeddings, emb_pca=emb_pca, train_subsample=train_subsample)
        self.weights_ = None

    def _fit_model(self, X, y):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(positive=True, fit_intercept=False)
        model.fit(X, y)
        w = np.asarray(model.coef_, dtype=np.float64)
        total = w.sum()
        self.weights_ = w / total if total > 0 else np.ones_like(w) / len(w)
        self._model = model

    def _predict_proba(self, X):
        return np.clip(X @ self.weights_, 0.0, 1.0)


class LogRegMethod(Tier2Method):
    name = "logreg"
    def __init__(self, C=1.0, max_iter=1000, class_weight=None, use_embeddings=False, emb_pca=None, train_subsample=1.0, **kwargs):
        super().__init__(use_embeddings=use_embeddings, emb_pca=emb_pca, train_subsample=train_subsample)
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight

    def _fit_model(self, X, y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=self.C, max_iter=self.max_iter, class_weight=self.class_weight, solver='lbfgs')
        model.fit(X, y.astype(int))
        self._model = model

    def _predict_proba(self, X):
        return self._model.predict_proba(X)[:, 1]


class XGBoostMethod(Tier2Method):
    name = "xgb"
    def __init__(
        self, n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9,
        colsample_bytree=1.0, reg_lambda=1.0, n_jobs=-1, random_state=42,
        use_embeddings=False, emb_pca=None, train_subsample=1.0, device='auto', **kwargs
    ):
        super().__init__(use_embeddings=use_embeddings, emb_pca=emb_pca, train_subsample=train_subsample)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.device = device

    def _fit_model(self, X, y):
        from xgboost import XGBClassifier
        
        xgb_kwargs = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_lambda': self.reg_lambda,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
        }
        
        # --- NEW: Explicitly trigger GPU training ---
        if str(self.device).lower() in ['cuda', 'gpu']:
            xgb_kwargs['tree_method'] = 'hist'
            xgb_kwargs['device'] = 'cuda'
            print("    -> [XGBoost] GPU Acquired! Training with tree_method='hist'")
            
        model = XGBClassifier(**xgb_kwargs)
        model.fit(X, y.astype(int))
        self._model = model

    def _predict_proba(self, X):
        return self._model.predict_proba(X)[:, 1]


class MLPMethod(Tier2Method):
    name = "mlp"
    def __init__(
        self, hidden=(64, 32), dropout=0.3, lr=1e-3, weight_decay=1e-5, batch_size=81920,
        epochs=50, patience=5, val_frac=0.1, device=None, random_state=42,
        use_embeddings=False, emb_pca=None, train_subsample=1.0, **kwargs
    ):
        super().__init__(use_embeddings=use_embeddings, emb_pca=emb_pca, train_subsample=train_subsample)
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_frac = val_frac
        self.device = device
        self.random_state = random_state,

    def _build_net(self, in_dim):
        import torch.nn as nn
        layers = []
        prev = in_dim
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        return nn.Sequential(*layers)

    def _resolve_device(self):
        import torch
        if self.device is not None:
            return torch.device(self.device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fit_model(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        n = len(X)
        idx = rng.permutation(n)
        n_val = max(1, int(n * self.val_frac))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        device = self._resolve_device()
        net = self._build_net(X.shape[1]).to(device)
        opt = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        tr_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_tr).float(),
                torch.from_numpy(y_tr).float(),
            ),
            batch_size=self.batch_size, shuffle=True,
        )
        X_va_t = torch.from_numpy(X_va).float().to(device)
        y_va_t = torch.from_numpy(y_va).float().to(device)

        best_val = float('inf')
        best_state = None
        patience_left = self.patience

        for epoch in range(self.epochs):
            net.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = net(xb).squeeze(-1)
                loss = loss_fn(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()

            net.eval()
            with torch.no_grad():
                val_logits = net(X_va_t).squeeze(-1)
                val_loss = loss_fn(val_logits, y_va_t).item()

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._net = net
        self._device = device
        self._best_val_loss = best_val

    def _predict_proba(self, X, batch_size=65536):
        net = self._net
        device = self._device
        n = len(X)
        out = np.empty(n, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_t = torch.from_numpy(X[start:end]).float().to(device, non_blocking=True)
                logits = net(X_t).squeeze(-1)
                out[start:end] = torch.sigmoid(logits).cpu().numpy()
                del X_t, logits
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return out


# ============================================================================
# Registry
# ============================================================================

TIER2_METHODS = {
    WeightedMeanMethod.name: WeightedMeanMethod,
    LogRegMethod.name:       LogRegMethod,
    XGBoostMethod.name:      XGBoostMethod,
    MLPMethod.name:          MLPMethod,
}


def build_tier2_methods(names=None, use_embeddings=False, emb_pca=None):
    """Instantiate Tier 2 methods by name. None -> all of them.

    When use_embeddings=True (optionally with emb_pca=<int>), all methods
    are built to expect an `embeddings` argument at fit/predict time.
    """
    if names is None:
        names = list(TIER2_METHODS.keys())
    return [
        TIER2_METHODS[n](use_embeddings=use_embeddings, emb_pca=emb_pca)
        for n in names
    ]