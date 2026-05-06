"""
Tier 3 ensemble methods for CAFA-5: routing-based combiners.

Method 9 (SoftMoEMethod) and Method 10 (HardRoutingMethod) in the course
writeup. These methods use protein ProtT5 embeddings as a "context" signal
to decide how to combine the base models' confidences.

Hybrid partial-coverage mode
----------------------------
When the embeddings file only covers a subset of proteins (e.g. during
ongoing extraction), Tier 3 methods automatically:
    - Train only on rows whose proteins have real embeddings.
    - At inference, route via the gate/classifier for rows with embeddings,
      and fall back to a simple rule (default: per-row max over base
      confidences) for rows without.

Turn this off with `fallback=None` to get an error on missing embeddings
instead. Other fallback options: 'max' (default), 'mean'.

Methods:
    soft_moe       — Gate(emb) -> softmax(K) -> weighted sum of confidences.
                     Classical Mixture-of-Experts (Jacobs et al., 1991).
    hard_routing   — Classifier(emb) -> argmax(K) -> use that team's conf verbatim.

For Method 8 ("Enriched Stacking" — concat protein embedding with conf features
and train a flat meta-learner), use Tier 2 methods with `use_embeddings=True`
and optionally `emb_pca=<int>`. No separate class is needed.

Example (partial embeddings; hybrid inference):
    from src.models.tier3 import SoftMoEMethod, HardRoutingMethod

    data = EnsembleData.from_merged_dir("data/final_data")  # picks up partial h5

    soft = SoftMoEMethod(hidden=(256, 128), entropy_reg=0.01, fallback='max')
    soft.fit(train_df, train_terms, ia_weights, embeddings=data)
    preds = soft.predict(test_df, embeddings=data)
    # rows for proteins not in embeddings.h5 get max(conf_*) instead of routed preds
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.tier1 import EnsembleMethod
from src.models.tier2 import rows_to_emb_matrix


VALID_FALLBACKS = {'max', 'mean', None}


# ============================================================================
# Base class
# ============================================================================

class Tier3Method(EnsembleMethod):
    """Shared scaffolding for routing-based combiners.

    Subclasses implement:
        _fit_impl(self, train_data, embeddings)
                -> train on the already-filtered rows
        _predict_impl(self, test_data, embeddings)
                -> (N,) array of [0,1] scores

    The base class handles:
        - validating that embeddings are provided
        - filtering training rows to proteins with real embeddings
        - blending routed predictions with a fallback for proteins that
          lack embeddings at predict time
    """
    name: str = "tier3_base"

    def __init__(self, fallback='max', verbose=True):
        super().__init__()
        if fallback not in VALID_FALLBACKS:
            raise ValueError(
                f"fallback must be one of {VALID_FALLBACKS}, got {fallback!r}"
            )
        self.fallback       = fallback
        self.verbose        = verbose
        self.conf_cols_     = None
        self.n_teams_       = None
        self.emb_dim_       = None
        self.train_coverage_ = None

    # ---- public API -----------------------------------------------------

    def fit(self, train_data, train_labels, ia_weights, embeddings=None):
        if embeddings is None:
            raise ValueError(f"{self.name}: requires an embeddings argument at fit time")
        if 'label' not in train_data.columns:
            raise ValueError(f"{self.name}: train_data must have a 'label' column")

        self.conf_cols_ = self._conf_cols(train_data)
        self.n_teams_   = len(self.conf_cols_)

        # Coverage check + filter
        _, mask = rows_to_emb_matrix(
            train_data['protein_id'].values, embeddings, return_mask=True,
        )
        n_total      = len(train_data)
        n_with       = int(mask.sum())
        n_prot_total = train_data['protein_id'].nunique()
        n_prot_with  = train_data.loc[mask, 'protein_id'].nunique() if n_with else 0

        self.train_coverage_ = {
            'rows_total':     n_total,
            'rows_with_emb':  n_with,
            'proteins_total': n_prot_total,
            'proteins_with_emb': n_prot_with,
        }

        if self.verbose:
            print(f"{self.name}: training on {n_with:,}/{n_total:,} rows "
                  f"({n_prot_with:,}/{n_prot_total:,} proteins) with embeddings")
        if n_with == 0:
            raise ValueError(
                f"{self.name}: no training rows have embeddings. "
                "Check embeddings_path or wait for extraction to progress."
            )

        filtered = train_data.loc[mask].reset_index(drop=True)
        self._fit_impl(filtered, embeddings)
        return self

    def predict(self, test_data, embeddings=None):
        if embeddings is None:
            raise ValueError(f"{self.name}: requires an embeddings argument at predict time")
        if self.conf_cols_ is None:
            raise RuntimeError(f"{self.name} must be fit before predict")
        missing_conf = set(self.conf_cols_) - set(self._conf_cols(test_data))
        if missing_conf:
            raise ValueError(f"{self.name}: test_data missing conf columns {missing_conf}")

        pids = test_data['protein_id'].values
        _, mask = rows_to_emb_matrix(pids, embeddings, return_mask=True)
        n_total = len(test_data)
        n_with  = int(mask.sum())

        if self.verbose:
            n_missing = n_total - n_with
            msg = (f"{self.name}: routing {n_with:,}/{n_total:,} rows; "
                   f"{n_missing:,} rows without embeddings")
            if n_missing:
                if self.fallback is None:
                    msg += " will ERROR (fallback=None)"
                else:
                    msg += f" use fallback='{self.fallback}'"
            print(msg)

        if n_with < n_total and self.fallback is None:
            raise RuntimeError(
                f"{self.name}: {n_total - n_with:,} test rows lack embeddings "
                "and fallback=None. Either set fallback='max'/'mean' or "
                "finish embedding extraction first."
            )

        # Compute routed predictions ONLY on rows with embeddings (avoids
        # feeding zero-vectors through the gate).
        out = np.zeros(n_total, dtype=np.float32)
        if n_with == n_total:
            out[:] = self._predict_impl(test_data, embeddings)
        elif n_with > 0:
            routed_subset = self._predict_impl(
                test_data.loc[mask].reset_index(drop=True), embeddings,
            )
            out[mask] = routed_subset

        # Fill missing rows with the fallback
        if n_with < n_total and self.fallback is not None:
            fb = self._compute_fallback(test_data)
            out[~mask] = fb[~mask]

        out_df = test_data[['protein_id', 'GO_term']].copy()
        out_df['confidence'] = np.clip(out, 0.0, 1.0)
        return out_df

    # ---- fallback computation ------------------------------------------

    def _compute_fallback(self, df):
        conf = df[self.conf_cols_].to_numpy(dtype=np.float32)
        if self.fallback == 'max':
            return conf.max(axis=1)
        if self.fallback == 'mean':
            return conf.mean(axis=1)
        raise ValueError(f"unknown fallback: {self.fallback}")

    # ---- device helper (shared by PyTorch subclasses) ------------------

    def _resolve_device(self):
        import torch
        if getattr(self, 'device', None) is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    # ---- hooks for subclasses ------------------------------------------

    def _fit_impl(self, train_data, embeddings):
        raise NotImplementedError

    def _predict_impl(self, test_data, embeddings):
        raise NotImplementedError


# ============================================================================
# Soft Mixture-of-Experts
# ============================================================================

class SoftMoEMethod(Tier3Method):
    """Gate network produces softmax routing weights over K base models.

    Architecture:
        gate_logits  = MLP(protein_embedding)              # shape (K,)
        gate_weights = softmax(gate_logits)                # shape (K,), sums to 1
        mixed_conf   = sum_k gate_weights[k] * conf[k]     # scalar in [0, 1]

    Loss: binary cross-entropy on mixed_conf vs label, with optional
    entropy regularization on gate_weights to discourage collapse to a
    single expert.

    All rows of the same protein share gate weights by construction —
    the gate sees only the protein embedding, not the GO term.
    """
    name = "soft_moe"

    def __init__(
        self,
        hidden=(256, 128),
        dropout=0.3,
        entropy_reg=0.0,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=4096,
        epochs=30,
        patience=5,
        val_frac=0.1,
        device=None,
        random_state=42,
        fallback='max',
        verbose=True,
    ):
        super().__init__(fallback=fallback, verbose=verbose)
        self.hidden        = tuple(hidden)
        self.dropout       = dropout
        self.entropy_reg   = entropy_reg
        self.lr            = lr
        self.weight_decay  = weight_decay
        self.batch_size    = batch_size
        self.epochs        = epochs
        self.patience      = patience
        self.val_frac      = val_frac
        self.device        = device
        self.random_state  = random_state

    def _build_gate(self, emb_dim, n_teams):
        import torch.nn as nn
        layers = []
        prev = emb_dim
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers += [nn.Linear(prev, n_teams)]
        return nn.Sequential(*layers)

    def _fit_impl(self, train_data, embeddings):
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        conf = train_data[self.conf_cols_].to_numpy(dtype=np.float32)
        emb  = rows_to_emb_matrix(train_data['protein_id'].values, embeddings)
        y    = train_data['label'].to_numpy(dtype=np.float32)
        self.emb_dim_ = emb.shape[1]

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        n = len(conf)
        idx = rng.permutation(n)
        n_val = max(1, int(n * self.val_frac))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        conf_tr, emb_tr, y_tr = conf[tr_idx], emb[tr_idx], y[tr_idx]
        conf_va, emb_va, y_va = conf[val_idx], emb[val_idx], y[val_idx]

        device = self._resolve_device()
        gate = self._build_gate(self.emb_dim_, self.n_teams_).to(device)
        opt = torch.optim.Adam(
            gate.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

        tr_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(conf_tr).float(),
                torch.from_numpy(emb_tr).float(),
                torch.from_numpy(y_tr).float(),
            ),
            batch_size=self.batch_size, shuffle=True,
        )
        conf_va_t = torch.from_numpy(conf_va).float().to(device)
        emb_va_t  = torch.from_numpy(emb_va).float().to(device)
        y_va_t    = torch.from_numpy(y_va).float().to(device)

        best_val = float('inf')
        best_state = None
        patience_left = self.patience

        for epoch in range(self.epochs):
            gate.train()
            for conf_b, emb_b, y_b in tr_loader:
                conf_b = conf_b.to(device)
                emb_b  = emb_b.to(device)
                y_b    = y_b.to(device)

                gate_w = torch.softmax(gate(emb_b), dim=-1)           # (B, K)
                mixed  = (gate_w * conf_b).sum(dim=-1)                # (B,)
                mixed  = mixed.clamp(1e-7, 1 - 1e-7)
                loss = -(y_b * mixed.log() + (1 - y_b) * (1 - mixed).log()).mean()

                if self.entropy_reg > 0.0:
                    ent = -(gate_w * (gate_w + 1e-12).log()).sum(dim=-1).mean()
                    loss = loss - self.entropy_reg * ent

                opt.zero_grad(); loss.backward(); opt.step()

            gate.eval()
            with torch.no_grad():
                gate_w_va = torch.softmax(gate(emb_va_t), dim=-1)
                mixed_va  = (gate_w_va * conf_va_t).sum(dim=-1).clamp(1e-7, 1 - 1e-7)
                val_loss = -(
                    y_va_t * mixed_va.log() + (1 - y_va_t) * (1 - mixed_va).log()
                ).mean().item()

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in gate.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            gate.load_state_dict(best_state)
        gate.eval()
        self._gate = gate
        self._device = device
        self._best_val_loss = best_val

    def _predict_impl(self, test_data, embeddings):
        import torch
        import pandas as pd
        import numpy as np

        # 1. Isolate unique proteins and map their positions
        # return_inverse gives us an array to instantly map back to the 25M rows
        unique_pids, inv_indices = np.unique(test_data['protein_id'].values, return_inverse=True)

        # 2. Get embeddings ONLY for unique proteins (~142k instead of 25M)
        emb = rows_to_emb_matrix(unique_pids, embeddings)
        emb_t = torch.from_numpy(emb).float().to(self._device)

        # 3. Calculate gate weights
        with torch.no_grad():
            gate_w_unique = torch.softmax(self._gate(emb_t), dim=-1).cpu().numpy()

        # --- Interpretability Export (Now much faster and cleaner) ---
        df_gates = pd.DataFrame(gate_w_unique, columns=self.conf_cols_)
        df_gates['protein_id'] = unique_pids
        self.test_gates = df_gates
        # -------------------------------------------------------------

        # 4. Broadcast the unique weights back to the full 25M rows instantly
        gate_w_full = gate_w_unique[inv_indices]

        # 5. Multiply with confidence scores and sum
        conf = test_data[self.conf_cols_].to_numpy(dtype=np.float32)
        mixed = (gate_w_full * conf).sum(axis=-1)

        return mixed

    def inspect_gates(self, proteins, embeddings):
        """Return a DataFrame of per-protein gate weights for inspection."""
        import torch
        emb, mask = rows_to_emb_matrix(
            np.asarray(proteins), embeddings, return_mask=True,
        )
        emb_t = torch.from_numpy(emb).float().to(self._device)
        with torch.no_grad():
            gate_w = torch.softmax(self._gate(emb_t), dim=-1).cpu().numpy()
        df = pd.DataFrame(gate_w, index=proteins, columns=self.conf_cols_)
        df['has_embedding'] = mask
        return df


# ============================================================================
# Hard Routing (Selector Network)
# ============================================================================

class HardRoutingMethod(Tier3Method):
    """Classifier selects one base model per protein; use its conf verbatim.

    Training proceeds in two stages:
      1. For each unique training protein, compute per-team log-likelihood
         over that protein's annotations. Best team label = argmax_k LL_k.
      2. Train a K-class classifier  emb -> team_index  via cross-entropy.

    At inference, predict best team per (unique) protein, then for each row
    output conf[predicted_team]. No blending.
    """
    name = "hard_routing"

    def __init__(
        self,
        hidden=(256, 128),
        dropout=0.3,
        min_annotations=3,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=512,
        epochs=50,
        patience=5,
        val_frac=0.1,
        device=None,
        random_state=42,
        fallback='max',
        verbose=True,
    ):
        super().__init__(fallback=fallback, verbose=verbose)
        self.hidden          = tuple(hidden)
        self.dropout         = dropout
        self.min_annotations = min_annotations
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.batch_size      = batch_size
        self.epochs          = epochs
        self.patience        = patience
        self.val_frac        = val_frac
        self.device          = device
        self.random_state    = random_state

    def _compute_best_team_labels(self, train_data, eps=1e-7):
        """Per-protein argmax team by log-likelihood over its annotations."""
        conf = train_data[self.conf_cols_].to_numpy(dtype=np.float64)
        conf = np.clip(conf, eps, 1.0 - eps)
        y    = train_data['label'].to_numpy(dtype=np.float64)

        log_probs = y[:, None] * np.log(conf) + (1.0 - y[:, None]) * np.log(1.0 - conf)

        ll_df = pd.DataFrame(log_probs, columns=self.conf_cols_)
        ll_df['protein_id'] = train_data['protein_id'].values
        grouped = ll_df.groupby('protein_id')
        counts  = grouped.size()
        sums    = grouped[self.conf_cols_].sum()

        keep = counts >= self.min_annotations
        sums = sums[keep]
        if len(sums) == 0:
            raise ValueError(
                f"{self.name}: no proteins have >= {self.min_annotations} "
                "annotations; lower min_annotations."
            )

        best = sums.to_numpy().argmax(axis=1)
        return pd.Series(best, index=sums.index, name='best_team'), counts[keep]

    def _build_classifier(self, emb_dim, n_teams):
        import torch.nn as nn
        layers = []
        prev = emb_dim
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers += [nn.Linear(prev, n_teams)]
        return nn.Sequential(*layers)

    def _fit_impl(self, train_data, embeddings):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        best_team, counts = self._compute_best_team_labels(train_data)
        protein_ids = best_team.index.to_numpy()
        y = best_team.to_numpy(dtype=np.int64)

        dist = np.bincount(y, minlength=self.n_teams_)
        self._team_distribution_ = dict(zip(self.conf_cols_, dist.tolist()))
        if self.verbose:
            print(f"  team distribution (best per protein): {self._team_distribution_}")

        emb = rows_to_emb_matrix(protein_ids, embeddings)
        self.emb_dim_ = emb.shape[1]

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        n = len(emb)
        idx = rng.permutation(n)
        n_val = max(1, int(n * self.val_frac))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        emb_tr, y_tr = emb[tr_idx], y[tr_idx]
        emb_va, y_va = emb[val_idx], y[val_idx]

        device = self._resolve_device()
        net = self._build_classifier(self.emb_dim_, self.n_teams_).to(device)
        opt = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        tr_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(emb_tr).float(),
                torch.from_numpy(y_tr),
            ),
            batch_size=self.batch_size, shuffle=True,
        )
        emb_va_t = torch.from_numpy(emb_va).float().to(device)
        y_va_t   = torch.from_numpy(y_va).to(device)

        best_val = float('inf')
        best_state = None
        patience_left = self.patience

        for epoch in range(self.epochs):
            net.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = net(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()

            net.eval()
            with torch.no_grad():
                val_logits = net(emb_va_t)
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
        self._net    = net
        self._device = device
        self._best_val_loss = best_val
        self._n_train_proteins = n

    def _predict_impl(self, test_data, embeddings):
        import torch
        unique_pids = test_data['protein_id'].unique()
        emb = rows_to_emb_matrix(unique_pids, embeddings)
        emb_t = torch.from_numpy(emb).float().to(self._device)
        with torch.no_grad():
            logits = self._net(emb_t)
            pred_team = logits.argmax(dim=-1).cpu().numpy()
        pid_to_team = dict(zip(unique_pids, pred_team))

        conf = test_data[self.conf_cols_].to_numpy(dtype=np.float32)
        teams_for_rows = np.fromiter(
            (pid_to_team[p] for p in test_data['protein_id'].values),
            dtype=np.int64, count=len(test_data),
        )
        return conf[np.arange(len(conf)), teams_for_rows]

    def inspect_routing(self, proteins, embeddings):
        """Return per-protein team probabilities + argmax."""
        import torch
        emb, mask = rows_to_emb_matrix(
            np.asarray(proteins), embeddings, return_mask=True,
        )
        emb_t = torch.from_numpy(emb).float().to(self._device)
        with torch.no_grad():
            probs = torch.softmax(self._net(emb_t), dim=-1).cpu().numpy()
        df = pd.DataFrame(probs, index=proteins, columns=self.conf_cols_)
        df['argmax'] = [self.conf_cols_[i] for i in probs.argmax(axis=1)]
        df['has_embedding'] = mask
        return df


# ============================================================================
# Registry
# ============================================================================

TIER3_METHODS = {
    SoftMoEMethod.name:     SoftMoEMethod,
    HardRoutingMethod.name: HardRoutingMethod,
}


def build_tier3_methods(names=None, fallback='max', verbose=True):
    """Instantiate Tier 3 methods by name. None -> all of them."""
    if names is None:
        names = list(TIER3_METHODS.keys())
    return [
        TIER3_METHODS[n](fallback=fallback, verbose=verbose) for n in names
    ]