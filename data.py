"""
data.py — Synthetic Prior Generation
======================================

This is the secret sauce. We train entirely on *synthetic* data so the
model learns a general-purpose prediction algorithm. At test time, it
transfers to any real tabular dataset.

We generate tasks from three families:

  1. Linear    — random hyperplane classifiers
  2. Tree      — random decision trees
  3. GMM       — Gaussian mixture models

Each call to `sample_batch()` returns a fresh batch of random tasks.
The model has never seen a repeated task during training.

Why does this work?
  The model can't memorise answers (the tasks are random). Instead it's
  forced to learn *how to predict* from context — the meta-algorithm.
  This is the Prior-Data Fitted Networks (PFN) insight.
"""

import torch
import numpy as np
from typing import Tuple


# ─────────────────────────────────────────────────────────────
# Individual dataset generators
# ─────────────────────────────────────────────────────────────

def sample_linear(n: int, D: int, C: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random linear classification problem.
    A random weight matrix W defines C hyperplanes; the class with the
    highest score wins.

    Simple but covers a huge space of linear separability patterns.
    """
    X = torch.randn(n, D)
    W = torch.randn(D, C) * (1.0 / D**0.5)  # Xavier-ish init
    y = (X @ W).argmax(dim=1)  # (n,)
    return X, y


def sample_tree(n: int, D: int, C: int, max_depth: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random decision tree classifier.
    We randomly build a tree of depth `max_depth`, then classify each
    sample by routing it from the root to a leaf.

    Captures axis-aligned, non-linear decision boundaries.
    """
    X = torch.randn(n, D)
    X_np = X.numpy()

    # Each internal node: random feature + threshold
    n_nodes = 2 ** max_depth
    feats   = np.random.randint(0, D, size=n_nodes)
    thresh  = np.random.randn(n_nodes)
    # Each leaf: random label
    leaf_labels = np.random.randint(0, C, size=n_nodes)

    y_np = np.zeros(n, dtype=np.int64)
    for i in range(n):
        node = 0
        for _ in range(max_depth):
            f, t = feats[node % n_nodes], thresh[node % n_nodes]
            node = 2 * node + (1 if X_np[i, f] < t else 2)
        y_np[i] = leaf_labels[node % n_nodes]

    return X, torch.tensor(y_np, dtype=torch.long)


def sample_gmm(n: int, D: int, C: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gaussian Mixture Model.
    Each class has a random Gaussian cluster; samples are drawn from the
    cluster corresponding to their class.

    Captures compact, blob-like decision boundaries with overlap.
    """
    means = torch.randn(C, D) * 2.0        # random class centres
    scales = torch.rand(C, 1) * 1.5 + 0.3  # random per-class noise level

    y = torch.randint(0, C, (n,))
    X = means[y] + torch.randn(n, D) * scales[y]
    return X, y


def sample_poly(n: int, D: int, C: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random polynomial classifier.
    We add random quadratic features, then apply a linear classifier on top.
    Captures curved, non-linear boundaries.
    """
    X    = torch.randn(n, D)
    # Quadratic features: all pairs xi*xj
    pairs = []
    for i in range(min(D, 6)):       # cap at 6 features to avoid explosion
        for j in range(i, min(D, 6)):
            pairs.append((X[:, i] * X[:, j]).unsqueeze(1))
    X_aug = torch.cat([X] + pairs, dim=1)  # (n, D + n_pairs)

    W = torch.randn(X_aug.shape[1], C)
    y = (X_aug @ W).argmax(dim=1)
    return X, y  # return original features (not augmented)


# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────

def normalize(X: torch.Tensor) -> torch.Tensor:
    """
    Standardise features to zero mean and unit variance.
    Applied per-task so the model sees consistently scaled inputs.
    """
    mean = X.mean(dim=0, keepdim=True)
    std  = X.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (X - mean) / std


def pad_or_trim(X: torch.Tensor, target_D: int) -> torch.Tensor:
    """
    Ensure X has exactly target_D features.
    Pad with zeros if too few; trim if too many.
    """
    n, D = X.shape
    if D < target_D:
        return torch.cat([X, torch.zeros(n, target_D - D)], dim=1)
    return X[:, :target_D]


# ─────────────────────────────────────────────────────────────
# Batch sampler — used during training
# ─────────────────────────────────────────────────────────────

GENERATORS = [sample_linear, sample_tree, sample_gmm, sample_poly]


def sample_batch(
    batch_size:  int,
    n_support:   int,
    n_query:     int,
    n_features:  int,
    n_classes:   int,
    feature_noise: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a batch of B independent classification tasks.

    Each task is generated by a randomly chosen generator with randomly
    sampled hyper-parameters (e.g. tree depth, GMM scale). This gives
    virtually infinite task diversity.

    Returns:
        x_support : (B, N, D) — support feature matrices
        y_support : (B, N)    — support labels
        x_query   : (B, M, D) — query feature matrices
        y_query   : (B, M)    — query labels (targets for training loss)
    """
    x_s_list, y_s_list = [], []
    x_q_list, y_q_list = [], []

    n_total = n_support + n_query

    for _ in range(batch_size):
        # Pick a random generator
        gen = GENERATORS[np.random.randint(len(GENERATORS))]

        # Optionally sample a random smaller feature dimension,
        # then pad to n_features. This teaches the model to handle
        # datasets with fewer features.
        d_actual = np.random.randint(max(1, n_features // 4), n_features + 1)

        X, y = gen(n_total, d_actual, n_classes)
        X    = pad_or_trim(X, n_features)
        X    = normalize(X)

        # Optional: add feature noise for robustness
        if feature_noise > 0:
            X = X + torch.randn_like(X) * feature_noise

        x_s_list.append(X[:n_support])
        y_s_list.append(y[:n_support])
        x_q_list.append(X[n_support:])
        y_q_list.append(y[n_support:])

    return (
        torch.stack(x_s_list),   # (B, N, D)
        torch.stack(y_s_list),   # (B, N)
        torch.stack(x_q_list),   # (B, M, D)
        torch.stack(y_q_list),   # (B, M)
    )


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    x_s, y_s, x_q, y_q = sample_batch(
        batch_size=8, n_support=32, n_query=8, n_features=16, n_classes=3
    )
    print(f"x_support : {tuple(x_s.shape)}  (batch, support, features)")
    print(f"y_support : {tuple(y_s.shape)}  (batch, support)")
    print(f"x_query   : {tuple(x_q.shape)}  (batch, query, features)")
    print(f"y_query   : {tuple(y_q.shape)}  (batch, query)")
    print(f"label range: [{y_s.min()}, {y_s.max()}]")
    print("✓ Batch sampling OK")
