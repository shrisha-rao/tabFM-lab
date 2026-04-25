"""
demo.py — Zero-Shot Tabular Classification
============================================

We test our trained model on real datasets from scikit-learn.
The model was trained ONLY on synthetic data — it has never seen
Iris, Wine, or Breast Cancer. Yet it can predict accurately.

This is in-context learning: no fine-tuning, no gradient updates.
We just hand the model some labeled examples and ask it to predict.

Usage:
  python demo.py                      # uses tabfm.pt checkpoint
  python demo.py --ckpt my_model.pt
  python demo.py --n_support 64       # more context → better accuracy
"""

import argparse
import numpy as np
import torch
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from model import TabularFoundationModel


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_checkpoint(path: str, device: str) -> TabularFoundationModel:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = TabularFoundationModel(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def preprocess(X: np.ndarray, n_features: int) -> np.ndarray:
    """Standardise and pad/trim to n_features."""
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if X.shape[1] < n_features:
        X = np.hstack([X, np.zeros((X.shape[0], n_features - X.shape[1]))])
    else:
        X = X[:, :n_features]
    return X.astype(np.float32)


def predict_zero_shot(
    model:     TabularFoundationModel,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query:   np.ndarray,
    device:    str,
) -> np.ndarray:
    """Run one forward pass, return predicted labels."""
    x_s = torch.tensor(X_support, dtype=torch.float32).unsqueeze(0).to(device)
    y_s = torch.tensor(y_support, dtype=torch.long).unsqueeze(0).to(device)
    x_q = torch.tensor(X_query,   dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x_s, y_s, x_q)  # (1, M, C)
    return logits.squeeze(0).argmax(-1).cpu().numpy()


def cross_val_accuracy(model, X, y, n_features, n_splits, device):
    """K-fold cross-validation accuracy."""
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr = preprocess(X[train_idx], n_features)
        X_te = preprocess(X[test_idx],  n_features)
        y_tr, y_te = y[train_idx], y[test_idx]

        preds = predict_zero_shot(model, X_tr, y_tr, X_te, device)
        accs.append((preds == y_te).mean())
    return float(np.mean(accs)), float(np.std(accs))


def knn_accuracy(X, y, n_splits=5, k=5):
    """KNN baseline for comparison."""
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        sc = StandardScaler().fit(X[train_idx])
        X_tr, X_te = sc.transform(X[train_idx]), sc.transform(X[test_idx])
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_tr, y[train_idx])
        accs.append((knn.predict(X_te) == y[test_idx]).mean())
    return float(np.mean(accs)), float(np.std(accs))


# ─────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────

DATASETS = {
    "Iris          (150 rows, 4 feat, 3 cls)":    load_iris(return_X_y=True),
    "Wine          (178 rows,13 feat, 3 cls)":    load_wine(return_X_y=True),
    "Breast Cancer (569 rows,30 feat, 2 cls)":    load_breast_cancer(return_X_y=True),
}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      type=str, default="tabfm.pt")
    p.add_argument("--n_splits",  type=int, default=5, help="Cross-validation folds")
    p.add_argument("--device",    type=str, default="cpu")
    args = p.parse_args()

    print(f"\nLoading checkpoint: {args.ckpt}")
    model, cfg = load_checkpoint(args.ckpt, args.device)
    n_features = cfg["n_features"]
    print(f"Model: d_model={cfg['d_model']}, n_layers={cfg['n_layers']}, n_features={n_features}")

    print("\n" + "─" * 62)
    print(f"  {'Dataset':<38}  {'TabFM':>7}  {'KNN-5':>7}")
    print("─" * 62)

    for name, (X, y) in DATASETS.items():
        # Cap class indices (our model was trained with n_classes classes)
        y = y % cfg["n_classes"]

        tfm_acc, tfm_std = cross_val_accuracy(model, X, y, n_features, args.n_splits, args.device)
        knn_acc, knn_std = knn_accuracy(X, y, n_splits=args.n_splits)

        better = "↑" if tfm_acc >= knn_acc else " "
        print(f"  {name:<38}  {tfm_acc:.1%}±{tfm_std:.2f}  {knn_acc:.1%}±{knn_std:.2f}  {better}")

    print("─" * 62)
    print("\nTabFM = zero-shot (no fine-tuning, trained only on synthetic data)")
    print("KNN-5 = k-nearest neighbours with k=5 (a strong tabular baseline)")


    # ── Bonus: show how accuracy scales with support set size ──
    print("\n── Support size vs. accuracy (Iris) ──────────────────────")
    X_iris, y_iris = load_datasets_raw()["Iris"]
    for n_s in [4, 8, 16, 32, 64]:
        if n_s >= len(X_iris) * 0.8:
            break
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        accs = []
        for tr_idx, te_idx in skf.split(X_iris, y_iris):
            # Use only n_s support examples
            idx = tr_idx[:n_s]
            X_tr = preprocess(X_iris[idx], n_features)
            X_te = preprocess(X_iris[te_idx], n_features)
            preds = predict_zero_shot(model, X_tr, y_iris[idx], X_te, args.device)
            accs.append((preds == y_iris[te_idx]).mean())
        print(f"  n_support={n_s:3d}  →  acc={np.mean(accs):.1%}")


def load_datasets_raw():
    return {"Iris": load_iris(return_X_y=True)}


if __name__ == "__main__":
    main()
