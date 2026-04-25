"""
train.py — Training Loop
==========================

Training is dead simple:
  1. Sample a synthetic task batch
  2. Forward pass → logits for query rows
  3. Cross-entropy loss vs ground-truth query labels
  4. Backprop + AdamW + cosine LR schedule
  5. Repeat

The model never sees the same task twice. It learns to predict by
reading context, not by memorising data.

Usage:
  python train.py                        # default config (fast)
  python train.py --d_model 256 --n_layers 6 --n_steps 20000
"""

import argparse
import time
import torch
import torch.nn.functional as F

from model import TabularFoundationModel
from data  import sample_batch


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Train a Tabular Foundation Model")

    # Model
    p.add_argument("--n_features",  type=int,   default=16,    help="Fixed feature dim (pad/trim real data to this)")
    p.add_argument("--n_classes",   type=int,   default=10,    help="Max number of classes")
    p.add_argument("--d_model",     type=int,   default=128,   help="Transformer hidden dim")
    p.add_argument("--n_heads",     type=int,   default=4,     help="Attention heads")
    p.add_argument("--n_layers",    type=int,   default=4,     help="Transformer layers")
    p.add_argument("--dropout",     type=float, default=0.0,   help="Dropout (0 works well for small model)")

    # Data
    p.add_argument("--n_support",   type=int,   default=32,    help="Support set size per task")
    p.add_argument("--n_query",     type=int,   default=8,     help="Query set size per task")
    p.add_argument("--feature_noise", type=float, default=0.01, help="Feature noise for robustness")

    # Training
    p.add_argument("--batch_size",  type=int,   default=64,    help="Tasks per batch")
    p.add_argument("--lr",          type=float, default=3e-4,  help="Peak learning rate")
    p.add_argument("--n_steps",     type=int,   default=5000,  help="Training steps")
    p.add_argument("--warmup",      type=int,   default=200,   help="LR warmup steps")
    p.add_argument("--grad_clip",   type=float, default=1.0,   help="Gradient clipping norm")

    # Misc
    p.add_argument("--log_every",   type=int,   default=100,   help="Log interval")
    p.add_argument("--save_path",   type=str,   default="tabfm.pt", help="Checkpoint path")
    p.add_argument("--device",      type=str,   default="auto")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# LR schedule: linear warmup then cosine decay
# ─────────────────────────────────────────────────────────────

def get_lr(step: int, warmup: int, n_steps: int, lr: float) -> float:
    if step < warmup:
        return lr * step / max(warmup, 1)
    progress = (step - warmup) / max(n_steps - warmup, 1)
    return lr * 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train(args):
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Build model
    model = TabularFoundationModel(
        n_features = args.n_features,
        n_classes  = args.n_classes,
        d_model    = args.d_model,
        n_heads    = args.n_heads,
        n_layers   = args.n_layers,
        dropout    = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Training for {args.n_steps} steps...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95)
    )

    # ── Training loop ──────────────────────────────────────────
    running_loss = 0.0
    running_acc  = 0.0
    t0 = time.time()

    for step in range(1, args.n_steps + 1):

        # Set learning rate manually (gives us more control than a scheduler)
        lr_now = get_lr(step, args.warmup, args.n_steps, args.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        # ── 1. Sample a batch of synthetic tasks ──────────────
        x_s, y_s, x_q, y_q = sample_batch(
            batch_size    = args.batch_size,
            n_support     = args.n_support,
            n_query       = args.n_query,
            n_features    = args.n_features,
            n_classes     = args.n_classes,
            feature_noise = args.feature_noise,
        )
        x_s = x_s.to(device)
        y_s = y_s.to(device)
        x_q = x_q.to(device)
        y_q = y_q.to(device)

        # ── 2. Forward pass ────────────────────────────────────
        model.train()
        logits = model(x_s, y_s, x_q)  # (B, M, C)

        # ── 3. Loss ────────────────────────────────────────────
        # Flatten (B, M, C) → (B*M, C) for F.cross_entropy
        loss = F.cross_entropy(
            logits.reshape(-1, args.n_classes),
            y_q.reshape(-1),
        )

        # ── 4. Backward ────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # ── 5. Logging ─────────────────────────────────────────
        with torch.no_grad():
            acc = (logits.argmax(-1) == y_q).float().mean().item()
        running_loss += loss.item()
        running_acc  += acc

        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            avg_acc  = running_acc  / args.log_every
            elapsed  = time.time() - t0
            print(
                f"step {step:5d}/{args.n_steps} | "
                f"loss {avg_loss:.4f} | "
                f"acc {avg_acc:.3f} | "
                f"lr {lr_now:.2e} | "
                f"{elapsed:.1f}s"
            )
            running_loss = 0.0
            running_acc  = 0.0
            t0 = time.time()

    # ── Save checkpoint ────────────────────────────────────────
    torch.save({
        "state_dict":  model.state_dict(),
        "config": {
            "n_features": args.n_features,
            "n_classes":  args.n_classes,
            "d_model":    args.d_model,
            "n_heads":    args.n_heads,
            "n_layers":   args.n_layers,
        },
    }, args.save_path)
    print(f"\nSaved checkpoint → {args.save_path}")
    return model


if __name__ == "__main__":
    args = get_args()
    train(args)
