"""
model.py — Tabular Foundation Model
=====================================

The whole model in ~120 lines. Here's the architecture at a glance:

    SUPPORT ROWS (labeled):        QUERY ROWS (unlabeled):
    ┌──────────────────────┐       ┌──────────────────────┐
    │ [x1,y1] [x2,y2] ...  │   +   │   [q1,?] [q2,?] ...  │
    └──────────────────────┘       └──────────────────────┘
                    │                           │
                    └───────────┬───────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   FeatureTokenizer     │   x → H-dim vector
                    │   + label embedding    │   y → H-dim label
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  TransformerEncoder   │   all tokens attend
                    │  (pre-norm, L layers) │   to all tokens
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │    output_head        │   query tokens → logits
                    └───────────────────────┘
"""

import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────────────────────
# Step 1: Tokenizer — convert a tabular row into an H-dim vector
# ─────────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Converts a row of D numerical features into a single d_model-dim embedding.

    For SUPPORT rows (labeled):
        token = Linear(x) + LabelEmbedding(y)

    For QUERY rows (unlabeled):
        token = Linear(x) + unknown_token

    The unknown_token is a learned parameter: "I see these features
    but don't know the label yet."

    Why no positional encoding? Tabular rows have no meaningful order,
    so we deliberately want permutation invariance.
    """

    def __init__(self, n_features: int, n_classes: int, d_model: int):
        super().__init__()
        # Project D-dimensional feature vector to d_model
        self.feature_proj = nn.Linear(n_features, d_model)

        # Embed discrete class labels into d_model-dim space
        self.label_emb = nn.Embedding(n_classes, d_model)

        # Learned "unknown label" token for query rows
        self.unknown = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.unknown, std=0.02)

    def encode_support(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, D) — feature matrix for support rows
        y : (B, N)    — integer labels for support rows
        → (B, N, H)   — token embeddings
        """
        # x = x.float()  # Convert to float32 if it's not already
        # y = y.float()   # Ensure labels are in long format if needed for embedding

        return self.feature_proj(x) + self.label_emb(y)

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, M, D) — feature matrix for query rows
        → (B, M, H)   — token embeddings (label unknown)
        """
        return self.feature_proj(x) + self.unknown  # unknown broadcasts over M


# ─────────────────────────────────────────────────────────────
# Step 2: Full model — tokenize → transformer → predict
# ─────────────────────────────────────────────────────────────

class TabularFoundationModel(nn.Module):
    """
    The main model. Takes a support set (labeled rows) and a query set
    (unlabeled rows) and predicts labels for the query rows.

    This is in-context learning: no gradient update at test time.
    The transformer "figures out" the prediction rule from the support set.
    """

    def __init__(
        self,
        n_features: int,       # D: number of input features (pad/truncate to this)
        n_classes: int,        # C: number of output classes
        d_model: int   = 128,  # H: transformer hidden dimension
        n_heads: int   = 4,    # number of attention heads
        n_layers: int  = 4,    # number of transformer layers
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes  = n_classes

        self.tokenizer = FeatureTokenizer(n_features, n_classes, d_model)

        # Standard transformer encoder.
        # norm_first=True → pre-norm → more stable training.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward = d_model * 4,
            dropout        = dropout,
            batch_first    = True,   # (B, T, H) convention
            norm_first     = True,   # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Map final hidden states → class logits
        self.output_head = nn.Linear(d_model, n_classes)

        # Weight init: small weights → stable early training
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        x_support: torch.Tensor,  # (B, N, D) — support features
        y_support: torch.Tensor,  # (B, N)    — support labels
        x_query:   torch.Tensor,  # (B, M, D) — query features
    ) -> torch.Tensor:            # (B, M, C) — logits for query rows
        """
        Forward pass:
          1. Tokenize support rows (features + label)
          2. Tokenize query rows  (features + unknown)
          3. Concatenate into one sequence
          4. Run transformer — all tokens attend to each other
          5. Extract query outputs → logits
        """
        B, N, _ = x_support.shape
        M       = x_query.shape[1]

        # 1. Tokenize
        support_tokens = self.tokenizer.encode_support(x_support, y_support)  # (B, N, H)
        query_tokens   = self.tokenizer.encode_query(x_query)                 # (B, M, H)

        # 2. Concatenate: support rows first, then query rows
        #    The transformer will let query tokens attend to support tokens.
        tokens = torch.cat([support_tokens, query_tokens], dim=1)  # (B, N+M, H)

        # 3. Transformer encoder — full attention, no masking needed
        out = self.transformer(tokens)  # (B, N+M, H)

        # 4. Extract the positions corresponding to query rows
        query_out = out[:, N:, :]  # (B, M, H)

        # 5. Project to class logits
        return self.output_head(query_out)  # (B, M, C)

    def predict(
        self,
        x_support: torch.Tensor,  # (N, D)
        y_support: torch.Tensor,  # (N,)
        x_query:   torch.Tensor,  # (M, D)
    ) -> torch.Tensor:            # (M,) predicted labels
        """Convenience method for single-task inference (no batch dim)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                x_support.unsqueeze(0),
                y_support.unsqueeze(0),
                x_query.unsqueeze(0),
            )  # (1, M, C)
            return logits.squeeze(0).argmax(dim=-1)  # (M,)


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, N, M, D, C = 4, 32, 8, 16, 3  # batch, support, query, features, classes

    model = TabularFoundationModel(n_features=D, n_classes=C)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")  # should be ~340K for defaults

    x_s = torch.randn(B, N, D)
    y_s = torch.randint(0, C, (B, N))
    x_q = torch.randn(B, M, D)

    logits = model(x_s, y_s, x_q)
    print(f"Input:  x_support {tuple(x_s.shape)}, y_support {tuple(y_s.shape)}, x_query {tuple(x_q.shape)}")
    print(f"Output: logits {tuple(logits.shape)}")  # (B, M, C)
    assert logits.shape == (B, M, C), "Shape mismatch!"
    print("✓ Forward pass OK")
