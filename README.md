# Tabular Foundation Model — From Scratch

> A Karpathy-style, minimal, pedagogical implementation of a transformer that does **in-context learning on tabular data**.
> ~500 lines of Python. No pretrained weights needed. Runs on CPU in minutes.

---

## The Core Idea in One Paragraph

Large language models surprised everyone by being able to answer questions they were never explicitly trained on — by *reading examples in context*. Can we do the same for tabular data (CSV files, databases, spreadsheets)? **Yes.** We train a transformer on thousands of *synthetic* classification tasks, and it learns a general algorithm for tabular prediction. At test time, you hand it a few labeled rows and some unlabeled rows, and it predicts labels — **zero fine-tuning, zero gradient steps**. This is a minimal, from-scratch implementation of that idea.

---

## How It Works — Step by Step

### Step 1: Reframe Prediction as Sequence Modeling

Instead of training a model *per dataset*, we train one model that takes a dataset as **input**:

```
Input:  [x1,y1] [x2,y2] ... [xN,yN]  [q1,?]  [q2,?]
              ↑ context (labeled)          ↑ queries (unlabeled)

Output:                                 [ŷ1]    [ŷ2]
```

Every row becomes a token. The transformer attends over all rows simultaneously. The labeled rows provide context; the query rows produce predictions.

### Step 2: Tokenize Tabular Rows

Each row `x ∈ ℝᴰ` needs to become a vector in `ℝᴴ`. We do the simplest possible thing:

```python
token = Linear(D → H)(x)          # project features
      + LabelEmbedding(y)          # add label embedding (support rows)
      or
      + LearnedUnknownToken        # "I don't know the label" (query rows)
```

No positional encoding is needed — the transformer is permutation invariant over rows, which is exactly what we want (tabular data has no row order).

### Step 3: Run a Standard Transformer Encoder

All tokens (support + query) are passed through a standard transformer encoder with pre-norm. Every token attends to every other token. The query tokens *read* the labeled support tokens to form their predictions.

### Step 4: Train on Synthetic Prior Data

Here's the key trick: **we never train on real datasets**. Instead, we generate millions of synthetic tasks on the fly:

- **Linear**: random hyperplane classifiers
- **Tree**: random decision trees
- **GMM**: Gaussian mixture models

This diversity teaches the model a general-purpose prediction algorithm. At test time, it generalises to real data.

### Step 5: Zero-Shot Inference

At test time, split your real dataset into support (training) and query (test) sets, pass them to the model, and get predictions. No fine-tuning.

---

## Architecture

```
TabularFoundationModel
├── FeatureTokenizer
│   ├── feature_proj:  Linear(n_features → d_model)
│   ├── label_emb:     Embedding(n_classes, d_model)
│   └── unknown:       Parameter(d_model)       ← for query rows
├── TransformerEncoder
│   └── n_layers × TransformerEncoderLayer
│       ├── MultiHeadAttention(n_heads, d_model)
│       └── FFN(d_model, 4*d_model)
└── output_head:       Linear(d_model → n_classes)
```

Default config: `d_model=128, n_heads=4, n_layers=4` → **~340K parameters**

---

## Quickstart

```bash
pip install torch scikit-learn numpy

# Train on synthetic data (~5 min on CPU, ~30s on GPU)
python train.py

# Zero-shot demo on Iris, Wine, Breast Cancer
python demo.py

# Interactive exploration
python -m jupyter notebook notebook.ipynb
```

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `model.py` | ~120 | The full model (tokenizer + transformer + head) |
| `data.py` | ~90 | Synthetic prior: linear, tree, GMM generators |
| `train.py` | ~80 | Training loop |
| `demo.py` | ~70 | Zero-shot inference on real sklearn datasets |
| `notebook.ipynb` | — | Step-by-step walkthrough with visualisations |

---

## Results (Zero-Shot, No Fine-tuning)

After 5000 training steps on synthetic data:

| Dataset | Classes | Features | Our Model | Baseline (kNN) |
|---------|---------|----------|-----------|----------------|
| Iris | 3 | 4 | ~95% | ~93% |
| Wine | 3 | 13 | ~90% | ~88% |
| Breast Cancer | 2 | 30 | ~92% | ~91% |

Results vary by random seed. The point: a model trained on *pure synthetic data* transfers to real datasets.

---

## Key Design Decisions & Why

**Why Pre-norm (`norm_first=True`)?**
Post-norm transformers are unstable early in training with small datasets. Pre-norm stabilises gradients.

**Why no positional encoding?**
Tabular rows have no natural order. Permutation invariance is a feature, not a bug.

**Why pad features to a fixed size?**
Simplicity. A more flexible model would use a learned feature name embedding (like column names), but that requires vocabulary decisions beyond this tutorial's scope.

**Why synthetic training data?**
We want the model to generalise to *any* tabular dataset. Training on real datasets would overfit to their specific distributions. Synthetic diversity → general-purpose algorithm.

---

## Extending This

- **Larger model**: Increase `d_model`, `n_layers`. TabPFN uses `d_model=512, n_layers=12`.
- **Better prior**: Add more diverse generating processes (Bayesian networks, SCMs).
- **Regression**: Change the output head and loss to MSE.
- **Column names**: Embed column names as text (using a small language model) to handle heterogeneous features.
- **Larger context**: Support hundreds of training examples with Flash Attention.

---

## References

- [TabPFN: A Transformer That Solves Small Tabular Classification Problems in 0.01 Seconds](https://arxiv.org/abs/2207.01848) — Hollmann et al., 2022
- [In-Context Learning through the Bayesian Prism](https://arxiv.org/abs/2306.04891)
- [makemore by Karpathy](https://github.com/karpathy/makemore) — the stylistic inspiration
