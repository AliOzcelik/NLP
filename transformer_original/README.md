# Transformer: English to French Translation

A from-scratch PyTorch implementation of the original Transformer architecture presented in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), applied to English-to-French machine translation.

## Overview

This project implements every component of the Transformer from the ground up — no high-level sequence-to-sequence wrappers. The model is trained and evaluated on a parallel English-French corpus of ~175,000 sentence pairs.

## Architecture

| Hyperparameter | Value |
|---|---|
| Layers (encoder & decoder) | 6 |
| Attention heads | 8 |
| Model dimension (`d_model`) | 512 |
| Feed-forward dimension | 2048 |
| Dropout | 0.1 |
| Max sequence length | 64 |

**Implemented components:**
- Sinusoidal positional encoding
- Multi-head scaled dot-product attention (with causal masking in decoder)
- Position-wise feed-forward network
- Encoder and Decoder stacks with residual connections and layer normalization
- Label smoothing loss (KL divergence)

## Training

- **Dataset:** `eng_french.csv` — 175,621 English/French sentence pairs
- **Train/test split:** 99% / 1%
- **Optimizer:** AdamW (`lr=2e-4`) with linear warmup (10% of steps)
- **Batch size:** 256
- **Epochs:** 100
- **Evaluation metric:** Corpus BLEU score (with smoothing)

## Requirements

```
torch
transformers
nltk
scikit-learn
pandas
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Open `transformer.ipynb` and run all cells in order:

1. Data loading and preprocessing
2. Vocabulary building and encoding
3. Model definition and training
4. Translation and BLEU evaluation
5. Metric plots (loss + BLEU over epochs)

Best model weights are saved to `weights/model_best.pt` based on BLEU score.

<img width="1800" height="600" alt="image" src="https://github.com/user-attachments/assets/80a28f39-3005-494e-abb3-e5900bdda48a" />


## Reference

> Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
