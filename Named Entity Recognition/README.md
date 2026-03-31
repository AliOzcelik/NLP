# Named Entity Recognition with BERT

A multi-task NER model built on BERT that simultaneously predicts **Named Entity tags** and **Part-of-Speech tags** for each token in a sentence.

---

## Dataset

The dataset (`ner_dataset.csv`) contains **1,048,575 tokens** across **47,959 sentences**, with four columns:

| Column | Description |
|---|---|
| `Sentence #` | Sentence identifier |
| `Word` | Raw token |
| `POS` | Part-of-speech tag |
| `Tag` | Named entity tag (BIO format) |

### Named Entity Tags (17 classes)

The tags follow the **BIO scheme** (Beginning / Inside / Outside):

| Prefix | Entity Type | Meaning |
|---|---|---|
| `B-geo` / `I-geo` | Geography | Countries, cities, mountains |
| `B-per` / `I-per` | Person | People's names |
| `B-org` / `I-org` | Organization | Companies, institutions |
| `B-gpe` / `I-gpe` | Geo-Political Entity | Nations, states, regions |
| `B-tim` / `I-tim` | Time | Dates, time expressions |
| `B-art` / `I-art` | Artifact | Man-made objects, works |
| `B-eve` / `I-eve` | Event | Named events |
| `B-nat` / `I-nat` | Natural | Natural phenomena |
| `O` | Outside | Non-entity tokens |

### Part-of-Speech Tags (42 classes)

Standard Penn Treebank POS tags, including `NN`, `NNP`, `VB`, `JJ`, `IN`, `DT`, `CD`, etc.

---

## Model Architecture

The model is built on top of **`bert-base-uncased`** (110M parameters) with two parallel classification heads — one for NER tags and one for POS tags.

```
Input tokens
     │
  BERT (bert-base-uncased)
  last_hidden_state: [batch, seq_len, 768]
     │
     ├──── Linear(768 → 256) ──── Dropout(0.3) ──── Linear(256 → 43)  →  POS logits
     │
     └──── Linear(768 → 256) ──── Dropout(0.3) ──── Linear(256 → 18)  →  TAG logits
```

- **Tokenization**: each word is tokenized with `BertTokenizer`; the label of a word maps to its first sub-token position
- **Sequence length**: padded/truncated to 128 tokens
- **Special tokens**: `[CLS]` (101) prepended, `[SEP]` (102) appended; padding positions are masked out from loss computation
- **Loss**: `CrossEntropyLoss` applied separately to POS and TAG heads; averaged: `loss = (pos_loss + tag_loss) / 2`

---

## Training Pipeline

### Data Split
- **Train**: 90% of sentences
- **Validation**: 10% of sentences

### Preprocessing
Raw text is lowercased and common contractions are expanded (`won't` → `will not`, `didn't` → `did not`, etc.) before tokenization.

### Training Steps

1. **Initialize** BERT with pretrained weights and attach classification heads
2. **Freeze nothing** — full fine-tuning of all BERT layers and heads
3. **For each batch**:
   - Forward pass through BERT + both heads
   - Compute masked cross-entropy loss for POS and TAG (padding tokens excluded)
   - Backpropagate averaged loss
   - Clip gradients (`max_norm=1.0`) to prevent exploding gradients
   - Step optimizer and scheduler

### Hyperparameters

Hyperparameters were tuned over several training runs:

| Parameter | Value | Note |
|---|---|---|
| Base model | `bert-base-uncased` | Pretrained BERT |
| Learning rate | `1e-5` | Reduced from `2e-5` after observing instability |
| Optimizer | `AdamW` | Weight decay built-in |
| Scheduler | Linear warmup + linear decay | Warmup over first 10% of steps |
| Batch size | `16` | |
| Max sequence length | `128` | |
| Dropout | `0.3` | Increased from `0.1` to reduce overfitting |
| Gradient clipping | `1.0` | Added after observing loss spikes |
| Hidden size (MLP head) | `256` | Intermediate projection before classification |
| Epochs | `20` | Best checkpoint saved per epoch |

---

## Results

### Validation Loss Curve

| Epoch | POS Loss | TAG Loss |
|---|---|---|
| 1 | 1.8939 | 0.5320 |
| 2 | 1.0164 | 0.3291 |
| 3 | 0.8088 | 0.2817 |
| 4 | 0.7410 | 0.2672 |
| 5 | 0.6944 | 0.2543 |
| **6** | **0.6888** | **0.2490** |
| 7 | 0.6944 | 0.2472 |
| 8 | 0.7070 | 0.2486 |
| 9 | 0.7411 | 0.2582 |
| 10+ | diverging | diverging |

### Comments on Results

- **TAG loss converges faster and to a lower value than POS loss.** This is expected — the NER tag set (17 classes, dominated by `O`) is easier to classify than the POS set (42 fine-grained classes).
- **Best checkpoint is around epoch 6–7**, after which both losses begin to rise — a sign of overfitting. The saved best model corresponds to this checkpoint.
- **POS tagging is the harder task**: the loss plateaus around 0.69 and never fully converges within 20 epochs, suggesting the POS head may benefit from a larger hidden layer or longer training with stronger regularization.
- The early warmup scheduler stabilized training compared to earlier runs with `num_warmup_steps=0`, which caused oscillating loss from epoch 4 onward.
- Gradient clipping (`max_norm=1.0`) eliminated the `AcceleratorError` / CUDA assertion errors observed in initial runs caused by label index out-of-bounds during the backward pass.

---

## Inference

```python
predictor = NerPredictor(model, tokenizer, pre)

# Print token-level predictions
results = predictor.predict("Barack Obama visited London last Tuesday")
for r in results:
    print(f"{r['word']:15s}  TAG: {r['tag']:10s}  POS: {r['pos']}")

# Visualize with colored word boxes
predictor.visualize("Barack Obama visited London last Tuesday")
```

The `NerPredictor` class handles text cleaning, tokenization, word-to-token alignment, and maps predicted IDs back to human-readable label names.
