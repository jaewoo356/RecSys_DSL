# Known issues & corrected code

The original notebooks (`notebooks/05`, `07`) are kept **as authentic artifacts** of the DSL 24-1
project. They contain a few real correctness/rigor bugs that are worth flagging honestly rather than
quietly editing — the numbers they printed should be read with that in mind. Each fix below is drop-in.

> The clean recommendation + evaluation code in [`recsys/`](recsys/) (used by
> [`scripts/recommend_demo.py`](scripts/recommend_demo.py) and
> [`scripts/evaluate.py`](scripts/evaluate.py)) already follows these practices: fixed seeds,
> repo-root paths, real ranking metrics with baselines, and unit tests.

---

## 1. BERT validation never actually computed its metrics  *(notebook 05)*

The `validation()` loop hits **`break` on the first batch — before any `outputs`/`targets` are
computed** — so `fin_outputs`/`fin_targets` stay empty; the metric loop also `break`s after epoch 0.
**The reported Accuracy / F1 were therefore not measured on the test set.**

```python
# FIX — iterate the whole test set, compute once, no stray break:
def validation():
    model.eval()
    fin_targets, fin_outputs = [], []
    with torch.no_grad():
        for data in testing_loader:
            ids   = data['ids'].to(device, dtype=torch.long)
            mask  = data['mask'].to(device, dtype=torch.long)
            tt    = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, tt)
            fin_targets.extend(targets.cpu().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().numpy().tolist())
    return fin_outputs, fin_targets

outputs, targets = validation()
preds = (np.array(outputs) >= 0.5)
print("F1 micro:", metrics.f1_score(targets, preds, average='micro'))
print("F1 macro:", metrics.f1_score(targets, preds, average='macro'))
```

## 2. Wrong loss for multi-label  *(notebook 07)*

The audio probe targets are **multi-hot** 15-d vectors, but the loss is `nn.CrossEntropyLoss()`
(for single-label / a probability distribution). Use the same loss as the BERT head:

```python
criterion = nn.BCEWithLogitsLoss()          # model outputs raw logits, shape (B, 15)
# inference: probs = torch.sigmoid(logits); take top-k or threshold at 0.5
```

## 3. Unseeded train/test split → non-reproducible  *(notebook 07)*

`random_split(dataset, [train, test])` has no generator, so the split (and every metric on it)
changes every run. Seed globally **and** the split:

```python
import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
g = torch.Generator().manual_seed(42)
train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=g)
```

(or just call `recsys.seed_everything()`.)

## 4. Batch-size-1 Python training loop  *(notebook 07)*

Training iterates one example at a time (`for i in range(len(train_dataset))`), which is slow and
noisy. Use a batched `DataLoader`:

```python
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    loss = criterion(model(xb), yb)
    loss.backward(); optimizer.step()
```

## 5. Hardcoded Colab paths  *(all notebooks)*

Absolute `/content/drive/MyDrive/...` paths only run in one Colab account. Resolve from the repo root:

```python
from recsys.config import REPO_ROOT          # or pathlib.Path(__file__).resolve().parents[1]
csv = REPO_ROOT / "data" / "mood_labels" / "song_mood_15features.csv"
```

---

## Data-quality caveats (inherent, documented not "fixed")

- **Label coverage:** only **1,142 / 2,076 songs (55%)** have any mood label — **934 (45%) are
  unlabeled** and excluded from training/eval.
- **No human validation:** mood labels are Gemini-generated; there's no inter-annotator agreement or
  human gold set, so label quality is unmeasured.
- **Label skew:** support is uneven (Energized 563, Melancholic 536, Lonely 527 … Anxious 134), and
  moods co-occur heavily (Melancholic+Lonely 493) — the space is not balanced or independent.
- **No human relevance for playlists:** there is no ground truth for "the right playlist," so
  end-to-end quality can't be scored directly — see [`scripts/evaluate.py`](scripts/evaluate.py) for
  what *is* measurable (mood-space consistency, baseline lift, and a non-circular genre cross-check).
