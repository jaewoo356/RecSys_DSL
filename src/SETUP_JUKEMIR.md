# Setting up JukeMIR (Task 3 — audio → representation)

The audio branch extracts features with **JukeMIR** — representations pulled from **OpenAI's Jukebox**.
We do **not** vendor the Jukebox model code in this repo (it's a large third-party package with pinned,
conflicting dependencies). Instead, install it from the upstream *simplified-jukemir* project and run the
feature extraction in its **own environment**.

> Reference implementation: <https://github.com/ldzhangyx/simplified-jukemir>
> Original paper: Castellon, Donahue, Liang — *"Codified audio language modeling learns useful
> representations for music information retrieval"* (ISMIR 2021).

[`representation.py`](representation.py) in this folder is the project's feature-extraction script, kept
as a concrete reference. It mirrors the logic in
[`../notebooks/06_jukemir_embeddings.ipynb`](../notebooks/06_jukemir_embeddings.ipynb).

---

## 1. Environment

JukeMIR needs **older** pinned versions (incompatible with the main `requirements.txt`), so use a
dedicated venv/conda env with **a single GPU (≥12 GB VRAM)**:

```bash
conda create -n jukemir python=3.8 -y
conda activate jukemir
pip install -r ../requirements-jukemir.txt   # torch + the pinned JukeMIR deps
```

## 2. Install Jukebox

```bash
git clone https://github.com/ldzhangyx/simplified-jukemir
cd simplified-jukemir
python -m pip install --no-cache-dir -e jukebox
```

## 3. Download the pretrained weights (NOT in this repo)

```bash
wget https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar          # ~7.7 MB
wget https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar  # ~10 GB
```

## 4. Extract features

Point the paths in `representation.py` (or notebook 06) at your weights, your `.wav` audio folder, and
an output folder, then run:

```bash
python representation.py
```

**What it does** (per song): convert to spectrogram (librosa) → split into 4 segments → take a random
~25 s crop from each (`SAMPLE_LENGTH = 1,048,576` samples ≈ 23.77 s @ 44.1 kHz) → pass all 4 crops
through Jukebox → a per-song **`[4, 4800]`** representation (`AVERAGE_SLICES = 64` local pooling).

These representations are the input to the mood probe in
[`../notebooks/07_audio_mood_classifier.ipynb`](../notebooks/07_audio_mood_classifier.ipynb).
