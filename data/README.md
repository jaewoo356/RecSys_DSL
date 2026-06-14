# Data Dictionary

This folder contains the **small, shareable** tabular data for the project. Large binaries (raw `.wav`
audio, precomputed Jukebox feature `.pkl`s, model weights, large `.npy` dumps) are **not** included —
see the [Large Files](../README.md#-large-files-weights--data) section of the main README.

```
data/
├── metadata/      ← what songs we have, and where to fetch their audio
└── mood_labels/   ← each song's mood (the supervision signal), generated with Gemini
```

---

## `metadata/` — song catalog

| File | Rows | Columns | Description |
|---|---|---|---|
| `metadata.csv` | 2,075 | `Id, Title, Artist, Genre` | Master song list aggregated from the Melon genre charts (input to the URL crawl). |
| `metadata_url.csv` | 2,075 | `Id, Title, Artist, Genre, Url` | Master list with the resolved YouTube URL per song (used to download audio). |

### `metadata/melon_top100_by_genre/`

Per-genre **Melon Top-100** source lists, before aggregation. Columns: `index, Genre, Title, Artist`
(~241 rows each, including chart duplicates). Genres:

```
Ballad · Dance · Hiphop · Hiphop_Overseas · R&B · Rock · Rock_Overseas
Indie · Electronika · Blues · Blues_Overseas · Pop
```

These are aggregated and de-duplicated by `notebooks/01_genre_aggregation.ipynb` into `metadata.csv`.

---

## `mood_labels/` — song → mood (Tasks 2 & 3 supervision)

| File | Rows | Description |
|---|---|---|
| `About_Song.csv` | 4,494 | Crawled song **descriptions** (e.g. from Genius). Columns: `Song, Title, Singer, About, 일치여부` (match flag). The `About` text is the input to Gemini for mood labeling. |
| `song_mood_15features.csv` | 2,075 | **The final label table.** `Title, Artist, Genre, Url` + a `0/1` column for each of the **15 moods**. This is the per-song mood vector used in the feature database. |
| `song_mood_with_descriptions.csv` | 2,075 | Same 15-mood labels, kept alongside the `About` description and match flag — handy for inspection. |

**The 15 moods:**

```
Happy · Sad · Energized · Relaxed · Nostalgic · Romantic · Angry · Focused
Inspired · Melancholic · Peaceful · Anxious · Confident · Dreamy · Lonely
```

> An earlier, wider taxonomy of **288 mood adjectives** was used before condensing to these 15.
> Those intermediate label matrices are not shipped here to keep the repo lean.

---

## Notes

- Some CSVs carry an unnamed leading index column from `pandas.to_csv(...)` — drop it on load
  (`pd.read_csv(path, index_col=0)`) if present.
- Song IDs in `metadata*` correspond to the `{id}.wav` audio filenames produced by the download
  notebooks and to the keys in the Jukebox feature dumps.
