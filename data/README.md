# data/

IMDB 50K Movie Reviews — Kaggle `atulanandjha/imdb-50k-movie-reviews-test-your-bert`
(Stanford `aclImdb`).

- **Not committed.** The notebook fetches it with `kagglehub` (anonymous download works — no
  Kaggle token needed), falling back to whatever CSVs are dropped into this folder.
- **Dataset version 2** ships the split ready-made: `train.csv` and `test.csv`, 25,000 rows each,
  columns `text` (review) and `sentiment` (`pos` / `neg`), balanced 50/50 with labels on both
  sides. **This is the dataset's own split — do not re-split.**
- `load_imdb()` renames the columns to `review` / `label`, maps `pos → 1` and `neg → 0`, and drops
  exact-duplicate reviews **inside the training set only** (96 rows → 24,904 train / 25,000 test).
  About 123 review texts appear in both splits; noted in the notebook as a caveat, not altered.
- Cached copies live under `~/.cache/kagglehub/datasets/atulanandjha/…/versions/2/`.

See `../docs/ASSIGNMENT.md` §5 for the constraints.
