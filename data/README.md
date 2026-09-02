# data/

IMDB 50K Movie Reviews — Kaggle `atulanandjha/imdb-50k-movie-reviews-test-your-bert`
(Stanford `aclImdb`).

- **Not committed.** The notebook fetches it (via `kagglehub`, or a manual download of
  `imdb_master.csv` dropped here).
- `imdb_master.csv` columns: `type` (`train` / `test`), `review` (text), `label`
  (`pos` / `neg` / `unsup`), plus an index and a file-path column.
- Build train / test:
  1. drop rows where `label == 'unsup'` (~50k unlabeled) → 25k train + 25k test,
  2. split by the `type` column — **this is the dataset's own split; do not re-split**,
  3. map `pos → 1`, `neg → 0`.
- Optionally the notebook writes `data/train.csv` and `data/test.csv` here (gitignored).

See `../docs/ASSIGNMENT.md` §5 for the constraints.
