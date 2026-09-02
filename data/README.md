# data/

Train and test data for the assignment.

- The dataset is not yet chosen. Once it is, this folder holds either:
  - the small processed `train.csv` / `test.csv` (committed, so the notebook runs anywhere), or
  - a `download_data.py` / notebook cell that fetches it from Kaggle (with `data/raw/` gitignored).
- The brief requires a **fixed** train/test split — do not re-split. If the source is a single
  file, the notebook makes one stratified 80/20 split with `random_state=42` and writes the two
  files here.

See `../docs/ASSIGNMENT.md` §5 for the constraints.
