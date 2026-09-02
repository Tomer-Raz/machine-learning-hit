# Progress

Living checklist. Update at the end of every work session; keep the "current status" block in
`../CLAUDE.md` in sync.

## Current status

- **Milestone 1 — repo scaffold + docs: DONE.**
- **Dataset chosen:** IMDB 50K Movie Reviews (`atulanandjha/imdb-50k-movie-reviews-test-your-bert`).
- **Milestone 2 — Part 1: IN PROGRESS.**
  - `.venv` built with the full stack (`requirements.txt`).
  - `tools/build_notebook.py` written — Part 1 (title/students, AI-prompts, problem+dataset
    description, setup, `load_imdb()`, `.head()` of each, EDA/class-balance) + quality-index
    section with `score()`.
  - `notebook.ipynb` generated, code cells parse, imports + `score()` smoke-tested in the venv.
  - **Blocked on:** Kaggle token (`~/.kaggle/kaggle.json`) to download the data and execute.
- **Next action:** with the token in place — execute the notebook, confirm 25k/25k balanced
  split and `imdb_master.csv` schema (`Unnamed: 0,type,review,label,file`, latin-1), then extend
  the builder with Part 2 (feature engineering).

## Milestones

- [x] 1. Repo scaffold: `CLAUDE.md`, `README.md`, `requirements.txt`, `.gitignore`, `docs/`.
- [~] 2. Part 1 — data load (train/test via `imdb_master.csv` `type` column), `.head()` of each,
      one-paragraph description, class-balance chart, student-details + AI-prompts cells.
      *(written in the builder; awaiting Kaggle token to execute)*
- [ ] 3. Part 2 — feature engineering (clean/tokenize, vectorizer factory) + 2–3-example demo
      on train and test.
- [ ] 4. Part 3 — `NaiveBayesTextClassifier` from scratch (Multinomial + Bernoulli) + parity
      check vs `sklearn.MultinomialNB`.
- [ ] 5. Part 6a — grid-search + 5-fold CV loop, `results_df` of all permutations, best
      permutation shown separately.
- [ ] 6. Part 4 + Part 5 — retrain winning config on full trainset; evaluate on test set;
      first 5 predictions; confusion matrix + top-words plots.
- [ ] 7. Final polish — conclusions cell, Excel-values appendix, README submission links,
      re-run notebook top-to-bottom with outputs saved.
- [ ] 8. Record video (~5 min).

## Decisions log

- 2026-09-02 — Assignment type: **text analysis (NLP)**. Algorithm: **Naive Bayes from scratch**
  (Multinomial + Bernoulli variant). Scope: full Parts 1–5 + light 6a; skip 6b/6c.
- 2026-09-02 — Repo: `Tomer-Raz/machine-learning-hit`, working copy at
  `machine_learning/machine-learning-hit/`. Commit identity `Tomer-Raz <tomer532010@gmail.com>`,
  no Claude attribution. `gh auth switch --user Tomer-Raz` before pushing.
- 2026-09-02 — Considered TripAdvisor Hotel Reviews (20k, single file, would need our own
  split). **Switched to IMDB 50K** (`atulanandjha/imdb-50k-movie-reviews-test-your-bert`)
  because it ships a real pre-split 25k/25k with labels on both sides.
- 2026-09-02 — Learning type: **binary** (pos/neg sentiment). Metric: **F1 macro-average**
  (balanced binary, no central class). Delivery: **video**. Student cell: `Tomer R.` / `5130`.

## Open questions for the user

- (none blocking) — grid size, kagglehub-vs-manual download, and Bernoulli-in-grid are
  implementation choices tracked in `ASSIGNMENT.md` §9.
- Video: still to be recorded once the notebook is done.
