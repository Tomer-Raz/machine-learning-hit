# Progress

Living checklist. Update at the end of every work session; keep the "current status" block in
`../CLAUDE.md` in sync.

## Current status

- **Milestone 1 — repo scaffold + docs: DONE.**
- **Dataset:** IMDB 50K Movie Reviews (`atulanandjha/imdb-50k-movie-reviews-test-your-bert`),
  **dataset version 2** — ships `train.csv` + `test.csv` (25k each, cols `text`/`sentiment`,
  balanced, labels on both). No token needed: `kagglehub` anonymous download works (behind the
  corporate TLS proxy, via `truststore.inject_into_ssl()` in the setup cell).
- **Milestone 2 — Part 1: DONE.** `notebook.ipynb` executes end-to-end with outputs committed.
  - Part 1: title/students, AI-prompts cell, problem+dataset paragraph, setup, `load_imdb()`
    (loads the given split, renames to `review`/`label`, maps pos/neg→1/0, drops 96 within-train
    exact-duplicate reviews → train 24904 / test 25000), `.head()` of each, EDA (class balance
    plot, 123-row train/test text overlap noted, review-length stats).
  - Quality-index section: `score()` = macro-F1.
- **Milestone 3 (early) — Part 3: DONE.** `NaiveBayesTextClassifier` (Multinomial + Bernoulli,
  log-space, sparse-aware; hyper-params `alpha`, `fit_prior`, `model_type`) + a self-contained
  parity check vs scikit-learn (`max |log-prob diff| ≈ 1e-13`).
- **Next action:** extend the builder with **Part 2 — feature engineering** (clean/tokenize
  functions, BoW/TF-IDF vectorizer factory, n-grams/stopwords/stemming, 2–3-example demo on real
  train & test rows). Then Part 6a grid search, then Parts 4 & 5. Insert Part 2 *before* Part 3
  in the builder's cell order.

## Milestones

- [x] 1. Repo scaffold: `CLAUDE.md`, `README.md`, `requirements.txt`, `.gitignore`, `docs/`.
- [x] 2. Part 1 — data load (dataset's own `train.csv`/`test.csv`), `.head()` of each,
      one-paragraph description, class-balance chart, student-details + AI-prompts cells. Executed.
- [x] 4. Part 3 — `NaiveBayesTextClassifier` from scratch (Multinomial + Bernoulli) + parity
      check vs scikit-learn. Executed, matches to ~1e-13.
- [ ] 3. Part 2 — feature engineering (clean/tokenize, vectorizer factory) + 2–3-example demo
      on train and test. *(insert before Part 3 in builder order)*
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
