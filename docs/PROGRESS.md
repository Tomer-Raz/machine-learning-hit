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
- **Milestone 4 (done early) — Part 3: DONE.** `NaiveBayesTextClassifier` (Multinomial + Bernoulli,
  log-space, sparse-aware; hyper-params `alpha`, `fit_prior`, `model_type`) + a self-contained
  parity check vs scikit-learn (`max |log-prob diff| ≈ 1e-13`).
- **Milestone 3 — Part 2: DONE.** `clean_text` / `tokenize` / `build_vectorizer` factory
  (BoW vs TF-IDF, n-grams, stop-words, Porter stemming, `min_df` / `max_df` / `max_features`),
  plus a raw → cleaned → tokens → stemmed → vector trace on 2 train + 1 test review.
- **Milestone 5 — Part 6a: DONE.** Leakage-safe 5-fold CV (vectorizer re-fit inside every fold),
  36-permutation main grid, plus stop-word and `min_df` follow-up experiments. Winner:
  **TF-IDF · (1,2) · stemming · Multinomial · `alpha=0.1` · `min_df=5` · `max_features=30000`**
  at **CV macro-F1 0.8779 ± 0.0040**. Stop-word removal *hurts* (0.8684). The `min_df` sweep is
  flat (spread 4e-5) because `max_features=30000` is the binding pruning constraint.
- **Milestone 6 — Parts 4 & 5: DONE.** Re-fit on all 24,904 training reviews (vocabulary 30,000,
  resubstitution macro-F1 0.9068); test set touched exactly once → **test macro-F1 0.8648**,
  with the feature trace on 3 test reviews, the first 5 predictions, the confusion matrix and the
  top-predictive-terms chart.
- **Next action:** none outstanding — the notebook is the deliverable and it is done. Fill the
  shared Excel row from the notebook's appendix cell.

## Milestones

- [x] 1. Repo scaffold: `CLAUDE.md`, `README.md`, `requirements.txt`, `.gitignore`, `docs/`.
- [x] 2. Part 1 — data load (dataset's own `train.csv`/`test.csv`), `.head()` of each,
      one-paragraph description, class-balance chart, student-details + AI-prompts cells. Executed.
- [x] 3. Part 2 — feature engineering (clean/tokenize, vectorizer factory) + 2–3-example demo
      on train and test. Executed.
- [x] 4. Part 3 — `NaiveBayesTextClassifier` from scratch (Multinomial + Bernoulli) + parity
      check vs scikit-learn. Executed, matches to ~1e-13.
- [x] 5. Part 6a — grid-search + 5-fold CV loop, `results_df` of all permutations, best
      permutation shown separately, plus stop-word and `min_df` experiments. Executed.
- [x] 6. Part 4 + Part 5 — retrained winning config on full trainset; evaluated on test set;
      first 5 predictions; confusion matrix + top-words plots. Executed.
- [x] 7. Final polish — conclusions cell, Excel-values appendix, docs synced, notebook re-run
      top-to-bottom (55 cells, execution counts 1–25, zero errors) with outputs saved.

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
  (balanced binary, no central class). Student cell: `Tomer R.` / `5130`.
- 2026-09-09 — `min_df` tie-break: the sweep spans only 4e-5 of macro-F1 (noise) because
  `max_features=30000` already caps the vocabulary, so the winner is **pinned to `min_df=5`**
  rather than taking the raw argmax (`min_df=1`) — simpler and smaller model at an identical score.
  `WIN_CV` reports that chosen configuration's own CV score, not the max across experiments.
- 2026-09-09 — Test 0.8648 vs CV 0.8779: the ~1.3-point drop is expected and is explained in the
  conclusions cell (selection makes CV mildly optimistic; the Stanford split keeps test films
  disjoint from train films, so movie-specific vocabulary does not transfer).

- 2026-09-09 — **Test-set isolation fix.** The Part 3c parity check was comparing against
  scikit-learn on the *test* set and printing a test macro-F1 (0.8699) before any tuning, which
  contradicted Part 5's "used once" statement. It now holds out a stratified fifth of the training
  set (fit on 19,923, compare on 4,981; identical predictions, max |log-prob diff| 4.26e-13). Three
  further pre-Part-5 touches were removed at the same time: the Part 2c baseline no longer builds a
  test matrix, `yte` is no longer materialised early, and Part 4 no longer transforms the test set —
  `X_test_final` and `yte` are now created in Part 5, at the point of use. No result changed (test
  macro-F1 is still 0.8648); only the audit trail improved.

## Open questions for the user

- (none blocking) — grid size, kagglehub-vs-manual download, and Bernoulli-in-grid are
  implementation choices tracked in `ASSIGNMENT.md` §9.
