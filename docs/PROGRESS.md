# Progress

Living checklist. Update at the end of every work session; keep the "current status" block in
`../CLAUDE.md` in sync.

## Current status

- **Milestone 1 — repo scaffold + docs: DONE.**
- Notebook not started. Dataset not chosen.
- **Next action:** pick the Kaggle dataset with the user (criteria in `ASSIGNMENT.md` §5), then
  build Part 1 of `notebook.ipynb`.

## Milestones

- [x] 1. Repo scaffold: `CLAUDE.md`, `README.md`, `requirements.txt`, `.gitignore`, `docs/`.
- [ ] 2. Dataset chosen + Part 1 (load train/test, `.head()` of each, one-paragraph description,
      class-balance chart, student-details + AI-prompts cells).
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
- [ ] 8. Record video (~5 min) or present in class.

## Decisions log

- 2026-09-02 — Assignment type: **text analysis (NLP)**. Algorithm: **Naive Bayes from scratch**
  (Multinomial + Bernoulli variant). Scope: full Parts 1–5 + light 6a; skip 6b/6c.
- 2026-09-02 — Repo: `Tomer-Raz/machine-learning-hit`, working copy at
  `machine_learning/machine-learning-hit/`. Commit identity `Tomer-Raz <tomer532010@gmail.com>`,
  no Claude attribution. `gh auth switch --user Tomer-Raz` before pushing.

## Open questions for the user

- Which Kaggle dataset? (needs: English, text + label, small, test set with labels)
- Group members' details for the student-details cell (first name + last initial + last 4 ID digits).
- Delivery: record a video, or present in class?
