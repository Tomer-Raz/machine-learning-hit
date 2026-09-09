# CLAUDE.md — machine-learning-hit

Orientation for Claude Code sessions on this repo. **Read this first, then `docs/PROGRESS.md`.**

## What this is

HIT (Holon Institute of Technology) intro **Machine Learning** course assignment: build an
end-to-end **supervised-learning pipeline** on a Kaggle **NLP** dataset, with the learning
algorithm (**Naive Bayes**) **implemented from scratch**. Deliverables: one self-contained
notebook with committed outputs + a ~5-minute explainer video.

**Grading is primarily on process and understanding, not on the model score.**

- Original assignment brief (machine-translated from Hebrew, garbled in places): `ml.md`
- Decoded spec, point map, notebook skeleton, algorithm/CV specs: `docs/ASSIGNMENT.md`
- Living status + checklist: `docs/PROGRESS.md`

## Current status / next step

> **Status: the notebook is COMPLETE.** `notebook.ipynb` is 55 cells, executes top-to-bottom with
> outputs committed, zero errors — Parts 1, 2, 3, 4, 5, the quality-index section and the 6a
> grid-search extension are all done.
>
> Final result: winning pipeline **TF-IDF · uni+bigrams · Porter stemming · Multinomial NB ·
> `alpha=0.1` · `min_df=5` · `max_features=30000`** → cross-validated macro-F1 **0.8779** on the
> training set, **0.8648** on the untouched test set (train resubstitution 0.9068).
>
> **Next:** record the ~5-minute video from `docs/VIDEO_SCRIPT.md`, then paste its URL into
> `README.md`, `docs/PROGRESS.md` and the notebook's Excel-appendix cell (via the builder) and
> re-run the notebook once.
>
> Env notes: `.venv` has the full stack; `kagglehub` anonymous download works (no token) and the
> files are cached locally; `truststore.inject_into_ssl()` in the setup cell handles the corporate
> TLS proxy. A full build+execute takes ~7 minutes on an idle machine:
> `.venv/bin/python tools/build_notebook.py && .venv/bin/jupyter nbconvert --to notebook
> --execute --inplace --ExecutePreprocessor.timeout=14400 notebook.ipynb`

Update this block and `docs/PROGRESS.md` at the end of every work session.

## Decisions locked

| | |
|---|---|
| Assignment type | Text analysis (NLP) |
| Learning type | **Binary** classification — positive vs negative movie-review sentiment |
| Algorithm (implement from scratch) | Naive Bayes — Multinomial, plus a Bernoulli variant |
| Dataset | **IMDB 50K Movie Reviews** — Kaggle `atulanandjha/imdb-50k-movie-reviews-test-your-bert` (https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert). Stanford `aclImdb`; **ships pre-split 25k train / 25k test, both labeled, balanced 50/50**. |
| Quality metric | **F1 macro-average** (balanced binary, no single "central" class). Note in notebook that F1-on-positive is near-identical here. |
| Scope | Core Parts 1–5 in full + a light Part 6a (small grid-search / 5-fold CV). Skip 6b, 6c. |
| Deliverable | One `notebook.ipynb`, all code inline, outputs committed; ~5-min video (no class presentation) |

## Submission facts

- **Student details cell:** `Tomer R.` · last 4 digits of ID `5130` (full ID 322525130 — do NOT put the full ID in the repo).
- **Repository:** https://github.com/Tomer-Raz/machine-learning-hit (public)
- **Dataset URL:** https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert
- **Video URL:** TBD
- Excel row needs: assignment type · learning type · algorithm · dataset name · dataset URL · video URL · repo URL

## Conventions (important)

- **Git identity is repo-local** and already set: `Tomer-Raz <tomer532010@gmail.com>`.
  (The machine's global identity is a different, work account — do not use it here.)
- **Never** add a `Co-Authored-By: Claude` trailer or any Claude attribution to commits.
- Before any push: `gh auth switch --user Tomer-Raz` (multiple gh accounts exist on this machine;
  `Tomer-Raz` is not always the active one).
- Commit messages: plain, imperative subject line (e.g. `Add Part 2 feature engineering`).
- **The notebook is generated.** Edit `tools/build_notebook.py` (each cell is an `md(...)` /
  `code(...)` call), then regenerate + execute:
  ```
  .venv/bin/python tools/build_notebook.py
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=2400 notebook.ipynb
  ```
  **Never hand-edit `notebook.ipynb`.** The generated notebook still has all code inline and
  visible, as the brief requires. Constraint in the builder: cell sources are r-strings, so no
  triple-quotes inside them (use `#` comments in code cells, not docstrings).
- `src/` and `tests/` are optional iteration scratch; if added, keep them a copy, not a fork.
- Implement Naive Bayes **ourselves**. `scikit-learn` is allowed **only** for: vectorizers
  (`CountVectorizer` / `TfidfVectorizer`), `StratifiedKFold`, the metric functions, and a one-off
  parity check against `MultinomialNB`. Never as the model.
- `random_state = 42` everywhere.

## Hard requirements easy to miss

- Load the dataset's **existing train and test split**; show `.head()` of **each**. Do **not**
  re-split, and do **not** merge-then-resplit. For this dataset (version 2): load the shipped
  `train.csv` and `test.csv` (25k rows each, cols `text`/`sentiment`), rename to `review`/`label`,
  map `pos→1`/`neg→0`. Only within-train cleaning is allowed (we drop 96 exact-duplicate reviews);
  the test set is left untouched. ~123 review texts appear in both splits — noted as a caveat, not
  altered.
- The **test set has labels** here (pos/neg) — Part 5 scores it directly.
- **5-fold cross-validation runs only inside the trainset.** The test set is touched once, at the end.
  This is now enforced *structurally*, not just by intention: no test feature matrix and no `yte`
  exist anywhere before Part 5 — `X_test_final` and `yte` are created in the Part 5 cell, at the
  point of use. The Part 3c parity check compares on a held-out stratified fifth of the **training**
  set for the same reason. The only permitted pre-Part-5 contact with the test set is the four things
  the brief demands: loading it, `.head()`, the Part 1 data description, and one test review in the
  Part 2b demo. **Do not reintroduce an early `Xte`.**
- Feature engineering must be shown **step-by-step on 2–3 train + 2–3 test examples**
  (raw text → final vector), and again after the winning config is picked (Parts 4 & 5).
- **No leakage:** fit the vectorizer / IDF / vocabulary pruning **inside each CV fold** on the
  training portion only — never on the whole trainset before CV, never on the test set.
- The Naive Bayes class exposes hyperparameters (`alpha`, `fit_prior`, `model_type`) + `fit` +
  `predict`, and the notebook walks through the math.
- Mandatory cells: **student details** (`Tomer R.` + `5130`) and an **AI-prompts** cell (the
  prompts used, links/resources, and why).

## Quality metric (decoded from the garbled brief)

| Problem | Metric |
|---|---|
| Regression | R² |
| Multi-class, or binary without a single "central" class | **F1 macro-average** ← our case (balanced pos/neg) |
| Binary with one positive class of interest (e.g. spam) | F1 of the positive class only |

Use the *same* metric per-fold in CV, for model selection, and on the final test set. Implement
it once as `score(y_true, y_pred)`.

## Repo layout

```
ml.md                  original assignment brief (kept as-is)
CLAUDE.md              this file
README.md             human-facing overview + submission links
requirements.txt       Python deps
notebook.ipynb         THE deliverable (complete, executed, outputs committed)
data/                  IMDB data — download script / cell; raw files gitignored, see data/README.md
docs/
  ASSIGNMENT.md         decoded spec, point map, notebook skeleton, NB spec, grid-search/CV spec
  PROGRESS.md           living checklist — update every session
  VIDEO_SCRIPT.md       timed ~5-min script to record the video from
src/  tests/            optional iteration scratch (may not exist)
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebook.ipynb        # or open notebook.ipynb in Google Colab
```
The notebook downloads the dataset via `kagglehub` (needs Kaggle API credentials) or a manual
CSV drop into `data/`.
