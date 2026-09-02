# CLAUDE.md — machine-learning-hit

Orientation for Claude Code sessions on this repo. **Read this first, then `docs/PROGRESS.md`.**

## What this is

HIT (Holon Institute of Technology) intro **Machine Learning** course assignment: build an
end-to-end **supervised-learning pipeline** on a Kaggle **NLP** dataset, with the learning
algorithm (**Naive Bayes**) **implemented from scratch**. Deliverables: one self-contained
notebook with committed outputs + a ~5-minute explainer video (or an in-class presentation).

**Grading is primarily on process and understanding, not on the model score.**

- Original assignment brief (machine-translated from Hebrew, garbled in places): `ml.md`
- Decoded spec, point map, notebook skeleton, algorithm/CV specs: `docs/ASSIGNMENT.md`
- Living status + checklist: `docs/PROGRESS.md`

## Current status / next step

> **Status:** milestone 1 done — repo scaffolded, docs written. Notebook not started.
> **Next:** choose the Kaggle dataset with the user (criteria in `docs/ASSIGNMENT.md` → "Dataset
> selection"), then build Part 1 of `notebook.ipynb`.

Update this block and `docs/PROGRESS.md` at the end of every work session.

## Decisions locked

| | |
|---|---|
| Assignment type | Text analysis (NLP) |
| Learning type | Classification (expected binary) |
| Algorithm (implement from scratch) | Naive Bayes — Multinomial, plus a Bernoulli variant |
| Dataset | TBD — pick from https://www.kaggle.com/datasets?tags=13204-NLP |
| Scope | Core Parts 1–5 in full + a light Part 6a (small grid-search / 5-fold CV). Skip 6b, 6c. |
| Deliverable | One `notebook.ipynb`, all code inline, outputs committed |

## Conventions (important)

- **Git identity is repo-local** and already set: `Tomer-Raz <tomer532010@gmail.com>`.
  (The machine's global identity is a different, work account — do not use it here.)
- **Never** add a `Co-Authored-By: Claude` trailer or any Claude attribution to commits.
- Before any push: `gh auth switch --user Tomer-Raz` (multiple gh accounts exist on this machine;
  `Tomer-Raz` is not always the active one).
- Commit messages: plain, imperative subject line (e.g. `Add Part 2 feature engineering`).
- **Notebook-first.** `notebook.ipynb` is the single source of truth for the algorithm and
  feature-engineering code — the brief requires the code to be visible in the notebook.
  `src/` and `tests/` are optional iteration scratch; if added, keep them a copy, not a fork.
- Implement Naive Bayes **ourselves**. `scikit-learn` is allowed **only** for: vectorizers
  (`CountVectorizer` / `TfidfVectorizer`), `StratifiedKFold`, the metric functions, and a one-off
  parity check against `MultinomialNB`. Never as the model.
- `random_state = 42` everywhere.

## Hard requirements easy to miss

- Load a **train set** and a **test set**; show `.head()` of **each**. Do **not** re-split.
  If the chosen dataset is single-file, do **one** stratified 80/20 split at the very top and
  treat those halves as fixed from then on.
- The **test set must have labels** — Part 5 scores it. This rules out competition-style
  `test.csv` files that ship without a target column.
- **5-fold cross-validation runs only inside the trainset.** The test set is touched once, at the end.
- Feature engineering must be shown **step-by-step on 2–3 train + 2–3 test examples**
  (raw text → final vector), and again after the winning config is picked (Parts 4 & 5).
- **No leakage:** fit the vectorizer / IDF / vocabulary pruning **inside each CV fold** on the
  training portion only — never on the whole trainset before CV, never on the test set.
- The Naive Bayes class exposes hyperparameters (`alpha`, `fit_prior`, `model_type`) + `fit` +
  `predict`, and the notebook walks through the math.
- Mandatory cells: **student details** (per member: first name + first letter of last name + last
  4 digits of ID) and an **AI-prompts** cell (the prompts used, links/resources, and why).

## Quality metric (decoded from the garbled brief)

| Problem | Metric |
|---|---|
| Regression | R² |
| Multi-class, or binary without a single "central" class | F1 macro-average |
| Binary with one positive class of interest (e.g. spam) | **F1 of the positive class only** ← expected case |

Use the *same* metric per-fold in CV, for model selection, and on the final test set.

## Repo layout

```
ml.md                  original assignment brief (kept as-is)
CLAUDE.md              this file
README.md             human-facing overview + submission links
requirements.txt       Python deps
notebook.ipynb         THE deliverable (created later)
data/                  train/test CSVs — small ones committed; otherwise a download script + gitignore
docs/
  ASSIGNMENT.md         decoded spec, point map, notebook skeleton, NB spec, grid-search/CV spec
  PROGRESS.md           living checklist — update every session
src/  tests/            optional iteration scratch (may not exist)
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebook.ipynb        # or open notebook.ipynb in Google Colab
```
A data-download step is added to the notebook once the dataset is chosen.

## Submission checklist (fill links into README.md when ready)

- [ ] Kaggle dataset URL
- [ ] Repository URL — https://github.com/Tomer-Raz/machine-learning-hit (public)
- [ ] Video URL — YouTube/unlisted, viewable without download (or present in class)
- [ ] Shared Excel row: assignment type · learning type · algorithm · dataset name · dataset URL · video URL · repo URL
