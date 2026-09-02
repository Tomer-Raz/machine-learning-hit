# machine-learning-hit

HIT intro Machine Learning course assignment — an end-to-end **supervised-learning pipeline** for
a **text-analysis (NLP)** problem, with the learning algorithm (**Naive Bayes**) **implemented
from scratch**.

> Status: in progress. See [`docs/PROGRESS.md`](docs/PROGRESS.md).

## Problem

*(One-paragraph description of the learning problem and dataset goes here once the dataset is
chosen — mirrors the notebook's intro cell.)*

- **Task:** text classification
- **Algorithm:** Naive Bayes (Multinomial, with a Bernoulli variant), hand-implemented
- **Quality metric:** F1 on the positive class (binary) / F1 macro-average (multi-class)

## What's in here

| Path | What |
|---|---|
| `notebook.ipynb` | The assignment — all code, explanations, outputs, and visualisations (view directly on GitHub, no download needed) |
| `ml.md` | Original assignment brief |
| `docs/ASSIGNMENT.md` | Decoded spec, point map, notebook skeleton, algorithm & CV specs |
| `docs/PROGRESS.md` | Progress checklist |
| `data/` | Train / test data (or a script to fetch it) |

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebook.ipynb
```

Or open `notebook.ipynb` in [Google Colab](https://colab.research.google.com/).

## Submission links

| Item | Link |
|---|---|
| Repository | https://github.com/Tomer-Raz/machine-learning-hit |
| Kaggle dataset | _TBD_ |
| Video | _TBD_ |
