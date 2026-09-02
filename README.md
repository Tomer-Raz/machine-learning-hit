# machine-learning-hit

HIT intro Machine Learning course assignment — an end-to-end **supervised-learning pipeline** for
a **text-analysis (NLP)** problem, with the learning algorithm (**Naive Bayes**) **implemented
from scratch**.

> Status: in progress. See [`docs/PROGRESS.md`](docs/PROGRESS.md).

## Problem

Binary **sentiment classification** of movie reviews: given the text of an IMDB review, predict
whether it is **positive** or **negative**. Data is the Stanford Large Movie Review Dataset
(`aclImdb`) via Kaggle — 50,000 highly polar reviews, shipped **pre-split** into 25,000 train and
25,000 test, balanced 50/50, with labels on both sides. We build the feature pipeline
(cleaning → tokenization → Bag-of-Words / TF-IDF) and a hand-implemented Naive Bayes classifier,
tune it with 5-fold cross-validated grid search on the training set, and report quality on the
untouched test set.

- **Task:** binary text classification (positive vs negative)
- **Algorithm:** Naive Bayes (Multinomial, with a Bernoulli variant), hand-implemented
- **Quality metric:** F1 macro-average

## What's in here

| Path | What |
|---|---|
| `notebook.ipynb` | The assignment — all code, explanations, outputs, and visualisations (view directly on GitHub, no download needed) |
| `ml.md` | Original assignment brief |
| `docs/ASSIGNMENT.md` | Decoded spec, point map, notebook skeleton, algorithm & CV specs |
| `docs/PROGRESS.md` | Progress checklist |
| `data/` | IMDB data (fetched by the notebook; not committed) |

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebook.ipynb
```

Or open `notebook.ipynb` in [Google Colab](https://colab.research.google.com/).

## Dataset & credit

IMDB 50K Movie Reviews — https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert
(License: GNU LGPL 3.0). Please cite:

> Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011).
> *Learning Word Vectors for Sentiment Analysis.* ACL 2011.

## Submission links

| Item | Link |
|---|---|
| Repository | https://github.com/Tomer-Raz/machine-learning-hit |
| Kaggle dataset | https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert |
| Video | _TBD_ |
