"""
Build `notebook.ipynb` for the IMDB / Naive-Bayes-from-scratch assignment.

The notebook is GENERATED from this file so it is easy to iterate on with clean diffs.
The generated notebook still contains every line of code inline and visible, as the brief requires.

Workflow:
    .venv/bin/python tools/build_notebook.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=2400 notebook.ipynb

Rules for this file:
  * never hand-edit notebook.ipynb -- edit here and regenerate
  * cell sources are r-strings; do NOT put triple-quotes inside them
    (use '#' comments in code cells instead of docstrings)
"""
from __future__ import annotations

from pathlib import Path
import nbformat as nbf

cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ==========================================================================================
# 0. Title + student details
# ==========================================================================================
md(r"""
# IMDB Movie-Review Sentiment - Naive Bayes implemented from scratch

**HIT - Machine Learning course assignment**

| Student | ID (last 4 digits) |
|---|---|
| Tomer R. | 5130 |

- **Assignment type:** text analysis (NLP)
- **Learning type:** binary classification (positive vs negative review sentiment)
- **Algorithm implemented from scratch:** Naive Bayes - Multinomial, with a Bernoulli variant
- **Quality metric:** F1 macro-average
- **Dataset:** [IMDB 50K Movie Reviews](https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert)
  (Stanford *Large Movie Review Dataset*), shipped pre-split into 25,000 train / 25,000 test, balanced 50/50.

This notebook runs a full supervised-learning flow: load the fixed train/test split, engineer text
features, implement Naive Bayes from scratch, tune it with 5-fold cross-validated grid search on the
training set only, retrain the best configuration on the whole training set, and evaluate once on the
untouched test set.
""")

# ==========================================================================================
# 1. Part 1 - Introduction
# ==========================================================================================
md(r"""
---
## Part 1 - Introduction
""")

md(r"""
### 1a. AI assistance - prompts, tools and resources used

This assignment was developed with the help of **Claude (Anthropic)**, used through *Claude Code*.
All generated code and explanations were read, checked and understood before inclusion; the Naive
Bayes derivation below is standard textbook material (Manning, Raghavan & Schuetze, *Introduction to
Information Retrieval*, ch. 13) and was cross-checked against the scikit-learn user guide.

**What the AI was used for, with representative prompts (paraphrased):**

| Purpose | Prompt (paraphrased) |
|---|---|
| Understand the (machine-translated) brief | *"Read the assignment brief and explain exactly what each part requires."* |
| Choose a suitable dataset | *"Is the IMDB 50K movie-review dataset a good fit for a from-scratch Naive Bayes text-classification task that needs a pre-split train/test set with labels on both?"* |
| Derive and implement the algorithm | *"Derive Multinomial and Bernoulli Naive Bayes and implement them from scratch with a scikit-learn-style fit/predict, Laplace/Lidstone alpha smoothing and a fit_prior option, working in log space."* |
| Design the evaluation | *"Design a leakage-safe 5-fold cross-validated grid search over feature-engineering options x Naive Bayes hyper-parameters; the vectorizer must be fit inside each fold."* |
| Explanatory writing | *"Explain TF-IDF / n-grams / Laplace smoothing for a viewer who does not know the material."* |

**Other resources:**

- scikit-learn User Guide - *Naive Bayes* and *Working with text data*.
- NLTK documentation - stopword list and the Porter stemmer.
- Stanford *Large Movie Review Dataset* page (Maas et al., 2011).
""")

md(r"""
### 1b. The learning problem and the dataset

We tackle **binary sentiment classification**: given the free text of an IMDB movie review, predict
whether the reviewer's opinion is **positive** or **negative**. The data is the Stanford *Large Movie
Review Dataset* (accessed via Kaggle): 50,000 "highly polar" English reviews - reviews rated 5 or 6
out of 10 were deliberately excluded, so every example is clearly positive (rating >= 7) or clearly
negative (rating <= 4). It arrives **already split** into 25,000 training and 25,000 test reviews,
each split perfectly balanced (12,500 positive + 12,500 negative), with a disjoint set of movies in
each. We keep this split fixed: the test set is set aside and used exactly once, at the very end.
All model selection happens by 5-fold cross-validation inside the training set.
""")

md(r"""
### 1c. Setup
""")
code(r"""
import re
import time
import random
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB as SklearnMultinomialNB  # reference parity check only

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

pd.set_option("display.max_colwidth", 200)
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", category=FutureWarning)

print("numpy", np.__version__, "| pandas", pd.__version__)
""")

md(r"""
### 1d. Load the train and test sets

The Kaggle dataset provides a single file, `imdb_master.csv`, with a `type` column marking each row
as `train` or `test` (this *is* the official Stanford split) and a `label` column with values
`pos`, `neg` or `unsup`. We drop the `unsup` (unlabelled) rows and split on `type` - **we never
build our own split**. Labels are mapped `neg -> 0`, `pos -> 1`.
""")
code(r"""
def load_imdb():
    # Return (df_train, df_test), each with columns ['review', 'label'] (label 0/1).
    # Tries kagglehub first (needs ~/.kaggle/kaggle.json), then a local data/imdb_master.csv.
    paths = []
    try:
        import kagglehub
        root = Path(kagglehub.dataset_download("atulanandjha/imdb-50k-movie-reviews-test-your-bert"))
        paths += sorted(root.rglob("imdb_master.csv"))
    except Exception as e:
        print("kagglehub unavailable (%r); trying data/imdb_master.csv" % (e,))
    paths += [Path("data/imdb_master.csv")]

    csv = next((p for p in paths if p and p.exists()), None)
    if csv is None:
        raise FileNotFoundError(
            "imdb_master.csv not found - add a Kaggle token to ~/.kaggle/kaggle.json "
            "or place the file in data/."
        )
    print("loading", csv)

    # imdb_master.csv is latin-1 encoded and carries a leading unnamed index column
    df = pd.read_csv(csv, encoding="latin-1")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"sentimenttext": "review", "text": "review", "sentiment": "label"})
    df = df[["type", "review", "label"]].dropna(subset=["review", "label"])
    df["review"] = df["review"].astype(str)

    df = df[df["label"].isin(["pos", "neg"])].copy()
    df["label"] = (df["label"] == "pos").astype(int)

    tr = df[df["type"] == "train"].drop(columns="type").reset_index(drop=True)
    te = df[df["type"] == "test"].drop(columns="type").reset_index(drop=True)
    return tr, te


df_train, df_test = load_imdb()
print("train:", df_train.shape, "  test:", df_test.shape)
""")

md(r"""**First 5 rows of the training set:**""")
code(r"""df_train.head()""")

md(r"""**First 5 rows of the test set:**""")
code(r"""df_test.head()""")

md(r"""
### 1e. A quick look at the data

Both splits should be balanced 50/50. We also glance at review length, which motivates a few
feature-engineering choices in Part 2 (HTML `<br />` tags to strip, very long reviews, and so on).
""")
code(r"""
for name, d in [("train", df_train), ("test", df_test)]:
    c = d["label"].value_counts().sort_index()
    print("%-5s n=%6d  neg(0)=%6d  pos(1)=%6d  nulls=%d  empty=%d" % (
        name, len(d), c.get(0, 0), c.get(1, 0),
        d["review"].isna().sum(), (d["review"].str.strip() == "").sum()))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for a, (name, d) in zip(ax, [("train", df_train), ("test", df_test)]):
    (d["label"].map({0: "negative", 1: "positive"}).value_counts()
        .plot.bar(ax=a, color=["#cc4444", "#44aa44"]))
    a.set_title(name + " - class balance"); a.set_xlabel(""); a.tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.show()

wl = df_train["review"].str.split().str.len()
print("\nreview length (words): min %d, median %d, mean %.0f, 95th pct %d, max %d" % (
    wl.min(), wl.median(), wl.mean(), wl.quantile(0.95), wl.max()))
print("\nexample raw review (first 400 chars):\n", df_train["review"].iloc[0][:400])
""")

# ==========================================================================================
# 2. Quality index
# ==========================================================================================
md(r"""
---
## Quality index - which metric, and why

The brief fixes the metric by problem type:

| Problem type | Metric |
|---|---|
| Regression | R-squared |
| Multi-class, **or** binary with no single "class of interest" | **F1, macro-averaged** |
| Binary with one "central" class of interest (e.g. *spam* in spam detection) | F1 of that class only |

Our problem is **binary** (positive / negative) and the classes are **balanced and equally
important** - there is no single "central" class we care about more, unlike spam detection. So the
middle row applies: **macro-averaged F1** - F1 for the positive class and F1 for the negative class,
averaged. On perfectly balanced data this sits very close to plain accuracy and to positive-class
F1, but it is the metric the brief asks for and it stays honest if a preprocessing choice ever
unbalances a validation fold.

`score()` below is the single scoring function used **everywhere**: every fold in cross-validation,
the grid-search ranking, and the final test-set report.
""")
code(r"""
def score(y_true, y_pred):
    # Macro-averaged F1 - the assignment's quality index for this problem.
    return f1_score(y_true, y_pred, average="macro")


print("score() smoke test:", round(score([0, 0, 1, 1, 1], [0, 1, 1, 1, 0]), 4))
""")

# ==========================================================================================
# assemble
# ==========================================================================================
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata.update(
    {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
)

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "notebook.ipynb"
    nbf.write(nb, str(out))
    print("wrote %s  (%d cells)" % (out, len(nb.cells)))
