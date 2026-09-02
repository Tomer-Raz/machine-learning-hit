# Assignment — decoded spec & execution reference

The original brief (`../ml.md`) is a machine translation from Hebrew and is garbled in places.
This document is the working interpretation plus our execution plan. If the two ever conflict,
re-read `../ml.md` and update this file.

---

## 1. The task

Run a **supervised-learning flow** (classification or regression, per the dataset) on a Kaggle
dataset for a **text-analysis (NLP)** problem, **implementing the learning algorithm from
scratch**. Deliver:

1. **Code notebook** (Colab / Jupyter / GitHub) — outputs visible **without re-running**;
   markdown notes throughout; all visualisations and results included.
2. **Video** (~5 min, YouTube/unlisted, viewable without download) **or** present in class.
   Introduce group members at the start; walk through every part showing code + output +
   explanation, assuming the viewer doesn't know the material. Presenting live earns a bonus.
3. **Kaggle dataset link.**
4. **Repository link** — viewable and results-viewable without download.
5. **Shared Excel row:** assignment type · learning type · algorithm implemented · dataset name ·
   dataset URL · video URL · repository URL.

Assessment is **primarily on process**, not on model quality.

---

## 2. Our choices

| Decision | Choice |
|---|---|
| Assignment type | Text analysis (NLP) |
| Learning type | Binary classification — positive vs negative movie-review sentiment |
| Algorithm implemented from scratch | Naive Bayes — Multinomial + Bernoulli variant |
| Dataset | **IMDB 50K Movie Reviews** — `atulanandjha/imdb-50k-movie-reviews-test-your-bert` (https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert). Stanford `aclImdb`; tagged `nlp` + `text`; ships pre-split **25k train / 25k test**, both labeled, balanced 50/50. |
| Quality metric | F1 macro-average (see §3) |
| Scope | Parts 1–5 in full + a light Part 6a. Skip 6b, 6c. |
| Delivery | ~5-min video (no in-class presentation) |
| Student details cell | `Tomer R.` + last-4 ID `5130` |

---

## 3. Quality metric (decoded)

- **Regression** → R² (`sklearn.metrics.r2_score`).
- **Multi-class**, or binary **without a single "central" class** → F1 macro-average
  (`f1_score(..., average="macro")`).
- **Binary with one "central" (positive) class of interest** (e.g. spam vs ham) →
  F1 of the positive class only (`f1_score(..., pos_label=<positive>)`).

**Our case → F1 macro-average.** IMDB sentiment is a balanced binary problem (50% pos / 50% neg)
with no single class of interest, so the middle rule applies. Note in the notebook that on this
balanced data F1-on-positive and accuracy come out near-identical.

The same metric is used per-fold in cross-validation, to pick the best permutation, and on the
final test set. Implement it once as a `score(y_true, y_pred)` helper.

---

## 4. Point map

Translation numbering is loose — where the brief says "section 5" it often means the extension
in section 6.

| Part | Content | Max pts |
|---|---|---|
| 1 — Introduction | Student details cell; AI-prompts cell; one-paragraph problem + dataset description; load train + test; show `.head()` of each | 5 |
| Quality index | Pick & justify the correct metric, implement it, use it consistently | 10 |
| 2 — Feature engineering | Implement text feature-extraction techniques taught in class; demonstrate on 2–3 train + 2–3 test examples | 35 |
| 3 — Learning algorithm from scratch | Hyperparameters + `fit` + `predict` + written/oral explanation. (+5 bonus only for genuinely complex algorithms like ANN — N/A for Naive Bayes) | 35 |
| 4 — Training | Retrain the winning config on the **full trainset**; re-show the 2–3-example feature-engineering trace | 5 |
| 5 — Prediction & evaluation on test set | Feature-engineering trace on 2–3 test examples; **first 5 predictions**; quality index on the test set | 10 |
| 6a — Extension: grid-search + k-fold CV | Grid search over the Cartesian product of feature-engineering permutations × algorithm hyperparameters, wrapped in **5-fold CV**; results **DataFrame** of every permutation with its mean CV score; **best permutation shown separately**. Up to 5 pts per feature-engineering technique experimented with (max 20 if only FE) + up to 5 pts per hyperparameter (max 10 if only HP) | 25 |
| 6b — Special data adjustments *(skipping)* | Imbalanced-data handling: under-/over-sampling, SMOTE | 10 |
| 6c — Explainability *(skipping)* | Model-internal importances or post-hoc methods (e.g. SHAP) | 10 |

Base target ≈ 100 pts (Parts 1–5) plus partial 6a.

---

## 5. Hard constraints

- Load a **train set** and a **test set**; show `.head()` of **each**. Do **not** re-split, and
  do **not** merge-then-resplit.
  - **This dataset ships its own split.** Load `imdb_master.csv` (columns `type`, `review`,
    `label`), drop `label == 'unsup'` rows (~50k unlabeled), then split into train/test **by the
    `type` column** — that is the official Stanford split (25k train / 25k test, both labeled,
    balanced). Optionally persist to `data/train.csv` + `data/test.csv` and load those.
  - Map labels to binary: `pos → 1`, `neg → 0` (or keep strings — just be consistent).
- The **test set has labels** here (`pos`/`neg`) — Part 5 scores it directly.
- **5-fold cross-validation runs only inside the trainset.** The test set is touched once, at the end.
- Feature engineering shown **step-by-step on 2–3 concrete train examples and 2–3 test examples**
  (raw → transformed), and again after the winning config is chosen (Parts 4 & 5).
- **No leakage:** the vectorizer, IDF weights, and any vocabulary pruning
  (`min_df` / `max_df` / `max_features`) are fit **inside each CV fold** on the training portion
  only — never on the whole trainset before CV, never on the test set.
- The scratch algorithm exposes **hyperparameters** ("at least as taught in class") and provides
  a **train/`fit`** and a **`predict`** function; the notebook explains the algorithm.
- Mandatory cells: **student details** (per member: first name + first letter of last name + last
  4 digits of ID) and an **AI-prompts** cell (prompts used, links/resources, purpose).

---

## 6. Notebook skeleton (`notebook.ipynb`)

```
0.  Title + student details (per member: First N., last-4 ID)
1.  PART 1 — Introduction
    1a. Markdown: AI prompts + links used + purpose
    1b. Markdown: problem + dataset description (one tight paragraph)
    1c. Imports, seeds, nltk downloads
    1d. Load train + test; df_train.head(), df_test.head(); shapes; class-balance bar chart
2.  QUALITY INDEX
    2a. Markdown: which metric and why (F1-positive vs F1-macro vs R²)
    2b. score(y_true, y_pred) helper — used everywhere
3.  PART 2 — Feature engineering
    3a. Markdown: explain each technique for a lay viewer
        (tokenisation, lowercasing, stopwords, stemming/lemmatisation,
         Bag-of-Words / CountVectorizer, TF-IDF, n-grams, min_df/max_df/max_features)
    3b. clean_text() + tokenize()  (regex: lowercase, strip punctuation/digits/URLs/HTML)
    3c. build_vectorizer(kind, ngram_range, use_stemming, min_df, ...) factory
    3d. DEMO: 2–3 raw train docs + 2–3 raw test docs →
        cleaned → tokens → tokens after stopword/stemming → vocab indices →
        final count / tf-idf vector (non-zero entries as a small DataFrame)
4.  PART 3 — Naive Bayes from scratch
    4a. Markdown: derive Multinomial NB (priors, Laplace/Lidstone smoothing, log-space argmax);
        then the Bernoulli variant
    4b. class NaiveBayesTextClassifier (spec in §7)
    4c. Sanity check: our feature_log_prob_ / predictions vs sklearn MultinomialNB on a small slice
    4d. Peek: top-N most informative words per class
5.  PART 6a (light) — Grid search + 5-fold CV
    5a. Markdown: explain k-fold CV, grid search, and train/validation leakage
    5b. Own CV loop over the Cartesian product (spec in §8)
    5c. results_df: one row per permutation, param columns + mean_cv_score + std_cv_score, sorted
    5d. Print the best permutation + its score separately
6.  PART 4 — Retrain the winning config on the FULL trainset
    6a. Re-show the 2–3-example feature-engineering trace under the chosen config
    6b. Fit vectorizer + NB on all of train
7.  PART 5 — Predict + evaluate on the TEST set
    7a. Feature-engineering trace on 2–3 test examples
    7b. First 5 test predictions (predicted label vs actual, + predicted class probability)
    7c. Quality index on the full test set
    7d. Confusion-matrix heatmap + top-predictive-words bar chart
8.  Conclusions — what worked, what we'd try next (nods to the skipped 6b/6c)
9.  Appendix — the Excel field values
```

---

## 7. `NaiveBayesTextClassifier` — implementation spec (Part 3)

sklearn-compatible surface so it drops into our own CV loop and the parity check.

```python
class NaiveBayesTextClassifier:
    def __init__(self, alpha=1.0, fit_prior=True, model_type="multinomial"):
        # alpha:      Laplace/Lidstone smoothing (tune: 0.01, 0.1, 0.5, 1.0)
        # fit_prior:  learn P(c) from data vs assume uniform
        # model_type: "multinomial" | "bernoulli"  (Bernoulli = the explained variant)

    def fit(self, X, y):     # X: (n_docs, n_features) counts (scipy sparse OK); y: labels
        # classes_, class_count_, class_log_prior_
        # multinomial: feature_log_prob_[c,i] = log( (count(i,c)+alpha) / (sum_i count(i,c) + alpha*V) )
        # bernoulli:   binarise X; feature_log_prob_[c,i] = log( (df(i,c)+alpha) / (N_c + 2*alpha) )
        return self

    def predict_log_proba(self, X):
        # multinomial: class_log_prior_ + X @ feature_log_prob_.T
        # bernoulli:   include the (1 - p) term for absent features
    def predict_proba(self, X):   # exp + normalise via log-sum-exp
    def predict(self, X):         # argmax over classes_
```

- Work entirely in **log space**; use **log-sum-exp** for `predict_proba`.
- Accept scipy **sparse** input from the vectorizers (`X @ M.T` works sparse; binarise via `X > 0`).
- Hyperparameters exposed for the grid: `alpha`, `fit_prior`, `model_type`.
- **Parity check** (reference only, allowed): with `model_type="multinomial"`, our
  `feature_log_prob_`, `class_log_prior_`, and predictions should match
  `sklearn.naive_bayes.MultinomialNB(alpha=...)` to floating-point tolerance on a small slice.
- Cheap explainability to keep: `top_features_per_class(n)` from `feature_log_prob_`, and the
  per-word log-likelihood-ratio contribution for one explained prediction.

---

## 8. Grid search + 5-fold CV — spec (Part 6a, light)

- Split: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the **trainset only**.
- Grid = Cartesian product, kept small so it runs in minutes:
  - **Feature-engineering axis** (≥2 techniques): `vectorizer ∈ {BoW, TF-IDF}` ×
    `ngram_range ∈ {(1,1), (1,2)}` × `use_stemming ∈ {False, True}`.
  - **Hyperparameter axis** (≥1): `alpha ∈ {0.1, 0.5, 1.0}` (optionally
    `model_type ∈ {multinomial, bernoulli}`).
  - ⇒ ~24–48 permutations; trim if slow.
- Per permutation: for each of the 5 folds — fit the vectorizer on the 4 training folds only,
  transform the held-out fold, fit `NaiveBayesTextClassifier`, predict, score the validation
  fold with the quality metric. **Average the 5 fold scores** (keep std too).
- Output `results_df` (all permutations: params + `mean_cv_score` + `std_cv_score`), sorted
  descending; then print the single best permutation + its score separately.
- Part 4 rebuilds the pipeline from the best row and fits it on the entire trainset.
- Part 5 `transform`s (no re-fit) the entire test set and reports the metric.

---

## 9. Decisions — resolved & open

Resolved: dataset (IMDB 50K, `atulanandjha/...`), split model (use the dataset's own `type`
column), class structure (binary pos/neg), metric (F1-macro), delivery (video), student details
(`Tomer R.` / `5130`).

Still open / in-session:
1. Feature-engineering demo — plan: hand-roll a tiny tokenizer/counter for the 2–3-example demo
   (fully printable), use sklearn vectorizers for the full runs.
2. Bernoulli NB as a second variant in Part 3 + the grid — yes (cheap, strengthens both).
3. Grid size for 6a — start ~24–48 permutations; trim if the ~50k-row CV is slow.
4. `src/` + `tests/` mirror — optional; add only if iteration speed calls for it.
5. `kagglehub` download vs manual CSV drop into `data/` — decide when wiring Part 1.
