# 07 — Part 6a: grid search with 5-fold cross-validation

**Stage:** notebook Part 6a (the 25-point extension).
**Produced:** `cv_folds`, `run_grid`, `results_df`, the winner selection, and the two follow-up
feature-engineering experiments.

The whole of this part hinges on one thing being right: **where the vectorizer is fitted.**

---

## Prompt 1 — the explanation, written before the loop

```text
Write the markdown that opens Part 6a. Two concepts to explain to a viewer who does not know
either: k-fold cross-validation, and grid search. Then a third thing that matters more than both:
data leakage.

For cross-validation: why a single train/validation split is a noisy estimate, what stratification
means and why it matters here, and why five folds is a reasonable default.

For grid search: what a Cartesian product of configurations is, and that I am searching two kinds
of axis at once - feature-engineering choices and model hyper-parameters.

For leakage, be concrete and specific to MY pipeline: explain that if I vectorised the whole
training set once before splitting into folds, the vocabulary and the IDF weights would have been
computed using the validation rows, so information from the held-out data would leak into
training, and my cross-validated score would come out optimistically wrong. Then state what I do
instead.

Also state plainly that the test set is not touched anywhere in this section.
```

**What came back:** the Part 6a opening markdown, including the leakage paragraph that became the
single most important sentence in my video script.

---

## Prompt 2 — the cross-validation loop

```text
Implement the grid search. Structure:

  cv_folds(y, n_splits=5, seed=42) -> list of (train_idx, val_idx), using StratifiedKFold with
  shuffle=True and the fixed seed.

  run_grid(texts, y, grid, ...) -> DataFrame, one row per permutation.

The grid is a list of (feature_kwargs, nb_kwargs) pairs.

Non-negotiable correctness requirement: inside every fold, build a FRESH vectorizer from the
feature_kwargs, fit_transform it on that fold's training portion only, and merely transform the
held-out portion. Never fit on the full training set before the fold split, and never touch the
test set anywhere in this function.

Efficiency requirement that must not compromise the above: several NB configurations share the same
feature configuration, and re-vectorising for each one would waste minutes. So group the grid by
unique feature config, vectorise the five folds once per unique feature config, then score every NB
configuration that uses those same matrices. Verify for me that this reuse does not weaken the
leakage guarantee - the matrices are still per-fold and still fitted on training rows only.

Each row of the output DataFrame must carry: every feature-engineering parameter, every model
hyper-parameter, the mean cross-validated score, the standard deviation across folds, and the list
of the five individual fold scores. Sort by mean score descending with a reset index.

Score every fold with my score() helper - macro-F1 - and nothing else.

Print progress as it goes, with timings per feature configuration, so I can see it is alive during
a long run.
```

**What came back:** `run_grid` as it stands, plus an explicit confirmation that the reuse is safe:
the grouping happens *per fold*, so each cached matrix was still fitted only on that fold's
training rows.

**What I checked myself:** I read the loop specifically for the order of `fit_transform` versus
`transform`, since that single line is the difference between an honest result and a leaked one.

---

## Prompt 3 — choosing the grid

```text
Define the actual grid. The scoring rewards up to 5 points per feature-engineering technique
experimented with and up to 5 per hyper-parameter tuned, so I want good coverage without a run that
takes hours.

Axes:
- vectorizer: Bag-of-Words vs TF-IDF
- ngram_range: (1,1) vs (1,2)
- stemming: off vs on
- model_type: multinomial vs bernoulli - but Bernoulli only makes sense with count/binary features,
  so pair it with BoW only, not with TF-IDF. Tell me if you disagree with that reasoning.
- alpha: 0.1, 0.5, 1.0

Cap max_features at 30,000 and min_df at 5 for the main grid so it stays fast.

Then keep stop-word removal and min_df OUT of the main grid and run them as separate follow-up
experiments on top of the winning configuration, so the main results table stays readable. Write
those two experiments as their own cells with their own markdown.

For the stop-word experiment, tell me in the markdown what you EXPECT to happen for sentiment
specifically, before showing the result.
```

**What came back:** the 36-permutation main grid, plus the two follow-up experiments. The
prediction written into the markdown ahead of the result — that stop-word removal would *hurt*
sentiment classification, because the standard list contains *not*, *no* and *very* — turned out to
be correct: 0.8684 with removal versus 0.8779 without.

**Why I asked for the prediction up front:** a result you predicted and then confirmed demonstrates
understanding; the same number with no prior claim just demonstrates that you ran the code.

---

## Prompt 4 — reading the results honestly (composite of a short back-and-forth)

```text
Here is the min_df sweep output: min_df of 1, 2, 5, 10 and 20 give macro-F1 of 0.877923, 0.877801,
0.877882, 0.877641 and 0.875589.

Two questions:

1. Is the difference between the top rows meaningful, given the fold-to-fold standard deviation is
   around 0.0035? My code currently takes the raw argmax, which would pick min_df=1.
2. Why is the sweep so flat? I expected vocabulary pruning to matter more than this. Is there
   something in my own configuration that makes this experiment nearly a no-op? Look at the actual
   parameters I am passing.

If there is a good reason for the flatness, I want it explained in the notebook rather than
presented as a shrug - a negative result I can explain is worth more to me than a positive one I
cannot.
```

**What came back:** the spread across the top four rows is ~4e-5 against a fold standard deviation
of ~0.0035, so it is noise, not signal — and the flatness has a specific cause: `max_features=30000`
already caps the vocabulary at the 30,000 most frequent terms, so anything surviving that cut is
common enough to clear any `min_df` in the sweep. The two pruning knobs overlap and `max_features`
is the binding one.

**What I changed as a result:**

- pinned the winner to `min_df=5` rather than taking the argmax — same score, smaller vocabulary,
  and "when two models tie, prefer the simpler one" is defensible on camera in a way that "the
  fourth decimal was higher" is not;
- made `WIN_CV` report the *chosen* configuration's own cross-validated score rather than the best
  score seen anywhere, so the number printed next to the test score belongs to the model I actually
  shipped;
- had the explanation written into the 6a-iii markdown so the flat result reads as understood
  rather than as a failed experiment.
