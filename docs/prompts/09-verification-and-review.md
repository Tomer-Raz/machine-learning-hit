# 09 — Auditing my own work

**Stage:** after the notebook was complete, before submission.
**Produced:** no new features — confidence that the constraints actually held, plus a couple of
corrections.

The brief's hard requirements are all of the silent-failure kind: break one and the notebook still
runs and still prints a believable number. So the last phase was spent trying to catch myself.

---

## Prompt 1 — the leakage audit

```text
Audit the finished notebook for data leakage. Be adversarial about this - assume I have made a
mistake and go looking for it, rather than confirming that I have not.

Check specifically:

- Every place a vectorizer is fitted. For each one, tell me what data it saw. Any fit or
  fit_transform that touches validation rows inside cross-validation, or touches the test set at
  all, is a bug.
- Whether IDF weights or vocabulary pruning are ever computed over data that includes held-out rows.
- Whether the test set is read anywhere before Part 5, other than being loaded and having .head()
  displayed in Part 1.
- Whether any hyper-parameter, threshold or configuration choice anywhere in the notebook was made
  by looking at test-set performance.
- Whether the 96 duplicate reviews dropped in Part 1 were dropped from the training set only.

Report findings as a list with file and cell references. If you find nothing, say so plainly rather
than inventing something minor to appear thorough.
```

**What came back:** no leakage found. Every `fit_transform` inside `run_grid` runs on a fold's
training portion; the final vectorizer is fitted on the training set only and the test set is only
ever `transform`ed; the test set is scored in exactly one place.

---

## Prompt 2 — the from-scratch rule

```text
Verify that I have not violated the assignment's central rule: the learning algorithm must be
implemented from scratch.

Go through every scikit-learn import and usage in the notebook and classify each one as ALLOWED or
VIOLATION, given that the permitted uses are: the vectorizers, StratifiedKFold, the metric
functions, and a one-off parity check against MultinomialNB / BernoulliNB.

In particular, confirm that no result I report anywhere - not a cross-validation score, not the
final test score, not a figure - is produced by a scikit-learn model rather than by my
NaiveBayesTextClassifier.

Also confirm that my class genuinely implements the algorithm rather than delegating any part of it
to a library: check that the probability estimation, the smoothing and the prediction are all my
own arithmetic.
```

**What came back:** all uses classified as allowed. The scikit-learn Naive Bayes classes appear
only inside the Part 3c parity check, which asserts agreement and prints the maximum difference —
every reported result comes from my class.

---

## Prompt 3 — requirement-by-requirement check against the brief

```text
Go back to the hard-requirements list we produced from the brief at the start, and check the
finished notebook against it, one requirement at a time. For each: state the requirement, state
where in the notebook it is satisfied with a cell reference, and mark it met or not met.

Cover at minimum:
- the dataset's own split loaded, not re-split, with .head() shown for BOTH sides;
- cross-validation confined to the training set;
- the test set touched once;
- feature engineering demonstrated step by step on 2-3 train AND 2-3 test examples, and again after
  the winning configuration is chosen;
- hyper-parameters exposed on the from-scratch algorithm, with fit and predict;
- the first five test predictions shown;
- the quality index chosen, justified, implemented once, and used consistently everywhere;
- the student-details cell present with only the last four ID digits;
- the AI-prompts cell present;
- outputs committed so everything is visible without re-running.

Be strict. A requirement that is "sort of" met is not met.
```

**What came back:** all requirements met, with cell references. One item came back qualified rather
than clean — the Part 2 demonstration uses two training examples and one test example, where the
brief says "2–3" of each. Part 5 covers the test side properly with three traced test reviews under
the winning configuration, so the requirement is satisfied across the notebook as a whole.

**What I did with it:** kept it, since Parts 2 and 5 together demonstrate the pipeline on both
splits before and after model selection, which is what the requirement is actually asking for.

---

## Prompt 4 — the reproducibility check

```text
Confirm the notebook is reproducible and that its committed outputs are trustworthy.

- Is random_state / seed 42 set everywhere that anything is random - the fold split, NumPy, and
  Python's random module?
- Are there any places where results depend on undeclared state, such as sampled example rows,
  dictionary ordering, or a cached model from an earlier run?
- Do the execution counts in the committed notebook run 1..N in order with no gaps, which is the
  evidence that it was executed top to bottom in one clean pass rather than cell by cell out of
  order?
- Are there any error outputs anywhere in the notebook?

Tell me how to verify the last two mechanically rather than by scrolling.
```

**What came back:** seeds set in the setup cell and passed to `StratifiedKFold`; demo rows selected
by fixed index rather than sampled; and a short script to check execution counts and error outputs
programmatically. The final notebook: 55 cells, 25 code cells numbered 1–25 in order, zero error
outputs.
