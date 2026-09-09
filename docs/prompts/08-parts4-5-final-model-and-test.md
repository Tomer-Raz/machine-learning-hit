# 08 — Parts 4 & 5: final training and the one-shot test evaluation

**Stage:** notebook Parts 4 and 5.
**Produced:** the full-training-set refit, `trace_example`, the test-set traces, the first five
predictions, the headline test score, the confusion matrix and the top-terms chart.

---

## Prompt 1 — the refit

```text
Write Part 4. Take the winning configuration assembled at the end of Part 6a and re-fit it - both
the vectorizer and my Naive Bayes classifier - on the ENTIRE training set.

Explain in the markdown why this step exists at all: cross-validation only ever trained on four
fifths of the data at a time, so the final model should be re-fitted on everything available before
it faces the test set.

The output cell should print, so it is all verifiable from the committed output:
- how many training reviews it was fitted on;
- the winning feature configuration and model hyper-parameters, printed as dictionaries;
- the resulting vocabulary size;
- the macro-F1 on the training set itself, labelled clearly as RESUBSTITUTION so nobody mistakes it
  for a generalisation estimate;
- the cross-validated estimate from Part 6a next to it, for comparison.

The test set is still untouched at this point. Do not transform it in this cell beyond preparing
the matrix, and do not score anything on it.
```

**What came back:** the Part 4 cell. Output: 24,904 training reviews, vocabulary 30,000,
resubstitution macro-F1 0.9068 against the cross-validated 0.8779 — the gap between those two being
itself a teaching point about why we do not report training scores.

---

## Prompt 2 — re-showing the feature engineering under the chosen configuration

```text
The brief requires the feature-engineering demonstration to be repeated once the winning
configuration is known - not just the generic version from Part 2.

Write a helper trace_example(text, vec, clf, k=10) that, for one review, returns: the cleaned text,
the token list produced under the WINNING configuration's stemming and stop-word settings, the
number of non-zero features in its vector, and the k features with the largest absolute signed
contribution to the decision.

For the contributions, use the log-likelihood ratio between the two classes - feature_log_prob_ for
the positive class minus that for the negative class - multiplied by the feature's value in this
document. Positive means it pushed the prediction towards "positive", negative towards "negative".
Print the signed value next to each term so the direction is visible.

Then run it on the same three example reviews from Part 2b, and print each one's predicted
probability of being positive alongside its actual label.

This should let a viewer watch one specific review turn into a decision, with the individual words
that drove it.
```

**What came back:** `trace_example` and the Part 4 trace cell. Reading its output was what surfaced
something I would not have noticed otherwise: the model leans on movie-specific vocabulary — actor
and character names like *felix*, *lemmon*, *matthau* — which later became the explanation for why
the test score sits below the cross-validated one.

---

## Prompt 3 — Part 5, the test set, used exactly once

```text
Write Part 5. This is the ONLY place in the notebook where the test set is scored, and the markdown
should say so explicitly.

Requirements:
- Use the vectorizer fitted in Part 4 to TRANSFORM the test set. Never fit or fit_transform on test
  data. Make that visible in the code, not just claimed in prose.
- Run the feature-engineering trace on three held-out test reviews, chosen by fixed index, showing
  the same raw -> cleaned -> tokens -> contributions chain, with predicted probability and predicted
  label against the actual label. Pick indices that are interesting if possible - I would rather
  show a near-miss than three easy ones.
- Show the FIRST FIVE test predictions as a table: truncated review text, actual label, predicted
  label, and predicted probability of positive. The brief asks specifically for the first five.
- Report the headline quality index on the full test set using my score() helper, printed next to
  the cross-validated estimate from Part 6a for comparison.
- Print a full classification_report with per-class precision, recall and F1 as supporting detail.
- Two figures side by side: a confusion-matrix heatmap with the classes labelled neg/pos, and a
  horizontal bar chart of the most predictive terms per class by log-likelihood ratio - the top 15
  each way, coloured by direction.
```

**What came back:** the Part 5 cells. Test macro-F1 **0.8648**; the three traced test reviews
happened to include one genuine near-miss (a sarcastic review opening with the word "great",
predicted positive at 0.513 when the true label is negative), which is a far better thing to have on
camera than three confident hits.

---

## Prompt 4 — making the conclusions honest (composite)

```text
My test macro-F1 is 0.8648, the cross-validated estimate was 0.8779, and the resubstitution score
on the training set was 0.9068. The conclusions cell currently says the cross-validated estimate
and the test score "agree closely". That is overstating it - the gap is about 1.3 points.

Rewrite the conclusion to be accurate. Specifically:

- State the ordering of the three numbers and explain why that ordering is the EXPECTED, healthy
  one rather than something to apologise for.
- Give the two real reasons for the gap: that cross-validation is mildly optimistic because the
  configuration was chosen by its cross-validated score, and that the Stanford split deliberately
  keeps the films in the test set disjoint from those in the training set - which connects to the
  movie-specific vocabulary visible in my traced examples.
- Say what the WARNING sign would have looked like, i.e. what it would have meant if the test score
  had come out higher than the cross-validated one.
- Add a short "what I would try next" list.

Do not flatter the result. If a grader asks me about this gap on camera, I want the notebook to
already contain the answer.
```

**What came back:** the rewritten conclusions cell, which now states the drop, explains it, and
names the opposite pattern as the thing that would have indicated leakage or test-set tuning.

**Why I asked for this:** an overstated "they agree closely" is the kind of small dishonesty a
grader notices immediately when the numbers are printed two cells above.
