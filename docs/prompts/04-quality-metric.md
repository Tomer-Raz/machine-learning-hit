# 04 — The quality index

**Stage:** its own notebook section, between Parts 1 and 2.
**Produced:** the metric justification cell and the single `score(y_true, y_pred)` helper.

This part of the brief was the most badly mangled by the machine translation, and it is worth 10
points on its own, so it got its own investigation rather than a guess.

---

## Prompt 1 — reconstructing the rule from a garbled sentence

```text
This passage in ml.md decides which quality metric I must use, and the translation is close to
unreadable. Reconstruct the underlying rule.

Do this by reasoning about what rule would make sense in a machine-learning course brief, and by
telling me which standard metric-selection convention each fragment most likely corresponds to.
Give me the reconstruction as a small decision table: problem shape -> metric -> the scikit-learn
call that computes it.

Then apply the table to MY problem - balanced binary sentiment classification, positive vs
negative, with no single class I care about more than the other - and tell me which cell of the
table I land in.

Be explicit about your confidence. If more than one reading is defensible, say so and tell me which
one is safer to defend in a video.
```

**What came back:** the three-way rule now recorded in `docs/ASSIGNMENT.md` §3 and in `CLAUDE.md`:

| Problem | Metric |
|---|---|
| Regression | R² |
| Multi-class, or binary with no single "central" class | **F1 macro-average** |
| Binary with one positive class of interest (e.g. spam) | F1 of the positive class only |

My problem lands in the middle row: positive and negative reviews matter equally, so **macro-F1**.

---

## Prompt 2 — pressure-testing the choice before committing

```text
Push back on the macro-F1 choice before I lock it in.

- On a perfectly balanced binary problem, how different are macro-F1, F1 on the positive class,
  and plain accuracy in practice? If they are nearly identical here, say so - I want to know
  whether my metric choice is consequential or merely correct.
- Is there an argument that "positive review" is the central class and I should use F1 on the
  positive class instead? Make the strongest version of that argument, then tell me whether it
  holds for movie-review sentiment.
- What would change if the dataset were imbalanced?

I need to defend this on camera for maybe twenty seconds, so give me the version I can say out
loud.
```

**What came back:** macro-F1, F1-positive and accuracy come out near-identical on a 50/50 split,
so the choice is about being *right for the stated reason* rather than about moving the number; and
the "positive is central" argument fails here because there is no asymmetric cost — missing a
negative review is no worse than missing a positive one, unlike spam or disease screening.

**What I did with it:** wrote the metric-justification markdown cell around exactly that reasoning,
and noted the near-identity of the three metrics in the notebook so it does not look like I was
unaware of it.

---

## Prompt 3 — one implementation, used everywhere

```text
Implement the metric once as score(y_true, y_pred), wrapping sklearn's f1_score with
average="macro", and add a short docstring-style comment saying why macro.

Then enforce a rule for the rest of the project: this same function is used for every per-fold
cross-validation score, for choosing the best grid permutation, and for the final test-set score.
Nothing anywhere else in the notebook should call f1_score, accuracy_score or anything similar
directly.

Tell me where in the planned notebook structure that function needs to be called so I can check
later that I have not drifted.
```

**What came back:** the `score()` helper plus the list of call sites — inside the CV loop, in the
grid results table, in the Part 4 training-set report, and in the Part 5 test evaluation.

**How I verified it held:** part of the audit in [09](09-verification-and-review.md) — the one
permitted exception being `classification_report` in Part 5, which is printed alongside the
headline number for the per-class breakdown, not used to select anything.
