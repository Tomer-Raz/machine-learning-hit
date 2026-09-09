# 06 — Part 3: Naive Bayes implemented from scratch

**Stage:** notebook Part 3 (35 points — the other joint-largest part, and the core of the
assignment).
**Produced:** the derivation markdown, `NaiveBayesTextClassifier`, and the parity check against
scikit-learn.

This is the part where the assistant's role had to be tightly bounded: the algorithm has to be
mine, and I have to be able to explain every line of it on camera.

---

## Prompt 1 — the derivation, before any implementation

```text
Derive Naive Bayes for text classification, as markdown for my notebook. Do the derivation FIRST,
in its own cell - I do not want code yet, because I need to be able to explain the maths before I
can defend the implementation.

Work through, in order:

1. The classification goal: argmax over classes of P(class | document), and Bayes' rule turning it
   into prior x likelihood.
2. The "naive" conditional-independence assumption, stated precisely, plus an honest sentence on
   why it is false for natural language and why the classifier works anyway.
3. The Multinomial model: the document as a bag of term counts, and the resulting per-class term
   probability with Laplace/Lidstone smoothing - give the full formula with the alpha and the
   alpha*V in the denominator, and explain what each symbol is.
4. Why the smoothing is not optional: one unseen term drives an entire class posterior to zero.
5. Why everything must be done in log space: thousands of small probabilities underflow to zero in
   float arithmetic. Show how the product becomes a sum and how the prediction becomes an argmax
   of a matrix multiply.
6. The Bernoulli variant: presence/absence instead of counts, and crucially that it scores ABSENT
   terms too - give that formula and contrast it with the Multinomial one.

Use LaTeX for the formulas. Cite the standard reference for the derivation so it is clear this is
textbook material and not something I invented.
```

**What came back:** the Part 3a derivation cell, following Manning, Raghavan & Schütze ch. 13. The
absent-terms contrast for Bernoulli is the part I would have got wrong if I had written it from
memory, and it is the thing that makes the two variants genuinely different rather than a cosmetic
switch.

---

## Prompt 2 — the implementation, under an explicit rule

```text
Now implement it as class NaiveBayesTextClassifier. The hard rule: this is MY implementation.
scikit-learn's naive_bayes module may be imported ONLY for a parity check I will write separately,
and must never be the thing producing my reported results. Do not import it in this cell.

Interface, deliberately scikit-learn-shaped so it drops into my own cross-validation loop later:

  __init__(self, alpha=1.0, fit_prior=True, model_type="multinomial")
  fit(X, y) -> self
  predict(X)
  predict_log_proba(X)
  predict_proba(X)

Requirements:
- The three constructor arguments are my exposed hyper-parameters: alpha (smoothing strength),
  fit_prior (learn P(class) from the data vs assume uniform), and model_type ("multinomial" or
  "bernoulli"). The brief requires hyper-parameters to be exposed, and these are the ones taught
  in class.
- fit must accept SciPy sparse input, since that is what the vectorizers produce for a 25k x 30k
  matrix. Do not densify it - that matrix would be enormous.
- Work entirely in log space. predict_proba normalises via log-sum-exp; write the log-sum-exp
  helper yourself rather than importing scipy's, and make it numerically stable by subtracting the
  row max.
- Vectorise with NumPy - no Python loops over documents or over vocabulary terms.
- Expose the fitted state under the conventional names: classes_, class_count_, feature_count_,
  class_log_prior_, feature_log_prob_. I want feature_log_prob_ available afterwards because I
  intend to use the log-likelihood ratio between classes for explaining individual predictions and
  for a top-terms chart.
- For Bernoulli, binarise the input inside fit and predict, and include the absent-term
  contribution - implement it with the (log p - log(1-p)) trick plus the row sum of log(1-p), and
  add a comment explaining why that is algebraically equivalent to scoring every absent term.
- Comments explain the maths, not the syntax. Use # comments, not docstrings - the notebook builder
  stores cell sources as r-strings and triple quotes inside them would break the build.

Raise a clear error for an unknown model_type rather than silently defaulting.
```

**What came back:** the class as it stands in the notebook. The Bernoulli absent-term trick and the
stable log-sum-exp are the two pieces I made sure I could re-derive myself before accepting them —
they are also the two most likely things for a grader to ask about.

---

## Prompt 3 — proving it is correct, not merely plausible

```text
Write a correctness check for my implementation. "It runs and the accuracy looks reasonable" is not
evidence - a subtly wrong smoothing term would still produce a plausible-looking score.

Compare my class against scikit-learn's MultinomialNB and BernoulliNB as a reference oracle:

- On synthetic count data with a fixed seed, across several alpha values AND both fit_prior
  settings, assert that the predictions are identical and report the maximum absolute difference in
  predicted log-probabilities.
- Then on the REAL IMDB feature matrix from Part 2, assert my predictions match sklearn's on every
  test review, and print my from-scratch model's test macro-F1 alongside.
- Use assertions so the notebook FAILS LOUDLY if this ever stops holding - I do not want a check
  that silently prints a bad number.
- Print the maximum log-probability difference explicitly, so the committed output is evidence a
  grader can read without re-running anything.

Add a comment making it unambiguous that scikit-learn appears here as a reference only, and is
never the model that produces my results.
```

**What came back:** the Part 3c check. It reports a maximum log-probability difference of about
1e-13 — floating-point noise — and identical predictions on all 25,000 test reviews, for both the
Multinomial and Bernoulli variants.

**Why I wanted it this way:** it converts "I implemented Naive Bayes" from a claim into something
demonstrated in committed output, and the assertions mean the claim cannot rot silently if I edit
the class later.
