# Video script — ~5 minutes

> **Hebrew version:** [`VIDEO_SCRIPT_HE.md`](VIDEO_SCRIPT_HE.md) — the same script in spoken Hebrew,
> for recording in Hebrew.

Recording notes for the assignment video. The brief asks for ~5 minutes, viewable without
download (unlisted YouTube is fine), introducing the presenter at the start and walking through
**every part** showing code + output + explanation, assuming the viewer does not know the material.

## Before you record

**Open the notebook** (everything is already installed — no setup needed):

```bash
cd ~/Desktop/work/machine_learning/machine-learning-hit
source .venv/bin/activate
jupyter lab notebook.ipynb
```

Stop the server afterwards with `Ctrl-C` twice in that terminal.

**Do not re-run anything on camera.** The notebook already shows every output, which is exactly what
the brief asks for ("results visible without re-running"). A full run takes about 7 minutes; your
video is 5. If you want to show that it runs, run one instant cell — cell 15 (the `score()` smoke
test) or cell 9 (`df_train.head()`) — and nothing more.

> ### ⚠ Protect the committed notebook
>
> The submitted notebook has execution counts 1–25 in order, which is the evidence it ran cleanly in
> one pass. Running cells interactively renumbers them, and **saving** puts that mess in your
> submission. After any interactive poking:
>
> ```bash
> git status                       # notebook.ipynb listed as modified?
> git checkout -- notebook.ipynb   # if so, restore the good copy
> ```
>
> Check this before recording and before pushing. Nothing is lost — git has the clean version.

**Proving to yourself it still runs** (do this the day before, never during recording):

```bash
.venv/bin/python tools/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=14400 notebook.ipynb
```

About 7 minutes, no internet required (the Kaggle data is cached). Two stretches look like it has
frozen but have not: the grid search (~2.5 min) and the `min_df` sweep (~2.5 min).

**Framing and delivery**

- Zoom the browser to ~150 % (`Cmd +`); default-size code is unreadable in a compressed recording.
- Collapse the JupyterLab sidebar for a cleaner frame.
- Keep this script on a second screen or your phone — not on the screen you are recording.
- Record 30 seconds as a test and watch it back: legible code, clean audio.
- Cell indices below refer to the 55-cell notebook; the heading names are the reliable anchor if
  cells shift.
- Time is tight. Read code aloud only where it matters (the NB `fit`, the CV loop); everywhere else
  point at the output and say what it means.

---

---

## 0:00–0:30 — Who and what

**Show:** title cell `[0]`.

> "Hi, I'm Tomer R., ID ending 5130 — this is my machine-learning assignment, done solo.
> The task type is **text analysis**, the learning type is **binary classification** — is a movie
> review positive or negative — and the algorithm I implemented from scratch is **Naive Bayes**,
> in both its Multinomial and Bernoulli forms. The data is the IMDB 50K review dataset from
> Kaggle, which ships already split into 25,000 training and 25,000 test reviews, balanced 50/50.
> Everything you'll see runs top to bottom in this one notebook."

## 0:30–1:15 — Part 1: the data, and choosing the quality index

**Show:** `[7]` load output, `[9]` and `[11]` the two `.head()` tables, `[13]` EDA, then `[14]`/`[15]`.

> "The dataset comes with its own train/test split, so I load it as given — I never re-split or
> merge it. Here are the first five rows of each side: the review text, and the label mapped to
> 1 for positive, 0 for negative. I drop 96 exact-duplicate reviews *inside the training set only*;
> the test set is left completely untouched. Both classes are 50/50, and reviews average a few
> hundred words."

**The metric — say this deliberately, it is worth 10 points:**

> "For the quality index the rule is: regression uses R²; a problem with one 'central' class you
> care about — like spam detection — uses F1 on that class; and a balanced problem with no single
> class of interest uses the **macro-averaged F1**. Positive and negative reviews matter equally
> here, so macro-F1 is the right choice, and I implement it once as `score()` and use that same
> function for every fold, for model selection, and for the final test score."

## 1:15–2:15 — Part 2: turning text into numbers

**Show:** `[18]` the functions, then `[20]` the stage-by-stage table, `[21]`/`[22]` the vectors.

> "A model can't read text, so Part 2 turns each review into a vector of numbers. `clean_text`
> lower-cases and strips HTML tags, URLs, digits and punctuation. `tokenize` splits into words and
> can optionally drop stop words and apply Porter stemming, so that *amazing*, *amazed* and
> *amazingly* all collapse to one feature, `amaz`."

**Point at the demo table** — two training reviews and one test review, raw → cleaned → tokens →
without stop words → stemmed:

> "Then Bag-of-Words counts how often each term appears, and TF-IDF re-weights those counts so a
> word that shows up in every review counts for little, while a distinctive word counts for a lot.
> `ngram_range=(1,2)` also keeps two-word phrases, which is how the model can learn *'not good'*
> rather than just *not* and *good* separately. `build_vectorizer` bundles all of those knobs into
> one function — that's exactly what the grid search will sweep later."

## 2:15–3:15 — Part 3: Naive Bayes from scratch

**Show:** the derivation markdown `[25]`/`[26]`, the class `[27]`, the parity check output `[29]`.

> "Naive Bayes picks the class with the highest P(class | document). By Bayes' rule that's the
> class prior times the probability of the document given the class, and the 'naive' assumption is
> that words are independent given the class — so that product is just a product over words.
> Multiply thousands of small probabilities and you underflow to zero, so everything is done in
> **log space**: products become sums, and the prediction is an argmax of a matrix multiply."

> "`alpha` is Laplace/Lidstone smoothing — without it, one unseen word would multiply a whole class
> to probability zero. `fit_prior` chooses between learning P(class) from the data or assuming
> uniform. `model_type` switches between Multinomial, which uses term counts, and Bernoulli, which
> uses presence/absence and explicitly accounts for the words that are *absent*."

**Point at the check output:** *(note to self: this cell also prints 0.8892 on the held-out training
slice. If you mention it at all, say "that is a sanity check on held-out training data with the
untuned settings, not my result" and move on — the real number comes in Part 5.)*

> "To prove the implementation is right, not just plausible, I compare it against scikit-learn's
> `MultinomialNB` and `BernoulliNB` across several `alpha` values — the log-probabilities agree to
> about 1e-13, and every prediction is identical. Note that I run that comparison on a held-out fifth
> of the *training* set, not on the test set — the test set is reserved for Part 5, and a correctness
> check does not justify spending it early. scikit-learn is used only as a reference here; the model
> I actually use is mine."

## 3:15–4:15 — Part 6a: grid search with 5-fold cross-validation

**Show:** `[31]` the CV loop, `[32]` the results table, `[34]` best row + bar chart, then `[36]`
and `[38]`.

> "To choose a configuration I use 5-fold cross-validation **inside the training set only** — the
> test set is not touched anywhere in this section. The training set is split into five stratified
> parts; five times, four parts train and one part validates, and I average the five scores."

**This is the sentence that shows you understood the point:**

> "The critical detail is *where* the vectorizer is fitted. It is rebuilt from scratch inside every
> fold, on that fold's training portion only, and merely applied to the held-out part. If I had
> vectorized the whole training set once up front, the vocabulary and the IDF weights would carry
> information from the validation rows into training — that's leakage, and it would give me an
> optimistically wrong score."

> "The grid is the Cartesian product of Bag-of-Words vs TF-IDF, unigrams vs uni+bigrams, stemming
> on or off, Multinomial vs Bernoulli, and three `alpha` values — 36 permutations, each with its
> mean and standard deviation across folds. The winner is TF-IDF with uni+bigrams, stemming on,
> Multinomial, `alpha=0.1`, at about **0.878 macro-F1**."

**Two follow-up experiments — both worth mentioning because the *negative* results are the
interesting ones:**

> "Removing stop words actually **hurts**, dropping roughly a point. That makes sense for
> sentiment: the standard stop-word list throws away *not*, *no* and *very*, which are exactly the
> words that flip a review's meaning."

> "The `min_df` sweep comes out flat, and that's explainable rather than surprising: `max_features`
> already caps the vocabulary at the 30,000 most frequent terms, so anything that survives that cut
> is common enough to clear any `min_df` I tried. The spread is in the fourth decimal, which is
> noise, so I keep `min_df=5` — the smaller, simpler vocabulary — instead of chasing the argmax."

## 4:15–5:00 — Parts 4 and 5: final training and the test set

**Show:** `[42]` the refit, `[44]` the trace, then `[47]`, `[49]`, `[51]`, `[52]`.

> "Part 4 takes that winning configuration and re-fits it — vectorizer and classifier — on the
> **entire** training set, since cross-validation only ever trained on four fifths at a time. Here
> are the same example reviews pushed through the final pipeline, with the terms that pushed each
> prediction hardest toward positive or negative."

> "Part 5 is the only place the test set is used. The vectorizer only *transforms* it — no
> re-fitting — and here are the first five predictions next to the true labels with the model's
> probability. On the full 25,000 test reviews the macro-F1 is **0.865**, against a cross-validated
> estimate of 0.878 and a training-set score of 0.907. That ordering — training highest,
> cross-validation in the middle, test lowest — is what an honest evaluation looks like:
> cross-validation is slightly optimistic because I *picked* the configuration by its
> cross-validated score, and the Stanford split deliberately puts different films in the test set,
> so movie-specific words like actor names don't carry over. A test score that came out *higher*
> would have been the warning sign."

**Close on the confusion matrix and top-terms chart `[52]`:**

> "Errors are near-symmetric between the classes, and the most predictive terms are exactly what a
> human would pick — *worst*, *awful*, *waste* on one side, *excellent*, *wonderful*, *perfect* on
> the other. I skipped the imbalanced-data extension, since this dataset is already 50/50, and the
> SHAP explainability extension. Thanks for watching."

---

## Checklist before uploading

- [ ] Presenter introduced by name in the first 30 seconds.
- [ ] Every part (1, quality index, 2, 3, 4, 5, 6a) appears on screen with its output visible.
- [ ] Length ≈ 5 minutes.
- [ ] Uploaded unlisted, playable in-browser without downloading.
- [ ] URL pasted into `README.md`, `docs/PROGRESS.md`, and the notebook's Excel appendix cell,
      then the notebook rebuilt (`tools/build_notebook.py`) and re-executed.
