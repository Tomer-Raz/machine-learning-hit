# 02 — Choosing the dataset

**Stage:** before Part 1.
**Produced:** the locked decision — IMDB 50K Movie Reviews, Kaggle
`atulanandjha/imdb-50k-movie-reviews-test-your-bert`.

The binding constraint here was not size or topic: it was that the brief forbids re-splitting, so
the dataset had to **ship** a train/test split with labels on both sides.

---

## Prompt 1 — candidates against explicit criteria

```text
I need to pick a Kaggle dataset for this assignment. Hard criteria:

- Text / NLP, tagged as such on Kaggle.
- Suited to classification with Naive Bayes implemented from scratch - so a bag-of-words style
  problem where NB is a genuinely reasonable model, not a token gesture.
- CRITICAL: it must ship its own train/test split, with labels on BOTH sides. The brief forbids
  re-splitting or merge-then-resplit, and Part 5 has to score the test set directly, so a test set
  with hidden labels (a Kaggle-competition style submission file) is useless to me.
- Large enough that 5-fold cross-validation over a grid is meaningful, but small enough to run on a
  laptop in minutes, not hours.
- Publicly downloadable without a competition sign-up.

Give me three or four candidates. For each: the exact Kaggle slug, size, class structure and
balance, what files it ships and their columns, and - most importantly - whether the split is real
and labelled on both sides. Flag anything you are not certain about rather than assuming.
```

**What came back:** a shortlist, of which two were serious: TripAdvisor Hotel Reviews (~20k rows,
1–5 star ratings) and the IMDB 50K movie reviews.

---

## Prompt 2 — the decision, with the trade-off made explicit

```text
Compare the TripAdvisor hotel-review dataset and the IMDB 50K dataset for this specific
assignment. I care about, in order:

1. Whether the shipped split satisfies the "do not re-split" constraint without me having to
   invent anything.
2. Whether the label structure gives me a clean learning-type story for the notebook (binary vs
   multi-class, and whether there is a single "central" class - this decides my quality metric).
3. Whether Naive Bayes is a defensible model choice I can explain on camera.

TripAdvisor is a single CSV with a 1-5 star rating, so as far as I can tell I would have to both
binarise the labels AND create my own split - which is exactly what the brief prohibits. Tell me
if I have that wrong. Then recommend one, and say what the cost of the recommendation is.
```

**What came back:** confirmation that TripAdvisor would require inventing a split, and a
recommendation of IMDB — Stanford's *Large Movie Review Dataset*, which ships the official 25k/25k
split, both sides labelled and balanced 50/50, with polarity already binarised at source.

**Cost accepted:** IMDB is a well-worn benchmark, so the result is unsurprising. Since the
assignment grades process rather than novelty, an unsurprising, well-understood dataset was the
right trade.

---

## Prompt 3 — verifying the actual files before committing to it

```text
Before I lock this in, verify the dataset as it actually exists on Kaggle right now, not as it is
described in papers:

- For slug atulanandjha/imdb-50k-movie-reviews-test-your-bert, what files does the current version
  ship, and what are the exact column names and value formats?
- Does an anonymous kagglehub download work for it, or do I need Kaggle API credentials in the
  notebook? I would rather the notebook run for a grader without them.
- Confirm row counts per split and the class balance in each.
- Are there any data-quality issues I should handle - duplicates, empty reviews, encoding
  problems, or overlap between the train and test text?

Report what you actually observe from the downloaded files, and quote the numbers.
```

**What came back:** version 2 ships `train.csv` and `test.csv`, 25,000 rows each, columns `text`
and `sentiment` (`pos`/`neg`), balanced 50/50; anonymous `kagglehub` download works without a
token; and two data-quality findings — 96 exact-duplicate review texts inside the training set,
and about 123 review texts appearing in **both** splits.

**What I decided:** drop the 96 within-train duplicates (cleaning the training set is allowed) and
leave the 123 cross-split overlaps completely alone, since removing them would mean modifying the
test set. The overlap is disclosed in the notebook's EDA as a caveat on the test score rather than
quietly fixed — roughly 0.5 % of the test set, too small to change the conclusions but dishonest to
hide.

**Note for anyone re-reading the older docs:** an earlier version of this dataset shipped a single
`imdb_master.csv` with a `type` column and ~50k extra unlabelled `unsup` rows. That is not what
this project loads; `docs/ASSIGNMENT.md` §5 and `data/README.md` were corrected once the mismatch
was noticed.
