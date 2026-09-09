# AI prompts — working record

The assignment requires disclosing the AI prompts used, together with the resources consulted and
the purpose of each. This folder is that record, broken up by project phase.

**Assistant used:** Claude (Anthropic), through Claude Code in the terminal, working directly in
this repository.

## How to read these files

Each file covers one phase of the work and contains:

- **the prompt**, in a fenced block, as it was put to the assistant;
- **what came back** — a short summary of the answer and where it ended up in the repo;
- **what I checked, changed or rejected** — the part that matters, because nothing here was
  accepted without reading it.

> These prompts are **reconstructed** from the working sessions. They capture the substance of what
> was asked and, in particular, the constraints imposed on the assistant — they are condensed and
> tidied for reading, not a verbatim keystroke log. Where a prompt below is a composite of a short
> back-and-forth, it is marked as such.

## Index

| File | Phase | Main output |
|---|---|---|
| [01](01-understanding-the-brief.md) | Decoding the assignment brief | `docs/ASSIGNMENT.md` |
| [02](02-dataset-selection.md) | Choosing the dataset | IMDB 50K decision |
| [03](03-part1-loading-and-eda.md) | Part 1 — loading the given split, EDA | notebook Part 1 |
| [04](04-quality-metric.md) | Choosing and implementing the quality index | `score()` |
| [05](05-part2-feature-engineering.md) | Part 2 — text → vectors | `clean_text`, `tokenize`, `build_vectorizer` |
| [06](06-part3-naive-bayes-from-scratch.md) | Part 3 — the algorithm | `NaiveBayesTextClassifier` |
| [07](07-part6a-grid-search-and-cv.md) | Part 6a — grid search + 5-fold CV | `run_grid`, `results_df` |
| [08](08-parts4-5-final-model-and-test.md) | Parts 4 & 5 — final fit, test evaluation | final model, test macro-F1 |
| [09](09-verification-and-review.md) | Auditing my own work | leakage + rule-compliance checks |
| [10](10-repo-and-docs.md) | Repo and docs | builder script, `README.md` |

## Other resources consulted (not AI)

- Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, ch. 13 — the Naive Bayes
  derivation, Multinomial and Bernoulli models, and smoothing.
- scikit-learn User Guide — *Naive Bayes*, *Working with text data*, `TfidfVectorizer` reference
  (used to check my implementation and the vectorizer parameters, not to supply the model).
- NLTK documentation — Porter stemmer.
- Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis* — the source paper for the
  IMDB dataset and its train/test split.

## The one rule I gave the assistant throughout

Naive Bayes had to be **mine**. scikit-learn was allowed only for the vectorizers,
`StratifiedKFold`, the metric functions, and a one-off parity check against `MultinomialNB` /
`BernoulliNB` — never as the model that produces the reported results. Every prompt in these files
that touches the algorithm repeats that constraint, and [09](09-verification-and-review.md) is the
audit that it held.
