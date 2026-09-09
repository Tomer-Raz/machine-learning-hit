# 05 — Part 2: turning text into numbers

**Stage:** notebook Part 2 (35 points — the joint-largest part).
**Produced:** `clean_text`, `tokenize`, `build_vectorizer`, the stage-by-stage demo on real
reviews, and the baseline feature matrix.

---

## Prompt 1 — the explanation before the code

```text
Write the markdown that opens Part 2, explaining text feature engineering to someone who does not
know any of it - assume the viewer of my video has never heard of TF-IDF.

Cover, in an order that builds on itself: why a model cannot consume raw text at all; tokenisation;
lower-casing and cleaning; stop words; stemming vs lemmatisation; Bag-of-Words; TF-IDF and what
the IDF factor actually does; n-grams and why bigrams matter specifically for sentiment; and
vocabulary pruning via min_df, max_df and max_features.

Constraints:
- Explain the intuition first and the formula second, never the reverse.
- For TF-IDF, give the actual formula, but immediately follow it with a one-sentence plain reading
  of what it does to a common word versus a rare one.
- For bigrams, use a negation example - that is the reason I am going to enable them.
- No bullet-point soup. This is prose I will paraphrase out loud.
```

**What came back:** the Part 2a markdown. The negation example ("not good" surviving as a single
feature only if bigrams are on) became one of the project's main talking points, because it is
the clearest justification for the `ngram_range=(1,2)` that later wins the grid.

---

## Prompt 2 — the pipeline functions, with one code path

```text
Implement the feature-engineering layer as three pieces:

1. clean_text(s): raw review -> normalised string. Lower-case; strip HTML tags (this dataset has
   <br /> all over it); strip URLs; drop digits and punctuation, keeping letters and whitespace.
2. tokenize(s, use_stemming=False, remove_stopwords=False): normalised string -> list of tokens.
   Tokens are runs of two or more letters. Stop-word removal happens BEFORE stemming if both are
   on. Use the Porter stemmer, and cache it - it is called millions of times across a grid search.
3. build_vectorizer(kind, ngram_range, use_stemming, remove_stopwords, min_df, max_df,
   max_features, binary): returns a configured but UNFITTED CountVectorizer or TfidfVectorizer.

The critical design requirement: wire clean_text and tokenize into the vectorizer as its
preprocessor and tokenizer, so that the step-by-step demo I show the viewer and the full pipeline
that produces my results are literally the same code path. I do not want a hand-rolled demo that
merely resembles what the real pipeline does - if they can drift apart, they will, and the demo
becomes a lie.

build_vectorizer is the single knob-box the Part 6a grid search will sweep, so every axis I intend
to experiment with must be a parameter of it.
```

**What came back:** the three functions as they stand, with `lru_cache` on the stemmer and the
`preprocessor=` / `tokenizer=` wiring that keeps the demo and the pipeline unified.

**Why that mattered:** the brief requires demonstrating feature engineering on concrete examples.
Sharing one code path means the demonstration is evidence about the real pipeline rather than an
illustration of it.

---

## Prompt 3 — the demonstration on real rows

```text
Write the Part 2b demo. The brief wants the feature engineering shown step by step on 2-3 training
examples and 2-3 test examples, raw text through to the final vector.

Build it on real rows: one positive and one negative review from the training set, and one from
the test set, taken by index so they are reproducible - no random sampling.

Show, as a readable table with the stages as rows so long text wraps sensibly:
raw (truncated) -> cleaned -> token count -> first N tokens -> tokens after stop-word removal ->
tokens after stemming.

Then, in a second cell, take those same three reviews from tokens to numbers:
(a) raw Bag-of-Words counts over just these three documents, printing the highest-count terms per
    document - and note honestly in the output that the top counts are inevitably function words,
    because that is exactly what motivates TF-IDF;
(b) the same three reviews as TF-IDF over unigrams and bigrams, showing the non-zero entries with
    their weights, so the viewer can see a distinctive term outranking a common one.

The point is that a viewer should be able to follow one specific review from English sentence to
numeric vector without a gap.
```

**What came back:** the two demo cells. Part (a) deliberately shows the weakness of raw counts, and
part (b) shows TF-IDF fixing it — which makes the vectorizer comparison in the grid search
meaningful instead of arbitrary.

**What I changed:** I asked for fixed row indices rather than sampled rows, so the committed
outputs are stable across re-runs, so the surrounding markdown can name exactly what is shown.
