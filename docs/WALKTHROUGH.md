# Walkthrough — what this assignment is, what we built, and how to present it

This is the plain-English guide to the whole project. It assumes you know nothing about machine
learning, and explains every idea the first time it appears. Read it top to bottom before you
record the video.

**Contents**

1. [What the assignment asks for](#1-what-the-assignment-asks-for)
2. [The idea behind the project, in plain words](#2-the-idea-behind-the-project-in-plain-words)
3. [The words you need to know](#3-the-words-you-need-to-know)
4. [What is in the repository](#4-what-is-in-the-repository)
5. [How to run it](#5-how-to-run-it)
6. [The notebook, part by part](#6-the-notebook-part-by-part)
7. [The results, and what they mean](#7-the-results-and-what-they-mean)
8. [What to present in the video](#8-what-to-present-in-the-video)
9. [Questions you might be asked, and the answers](#9-questions-you-might-be-asked-and-the-answers)
10. [Submission checklist](#10-submission-checklist)

---

## 1. What the assignment asks for

The course brief (the original is `ml.md`, machine-translated from Hebrew and garbled in places;
our clean reading of it is `docs/ASSIGNMENT.md`) asks for one thing, done properly:

> Take a dataset from Kaggle, run a complete supervised-learning process on it, and **write the
> learning algorithm yourself** instead of calling a ready-made one.

### The five things you must hand in

1. **A code notebook** where the results are visible **without re-running it**. That last part
   matters: a grader opens it and sees the outputs already there.
2. **A video of about 5 minutes**, playable in a browser, where you introduce yourself and walk
   through every part showing the code, its output, and an explanation.
3. **The Kaggle link** to the dataset.
4. **The repository link**, viewable without downloading anything.
5. **A row in the shared Excel sheet** with: assignment type, learning type, algorithm, dataset
   name, dataset URL, video URL, repository URL.

### How the points are split

| Part | What it must contain | Points |
|---|---|---|
| 1 — Introduction | Student-details cell, AI-prompts cell, a paragraph describing the problem and data, load train and test, show the first rows of **each** | 5 |
| Quality index | Choose the right measurement, justify it, implement it, use it everywhere | 10 |
| 2 — Feature engineering | Turn text into numbers using the techniques taught in class, and demonstrate it on 2–3 real examples from train **and** test | 35 |
| 3 — The algorithm from scratch | Your own implementation with adjustable settings, a train function and a predict function, plus an explanation | 35 |
| 4 — Training | Retrain the winning setup on the whole training set, and show the feature steps again | 5 |
| 5 — Prediction and evaluation | Feature steps on test examples, the **first 5 predictions**, and the score on the test set | 10 |
| 6a — Extension: grid search + k-fold | Try many combinations with cross-validation, show a results table of all of them, and the best one separately | 25 |
| 6b — Imbalanced data | *Skipped* — our data is perfectly balanced, so there is nothing to fix | 10 |
| 6c — Explainability | *Skipped* | 10 |

We did Parts 1–5 in full plus 6a. That is about 125 points of available credit attempted.

**The most important line in the whole brief:** grading is mostly on **process and understanding**,
not on how high your score is. A well-explained 86 % beats a mysterious 91 %.

### The rules that are easy to break by accident

These are the ones where your code still runs, the output still looks fine, and you still lose
points:

- Use the dataset's **own** train/test split. Do not create your own. Do not merge the two and
  split again.
- Show the first rows of **both** the training set and the test set.
- Cross-validation happens **inside the training set only**.
- The test set is used **once**, at the very end.
- Show the feature engineering **step by step on real examples** — not just described in words.
- **No leakage** (explained in section 3) — the vectorizer must be built inside each fold.
- The algorithm must be **yours**, with adjustable settings exposed.
- Two cells are mandatory: **student details** and **AI prompts used**.

---

## 2. The idea behind the project, in plain words

We are teaching a computer to read a movie review and say whether the person liked the film or not.

That is it. Here is the whole project in seven steps:

1. **Get the data.** 50,000 movie reviews from IMDB. Half are positive, half are negative. Somebody
   already labelled every one of them, and already divided them into 25,000 for teaching the
   computer ("training") and 25,000 for testing it.

2. **Turn words into numbers.** A computer cannot do arithmetic on the sentence *"this film was
   wonderful"*. So we convert every review into a long list of numbers, where each number says
   something about which words appear in it.

3. **Write the learning algorithm ourselves.** We used **Naive Bayes**. In one sentence: it counts
   how often each word appears in positive reviews versus negative reviews, and then, for a new
   review, it asks *"which class makes these particular words more likely?"*

4. **Try many versions and pick the best one.** There are lots of choices to make when turning
   words into numbers. We tried 36 combinations and measured each one fairly.

5. **Retrain the winner on all the training data.**

6. **Test it once**, on the 25,000 reviews it has never seen.

7. **Report honestly**, including the things that did not work.

Our final answer: the computer gets it right about **86.5 %** of the time on reviews it has never
seen before.

---

## 3. The words you need to know

Learn these eight and everything else in the notebook makes sense.

**Supervised learning.** You give the computer examples *with* the right answers attached, and it
learns the pattern. Like studying with an answer key.

**Training set and test set.** The training set is what the computer learns from. The test set is
the exam — reviews it has never seen. You must keep them separate, or you are letting it cheat by
memorising the exam paper.

**Features.** The numbers that describe one review. If our vocabulary is 30,000 words, each review
becomes a list of 30,000 numbers, mostly zeros, with a non-zero value wherever one of those words
appears in that review.

**Bag of Words.** The simplest way to make those numbers: just count how many times each word
appears. It throws away word order — hence "bag" — which sounds like it should ruin everything, and
mostly does not.

**TF-IDF.** A smarter version of counting. It reduces the weight of words that appear in almost
every review (*the*, *movie*, *film* — they tell you nothing) and increases the weight of words
that are distinctive. TF = how often the word appears here; IDF = how rare the word is overall.

**Naive Bayes.** Our algorithm. It uses Bayes' rule from probability: to decide if a review is
positive, compare "how likely are these words if it is positive?" against "how likely are these
words if it is negative?", and pick the winner. It is called *naive* because it pretends every word
is independent of every other word — which is obviously false in real language, but the method
works well anyway.

**Cross-validation.** How you compare options fairly without touching the test set. Split the
training data into 5 equal parts. Train on 4, check on the 1 left out. Do that 5 times, so every
part gets a turn as the checker. Average the 5 scores. It gives a much more reliable comparison
than a single split.

**Leakage.** The mistake that ruins machine-learning projects. It means information from your
checking data sneaks into your training data, so your score comes out better than the truth. In our
project the danger is specific: if we built the vocabulary using *all* the training reviews before
splitting into 5 parts, then the vocabulary would already "know" about the reviews we are about to
check on. So instead, we rebuild the vocabulary from scratch inside every single fold, using only
that fold's training portion. This is the single most important technical decision in the project.

---

## 4. What is in the repository

```
machine-learning-hit/
├── notebook.ipynb          ← THE deliverable. 55 cells, already run, all outputs saved.
├── README.md               overview and the submission links
├── ml.md                   the original assignment brief
├── requirements.txt        the Python packages needed
├── tools/
│   └── build_notebook.py   the script that GENERATES notebook.ipynb (see the warning below)
├── data/
│   └── README.md           explains the data (the data files are not committed)
└── docs/
    ├── ASSIGNMENT.md       our clean reading of the brief, plus the point map
    ├── PROGRESS.md         the project diary — what was done, when, and why
    ├── VIDEO_SCRIPT.md     the timed script for your 5-minute video
    ├── WALKTHROUGH.md      this file
    └── prompts/            the AI prompts used, by phase (required disclosure)
```

> ### ⚠ Important: never edit `notebook.ipynb` by hand
>
> The notebook is **generated** by `tools/build_notebook.py`. Every cell in the notebook is a line
> in that script. If you edit the notebook directly, your change is destroyed the next time the
> script runs.
>
> To change anything in the notebook:
> 1. edit `tools/build_notebook.py`,
> 2. run the build command,
> 3. run the execute command (both in section 5 below).
>
> This does **not** mean code is hidden away in a module. The generated notebook still contains
> every line of code inline and visible, exactly as the brief requires. It is only a convenience
> for editing.

---

## 5. How to run it

### Option A — you do not need to run anything

This is the normal case, and it is what a grader will do. The notebook already has all its outputs
saved inside it. Open `notebook.ipynb` on GitHub, or in any editor, and read it top to bottom. Every
number, table and chart is already there.

### Option B — run it on your own machine

From inside the `machine-learning-hit` folder:

```bash
# 1. create an isolated Python environment (once)
python -m venv .venv

# 2. activate it (do this every time you open a new terminal)
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

# 3. install the packages (once)
pip install -r requirements.txt

# 4. open the notebook
jupyter lab notebook.ipynb
```

Then press **Run All**. It takes about **7 minutes** end to end. The data downloads automatically
from Kaggle the first time (no Kaggle account or password needed) and is cached afterwards.

### Option C — Google Colab

Upload `notebook.ipynb` to [Google Colab](https://colab.research.google.com/) and run it there.
Everything it needs is installed by the notebook itself or already present in Colab.

### Rebuilding the notebook after editing the builder

If you change `tools/build_notebook.py`, you must regenerate **and** re-run:

```bash
.venv/bin/python tools/build_notebook.py

.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=14400 notebook.ipynb
```

The first command rewrites the notebook (it will say `55 cells`). The second runs every cell and
saves the outputs back into the file. **Both** are needed — if you only run the first, you get a
notebook with no outputs, which fails the "results visible without re-running" requirement.

---

## 6. The notebook, part by part

Open `notebook.ipynb` next to this guide. Cell numbers are counted from 0.

### Cell 0 — Title and student details *(required)*

Your name in the form the brief asks for (`Tomer R.`) and the **last four digits** of your ID
(`5130`). Your full ID number appears nowhere in this repository, because the repository is public.

Underneath, the summary of the project: assignment type, learning type, algorithm, metric, dataset.

### Cell 2 — AI prompts *(required)*

The brief requires you to disclose the AI help you used. This cell lists what the assistant was used
for, with example prompts, plus the non-AI sources (the textbook chapter, the scikit-learn user
guide, the NLTK docs, the original IMDB paper). The longer record is in `docs/prompts/`.

### Cell 3 — The problem and the data, in one paragraph

Plain description of what we are predicting and what the data is.

### Cell 5 — Setup

Imports, and `RANDOM_STATE = 42` — a fixed "random seed" so that anything random (like how the data
is shuffled into 5 folds) happens the same way every time. That is what makes the results
reproducible.

There is also a small block using `truststore`, which fixes a certificate error when downloading
from behind a corporate network. It does nothing on a normal connection.

### Cells 7–11 — Loading the data *(Part 1)*

`load_imdb()` downloads the dataset and loads the two files the dataset **already provides**:
`train.csv` and `test.csv`. We do not split anything ourselves — that is the rule.

It renames the columns to `review` and `label`, and maps `pos → 1`, `neg → 0`.

It also removes 96 reviews from the **training** set that are exact duplicates of other training
reviews. Cleaning the training data is allowed; the test set is returned exactly as it came.

**Output to look at:** `train: (24904, 2)   test: (25000, 2)`. That is 24,904 training reviews after
removing the 96 duplicates, and all 25,000 test reviews untouched.

Cells 9 and 11 show `.head()` — the first five rows — of the training set and the test set. The
brief specifically asks for **both**.

### Cell 13 — A first look at the data *(Part 1)*

| What it shows | Result |
|---|---|
| Class balance, training | 12,432 negative / 12,472 positive |
| Class balance, test | 12,500 / 12,500 |
| Missing or empty reviews | none |
| Reviews appearing in **both** splits | 123 (0.49 % of the test set) |
| Review length in words | shortest 10, median 174, average 234, longest 2,470 |

Two things worth saying out loud:

- The classes are **balanced** (roughly 50/50). This directly decides which quality measurement we
  use in the next cell.
- The 123 overlapping reviews are **disclosed, not removed**. Removing them would mean modifying the
  test set, which the rules forbid. It is half a percent, too small to change the conclusions, and
  it is better to state it than to hide it.

### Cells 14–15 — The quality index *(worth 10 points)*

"Quality index" just means: **which single number do we use to say how good the model is?**

The rule from the brief:

| Type of problem | Measurement to use |
|---|---|
| Predicting a number (regression) | R² |
| Several classes, **or** two classes with neither being the one you care about | **F1 macro-average** ← us |
| Two classes where one is the one you care about (e.g. spam) | F1 of that one class |

We are in the middle row. Positive and negative reviews matter equally — there is no "the important
one" — so we use the **macro-average F1**.

*What F1 is:* a combination of two things. **Precision** = of the reviews I called positive, how
many really were? **Recall** = of the reviews that really were positive, how many did I catch? F1
combines them into one number. **Macro-average** means: compute F1 for the positive class, compute
it for the negative class, and take the plain average — so both classes count equally.

We implement it **once**, as `score(y_true, y_pred)`, and use that same function for every
measurement in the entire notebook. Consistency is part of what is being graded here.

### Cells 16–24 — Feature engineering *(Part 2 — 35 points)*

This is where text becomes numbers. Three functions do the work:

- **`clean_text`** — lowercases the review, removes the HTML tags (this dataset is full of `<br />`),
  removes web addresses, and strips out digits and punctuation.
- **`tokenize`** — splits the cleaned text into individual words, and can optionally remove *stop
  words* (extremely common words like *the*, *is*, *and*) and apply *stemming*. Stemming chops words
  down to a common root, so *amazing*, *amazed* and *amazingly* all become `amaz` and count as the
  same feature.
- **`build_vectorizer`** — the settings box. It produces the object that turns a list of reviews into
  the big table of numbers, with every choice adjustable: Bag-of-Words or TF-IDF, whether to include
  two-word phrases, stemming on or off, stop words on or off, and how aggressively to shrink the
  vocabulary.

**One design point worth mentioning in the video:** `clean_text` and `tokenize` are plugged directly
into the vectorizer. That means the demonstration you show the viewer and the pipeline that produced
the real results are *literally the same code*. A demo that merely resembles the real pipeline can
drift away from it and become a lie.

**Cells 20–22 are the demonstration the brief requires.** Three real reviews — one positive from
train, one negative from train, one from test — shown at every stage:

```
raw text → cleaned → word list → without stop words → stemmed → numbers
```

Cell 21 shows plain word counts, and deliberately points out that the highest counts are boring
words like *the* — which is exactly why TF-IDF exists. Cell 22 then shows the same three reviews as
TF-IDF, where distinctive words outrank common ones.

**Cell 24** builds the baseline feature table for the whole dataset: 24,904 reviews × 50,000
features. Only **0.47 %** of that table is non-zero — because any single review contains only a few
hundred distinct words out of 50,000 possible. That is why it is stored as a "sparse" table, which
records only the non-zeros.

### Cells 25–29 — Naive Bayes from scratch *(Part 3 — 35 points)*

The heart of the assignment.

**Cells 25–26** explain the mathematics: Bayes' rule, the "naive" independence assumption, the
smoothing parameter, and why everything is computed with logarithms. That last point is worth
understanding: multiplying thousands of small probabilities together gives a number so tiny the
computer rounds it to zero. Taking logarithms turns those multiplications into additions, which
computers handle fine.

**Cell 27** is the implementation: `class NaiveBayesTextClassifier`. It has

- three adjustable settings, as the brief requires: `alpha` (smoothing strength), `fit_prior`
  (whether to learn how common each class is, or assume 50/50), and `model_type` (`"multinomial"`
  uses word counts, `"bernoulli"` uses only whether a word is present or absent);
- `fit(X, y)` — the training function;
- `predict(X)` — the prediction function;
- plus `predict_proba` for the probability behind each prediction.

*Why smoothing (`alpha`) is not optional:* suppose the word "gorgeous" never appears in any negative
training review. Its probability given "negative" would be exactly zero, and since we multiply
probabilities together, a single such word would drive the whole answer to zero regardless of every
other word in the review. Adding a small amount (`alpha`) to every count prevents that.

**Cell 29 proves the implementation is correct.** It compares our classifier against scikit-learn's
ready-made one, across several settings, and checks they agree. The result: the probabilities match
to **1.42e-13** (that is 0.000000000000142 — floating-point rounding noise), and the predictions are
identical on all 25,000 reviews.

This is the difference between *claiming* you implemented Naive Bayes and *showing* it. scikit-learn
appears here as a ruler to measure against — never as the model that produces our results.

> **⚠ One flaw to fix before you submit.** The last two lines of this cell run the comparison on the
> **test** set, and print a test score (0.8699) for the untuned baseline. Nothing is decided from
> that number — the whole grid search that follows uses only the training data — but it contradicts
> the sentence in Part 5 that says the test set is used only once, and it invites the awkward
> question *"why is your baseline's test score higher than your final model's?"* (The answer is that
> they differ by half a point, which is noise, and that the baseline number was never used to choose
> anything.) The clean fix is to run this comparison on a held-out slice of the **training** set
> instead, which proves exactly the same thing without reading the test labels early. See section 10.

### Cells 30–40 — Grid search with cross-validation *(Part 6a — 25 points)*

Now we choose the best combination of settings, using **only the training data**.

**Cell 31** builds the cross-validation machinery. Read the loop carefully, because this is where
the leakage rule lives: inside every one of the 5 folds, a **brand-new** vectorizer is built and
fitted on that fold's training portion only, then merely applied to the held-out portion.

**Cell 32** runs the grid: every combination of

| Choice | Options tried |
|---|---|
| Bag-of-Words or TF-IDF | 2 |
| Single words, or single words + two-word phrases | 2 |
| Stemming off or on | 2 |
| Multinomial or Bernoulli Naive Bayes | 2 (Bernoulli only with Bag-of-Words) |
| `alpha` smoothing | 3 values: 0.1, 0.5, 1.0 |

= **36 combinations**, each measured 5 times. It finishes in about 150 seconds. The output is a
table with one row per combination, sorted best-first, showing the average score, how much it varied
across the 5 folds, and the 5 individual fold scores.

**Cell 34** prints the winner on its own, as the brief requires, with a bar chart of the top 12.

**Cells 35–36 — experiment: do stop words help?** We predicted in writing, *before* showing the
result, that removing stop words would **hurt** here. It did: 0.8684 with removal versus 0.8779
without. The reason is worth saying out loud in the video — the standard stop-word list contains
*not*, *no* and *very*, and those are precisely the words that flip the meaning of a review.

**Cells 37–38 — experiment: how much rare-word pruning?** We tried five settings of `min_df`
(the minimum number of reviews a word must appear in to be kept). The results are almost identical:
0.8779, 0.8778, 0.8779, 0.8776, 0.8756. The explanation is in the notebook: another setting,
`max_features=30000`, already keeps only the 30,000 most common words, so almost nothing that
`min_df` would remove survives that cut anyway. The two settings overlap, and `max_features` is the
one doing the work.

**Cell 40** assembles the winner:

```
TF-IDF · single words + two-word phrases · stemming on · Multinomial · alpha = 0.1 · min_df = 5
```

Note one deliberate decision here: the `min_df` results differ only in the fifth decimal place,
which is noise rather than a real difference. So instead of blindly taking the highest number
(`min_df=1`), we keep `min_df=5` — the same score with a smaller, simpler model. *"When two models
tie, prefer the simpler one"* is a good answer to give on camera; *"the fourth decimal was higher"*
is not.

### Cells 41–44 — Training the winner *(Part 4)*

Cross-validation only ever trained on 4/5 of the data at a time. Now we rebuild the winning setup
and train it on **all 24,904** training reviews.

Output: vocabulary of 30,000 features, and a score of **0.9068 on the training data itself**. That
last number is labelled "resubstitution" and is *not* a measure of how good the model is — it is the
model being tested on the same reviews it just memorised. It is shown for comparison only.

**Cell 44** repeats the feature-engineering demonstration using the *chosen* configuration, and adds
something useful: for each example review, the individual words that pushed the decision hardest,
with a sign showing which way. This is where you can see the model reasoning.

### Cells 45–52 — The test set *(Part 5)*

The exam. The vectorizer only **transforms** the test data — it is never re-fitted on it.

- **Cell 47** — three test reviews traced end to end. One of them is a genuine near-miss: a sarcastic
  review that opens with the word *"Great"*, which the model calls positive with probability 0.513
  when the true answer is negative. Show this one in the video — an honest near-miss is far more
  interesting than three easy successes.
- **Cell 49** — the **first 5 predictions**, exactly as the brief asks: the review text, the true
  label, the predicted label, and the probability. Four negatives and one positive, all five correct.
- **Cell 51** — the headline result: **test macro-F1 = 0.8648**, plus a full breakdown per class.
- **Cell 52** — two charts: the confusion matrix (how many of each class were right and wrong), and
  the most predictive words for each class.

### Cells 53–54 — Conclusions and the Excel values

The conclusions cell states what worked, explains the gap between the cross-validated estimate and
the test score honestly (see the next section), and lists what we would try next. The final cell
holds the values you need for the shared Excel sheet.

---

## 7. The results, and what they mean

| Measurement | Score | What it actually means |
|---|---|---|
| Training set (resubstitution) | 0.9068 | The model graded on the reviews it memorised. Always the highest. Not evidence of anything. |
| 5-fold cross-validation | 0.8779 | Our honest estimate, computed without ever touching the test set. This is what we chose the model with. |
| **Test set** | **0.8648** | **The real answer.** 25,000 reviews the model had never seen. |

### Why the test score is lower, and why that is good news

The three numbers go **highest on training, middle on cross-validation, lowest on test**. That
ordering is exactly what an honest project looks like. There are two reasons for the gap:

1. **We chose the configuration by its cross-validated score.** When you pick the best of 36 options
   by a measurement, the winner's score is partly genuine skill and partly good luck on that
   particular measurement. So the cross-validated number is always a little optimistic.
2. **The IMDB split deliberately puts different films in the test set.** No film in the test set
   appears in the training set. You can see the consequence in cell 44: the model has learned
   film-specific words — actor and character names like *felix*, *lemmon*, *matthau*. Those are
   genuinely useful for reviews of *that* film, and worth nothing for a film it has never seen.

**The warning sign would have been the opposite.** If the test score had come out *higher* than the
cross-validated one, that would suggest leakage or that we had been quietly tuning against the test
set. We were not, and the numbers show it.

### Per-class breakdown on the test set

|  | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.8595 | 0.8722 | 0.8658 |
| Positive | 0.8703 | 0.8574 | 0.8638 |

The two classes score almost identically, which is what you expect on balanced data. The model is
not biased toward either answer.

---

## 8. What to present in the video

The full script, timed to the second, is in **`docs/VIDEO_SCRIPT.md`** — use that as your actual
script. This section is the summary of *why* it is built that way.

You have about 5 minutes and 7 parts to cover, which is roughly 40 seconds each. So do not read code
aloud line by line. Point at the output, say what it means, and move on.

| Time | Cover | The one sentence that matters |
|---|---|---|
| 0:00–0:30 | Who you are, assignment type, learning type, algorithm, dataset | "The algorithm is implemented from scratch." |
| 0:30–1:15 | Loading the given split, both `.head()`s, the balance chart, and the metric choice | "The classes are balanced with no class of special interest, so the correct measurement is macro-F1." |
| 1:15–2:15 | Cleaning → tokens → stemming → Bag-of-Words → TF-IDF, on real reviews | "Bigrams are why the model can learn *not good* rather than *not* and *good* separately." |
| 2:15–3:15 | The Naive Bayes maths, the class, the correctness check | "It matches scikit-learn to 1e-13, so the implementation is provably right." |
| 3:15–4:15 | Grid search and cross-validation | "The vectorizer is rebuilt inside every fold — otherwise the vocabulary would leak information from the validation data." |
| 4:15–5:00 | Final training, the test result, the charts, conclusions | "Training highest, cross-validation middle, test lowest — that ordering is what an honest evaluation looks like." |

### The four things that win points

Most students can show a working notebook. These are what separate a good presentation:

1. **Explain leakage and show where you prevented it.** Point at the line inside the fold loop where
   the vectorizer is rebuilt. Say what would have gone wrong otherwise.
2. **Present your two negative results as findings, not failures.** Stop-word removal hurt, and you
   predicted it would, for a reason you can state. The `min_df` experiment was flat, and you can
   explain exactly which other setting made it flat. A negative result you can explain demonstrates
   more understanding than a positive one you cannot.
3. **Explain the gap between 0.8779 and 0.8648** rather than hoping nobody asks.
4. **Say what you skipped and why.** 6b (imbalanced-data handling) is meaningless on a 50/50
   dataset — saying so shows you understood what 6b is for.

### Practical recording tips

- Have the notebook open with all outputs already visible. **Do not run anything on camera.**
- Zoom to about 150 % so the code is readable in the recording.
- Upload as **unlisted** on YouTube — playable in a browser without downloading.
- Introduce yourself in the first sentence. It is an explicit requirement.

---

## 9. Questions you might be asked, and the answers

**"Why Naive Bayes?"**
It is the classic algorithm for text classification, it is fast, and — importantly for an assignment
about implementing something from scratch — the mathematics is small enough that I could implement
and fully explain it. It also gives interpretable output: I can show exactly which words drove any
individual prediction.

**"How do you know your implementation is correct?"**
Cell 29. I compared it against scikit-learn's implementation across several settings, on synthetic
data and on the real features. Probabilities agree to 1.42e-13 and every prediction is identical.

**"Isn't the independence assumption wrong?"**
Yes, completely — words in real sentences depend heavily on each other. Naive Bayes works anyway
because to classify correctly it does not need accurate probabilities, only the *correct ordering*
between the two classes. Adding bigrams also recovers some of the lost word-order information.

**"Where does the test set get used?"**
Part 5, at the end. Everything that makes a *decision* — all 36 grid combinations and both follow-up
experiments — uses only the training set, through 5-fold cross-validation. (As shipped, the
correctness check in Part 3 also predicts on the test set to compare against scikit-learn; no choice
is made from it, and section 6 explains how to remove it.)

**"Why is TF-IDF better than plain counting here?"**
Counting is dominated by words like *the* and *movie*, which appear in every review and separate
nothing. TF-IDF reduces their weight and raises the weight of distinctive words. The grid confirms
it: TF-IDF configurations occupy the top of the results table.

**"Why alpha = 0.1 rather than the usual 1.0?"**
It was chosen by cross-validation, not by preference. With a vocabulary of 30,000 features, a
smaller amount of smoothing turned out to fit this data better, and the grid table shows the
comparison across 0.1, 0.5 and 1.0.

**"Could you do better than 86.5 %?"**
Yes — a modern neural language model would reach the low-to-mid 90s on this dataset. That was not
the assignment. The assignment was to implement a learning algorithm from scratch and run an honest,
leakage-free process around it, and 86.5 % is a solid result for Naive Bayes on IMDB.

---

## 10. Submission checklist

### Values for the shared Excel sheet

| Field | Value |
|---|---|
| Assignment type | Text analysis (NLP) |
| Learning type | Classification (binary) |
| Algorithm implemented | Naive Bayes (Multinomial + Bernoulli), from scratch |
| Dataset name | IMDB 50K Movie Reviews |
| Dataset URL | https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert |
| Repository URL | https://github.com/Tomer-Raz/machine-learning-hit |
| Video URL | *(fill in after uploading)* |

### Fix this first

The correctness check in Part 3 (cell 29) scores the **test** set, which contradicts the "test set
used once" rule stated in Part 5. To fix it, edit `tools/build_notebook.py`, change the last block of
the Part 3c cell so the real-data comparison fits on one slice of the **training** matrix and checks
predictions on another slice, and drop the printed test score. Then rebuild and re-run (section 5).
It changes one printed line and removes the only rule violation in the notebook.

### Before you submit

- [ ] The Part 3 correctness check no longer touches the test set (see above)
- [ ] Notebook opens on GitHub with all outputs visible, no need to run it
- [ ] Student-details cell shows your name and the last 4 ID digits only — never the full number
- [ ] AI-prompts cell present
- [ ] Both `.head()` tables visible (train and test)
- [ ] First 5 test predictions visible
- [ ] Video recorded, about 5 minutes, uploaded unlisted, plays in a browser
- [ ] Video URL added to `README.md`, `docs/PROGRESS.md`, and the notebook's final Excel cell
      (edit `tools/build_notebook.py`, then rebuild **and** re-run — see section 5)
- [ ] Excel row filled in with all seven values above
- [ ] Everything committed and pushed to GitHub
