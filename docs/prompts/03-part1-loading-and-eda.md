# 03 — Part 1: loading the given split, and the first look at the data

**Stage:** notebook Part 1.
**Produced:** `load_imdb()`, the two `.head()` cells, the EDA cell (class balance, length stats,
overlap check), and the mandatory student-details and AI-prompts cells.

---

## Prompt 1 — the loader, with the constraint stated up front

```text
Write the Part 1 data-loading cell for the notebook. Requirements, in priority order:

- Load the dataset's OWN train/test split - the shipped train.csv and test.csv. Do not re-split,
  do not concatenate them and re-split, do not shuffle across the boundary. This is a hard
  requirement of the brief and the single easiest way to lose points.
- Download via kagglehub (anonymous works for this dataset), and fall back to reading CSVs from a
  local data/ folder if the download fails, so the notebook still runs offline or for a grader who
  drops the files in manually.
- Normalise the columns: rename text -> review and sentiment -> label, map pos -> 1 and neg -> 0.
  Be tolerant of the column-name variants the file might use, but fail loudly if the expected
  columns are absent - I would rather see an exception than silently get an empty frame.
- Drop exact-duplicate review texts INSIDE the training set only. The test set must come back
  exactly as shipped, untouched, with all 25,000 rows.
- Return (df_train, df_test) with a reset index on each.
- Print the resolved file paths and the resulting shapes so the output shows what was actually
  loaded.

Write it as one readable function plus the call. Comments should explain the why, not restate the
code.
```

**What came back:** `load_imdb()` essentially as it stands in the notebook — `kagglehub` download
with a `data/` fallback, a `_read` helper handling the column renames and the label mapping, and
`drop_duplicates(subset="review")` applied to the training frame only.

**What I checked:** that the returned test frame still has exactly 25,000 rows, and that the
duplicate drop is applied on the train side only. The printed shapes (24,904 / 25,000) are in the
committed output precisely so this is verifiable without re-running.

---

## Prompt 2 — the corporate TLS proxy problem

```text
The kagglehub download fails on my machine with an SSL certificate verification error. I am behind
a corporate network that intercepts TLS with its own root CA, which Python's bundled certifi store
does not trust. The system trust store does have the certificate - browsers and curl work fine.

Fix this INSIDE the notebook, with these conditions:
- It must be a no-op on a normal network and on Google Colab, since a grader will run it there.
- Do not disable certificate verification. I am not shipping verify=False in a submitted
  assignment.
- Do not require me to set environment variables or install anything outside requirements.txt.

Explain the fix in one comment in the setup cell so a reader understands why the code is there.
```

**What came back:** `truststore.inject_into_ssl()` in a `try`/`except` in the setup cell — it makes
Python use the operating system's trust store, and the `except` makes it silently inert anywhere it
is not needed.

**Why I kept it:** it solves the problem without weakening verification, and it does not change
behaviour for anyone running the notebook elsewhere.

---

## Prompt 3 — EDA that earns its place

```text
Write the Part 1 exploratory-data-analysis cell. I do not want a wall of generic pandas output -
every line should be something a viewer of my video would care about. Cover:

- Shape and class balance of each split, with one clear bar chart showing both splits side by side.
  State explicitly whether the classes are balanced, because that decision feeds directly into my
  choice of quality metric later.
- Review-length statistics in words: mean, median, and the spread, per split. I want to know
  whether train and test look like the same distribution.
- A check for how many review texts appear in BOTH splits. Report the count. Do not remove them -
  the test set stays untouched - but print it so I can disclose it as a caveat.
- Print a couple of full example reviews, one positive and one negative, so the viewer sees what
  the raw data actually looks like before any cleaning.

Keep it to one cell and make the printed output readable in a recording.
```

**What came back:** the EDA cell as it stands. The overlap check reported ~123 shared review texts,
which is where the caveat in the notebook and in `CLAUDE.md` comes from.

**What I changed:** I asked for the class-balance chart to plot both splits rather than only the
training set, because "the test set is balanced too" is part of justifying macro-F1 as the metric.

---

## Prompt 4 — the two mandatory cells

```text
The brief requires two specific cells that are easy to forget. Draft both:

1. A student-details cell: my first name, the first letter of my surname, and the LAST FOUR DIGITS
   of my ID only. Under no circumstances put my full ID number in this repository - it is public
   on GitHub.
2. An AI-prompts cell: which AI tools I used, representative prompts, the non-AI resources I
   consulted, and what each was used for. It must be honest about the extent of the assistance -
   I would rather disclose fully than have a grader wonder.

For the second one, keep it to a readable markdown table in the notebook. Assume I will also keep
a longer prompt record in docs/ for anyone who wants the detail.
```

**What came back:** the title/student-details cell (`Tomer R.` · `5130`) and the AI-assistance
markdown cell that stands as section 1a of the notebook.

**What I enforced:** the full ID appears nowhere in the repository — only the last four digits, as
the brief specifies. This prompt folder is the "longer record" that cell refers to.
