# 01 — Decoding the assignment brief

**Stage:** before any code.
**Produced:** `docs/ASSIGNMENT.md` (decoded spec, point map, notebook skeleton, hard constraints).

The brief (`ml.md`) is a machine translation from Hebrew and is garbled in places — section
numbering drifts, and the description of the quality metric is close to unreadable. The first job
was to turn it into something I could execute against.

---

## Prompt 1 — decode the brief into a working spec

```text
Read ml.md in this repo. It is a machine translation from Hebrew of my Machine Learning course
assignment brief, and it is garbled in places - the section numbering drifts and some sentences
are mangled.

Produce docs/ASSIGNMENT.md containing:

1. A plain-English statement of what the assignment actually asks for, part by part.
2. A point map: a table of each part, what it must contain, and its maximum score.
3. The list of deliverables and what "done" means for each.
4. A proposed notebook skeleton - the ordered list of sections and, for each, whether it is a
   markdown cell or a code cell and what goes in it.
5. A "hard constraints" section: the requirements that are easy to fail silently, i.e. things
   where the notebook would still run and look fine but would lose points.

Rules for this task:
- Where the translation is ambiguous, do NOT guess and move on. List the ambiguity explicitly and
  give me the candidate readings so I can decide.
- Quote the original garbled sentence next to your interpretation wherever you had to interpret.
- Do not write any project code yet. This step produces a document only.
```

**What came back:** the first version of `docs/ASSIGNMENT.md`, with three ambiguities flagged
rather than resolved: which metric applies to which problem shape, whether "section 5" in the
translation meant Part 5 or the extension in section 6, and whether the k-fold extension was
required or optional.

**What I checked:** I re-read the Hebrew-origin passages against the interpretation for each of
the three flagged items and settled them myself. The metric rule became its own investigation —
see [04](04-quality-metric.md).

---

## Prompt 2 — scope decision

```text
Using the point map you just produced, help me decide scope. My constraints: I am working solo,
I have limited time, and the brief says assessment is primarily on process and understanding
rather than on model quality.

Tell me:
- which parts I must do to reach roughly 100 points;
- what a "light" version of the 6a grid-search extension would look like - the minimum that still
  earns its points honestly, given the scoring is per feature-engineering technique experimented
  with and per hyper-parameter tuned;
- exactly what I lose by skipping 6b (imbalanced-data handling) and 6c (explainability, e.g.
  SHAP), and whether skipping 6b is defensible for a dataset that is already balanced 50/50.

Give me a recommendation, not a menu of options.
```

**What came back:** the recommendation to do Parts 1–5 in full plus a 6a with at least two
feature-engineering axes and one hyper-parameter axis, and to skip 6b and 6c — 6b in particular
being close to meaningless on a 50/50 balanced dataset, which is worth *saying* in the conclusions
rather than silently omitting.

**What I did with it:** adopted it as the locked scope. The conclusions cell in the notebook names
both skipped extensions and why, so the omission reads as a decision rather than an oversight.

---

## Prompt 3 — the silent-failure list

```text
From the brief, extract every requirement that I could satisfy incorrectly without the notebook
throwing an error - the ones where the code runs, the output looks plausible, and I still lose the
points. For each one, give me the requirement, the plausible-looking wrong version, and a concrete
way to check which one I have done.

I specifically want you to cover: how the train/test split must be obtained, where cross-validation
is allowed to run, when the test set may be touched, and what "demonstrate the feature engineering
on examples" actually requires.
```

**What came back:** the checklist that became the "Hard requirements easy to miss" section of
`CLAUDE.md`. The four that mattered most: load the dataset's *own* split and never re-split;
cross-validate inside the training set only; touch the test set exactly once, at the end; and show
the feature pipeline stage by stage on named example rows from **both** splits, not just describe
it in prose.

**What I did with it:** kept that list open while building every subsequent part, and turned it
into the audit in [09](09-verification-and-review.md) once the notebook was complete.
