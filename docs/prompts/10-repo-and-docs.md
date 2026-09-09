# 10 — Repository and documentation

**Stage:** throughout, and at the end.
**Produced:** `tools/build_notebook.py`, `CLAUDE.md`, `docs/PROGRESS.md`, `README.md`.

---

## Prompt 1 — the notebook builder

```text
Editing a large .ipynb by hand is painful and produces unreadable diffs - JSON with embedded
outputs, base64 images, and execution counts churning on every run. I want a better workflow.

Set up tools/build_notebook.py: a plain Python script that GENERATES notebook.ipynb, where each
cell is a call to a small md(...) or code(...) helper, using nbformat.

Requirements:
- The generated notebook must still contain every line of code inline and visible, because the
  brief requires a self-contained notebook. This is a workflow convenience, not a way of hiding
  code in an imported module.
- Cell sources are r-strings in the builder, so nothing inside them may contain triple quotes -
  code cells use # comments instead of docstrings. Note that constraint at the top of the file so I
  do not trip over it later.
- Document the build-and-execute command in the file header:
  build the notebook, then run it with jupyter nbconvert --execute --inplace so the outputs are
  committed.
- Make the section boundaries obvious with banner comments, so I can find Part 3 in a 900-line
  builder.

Then tell me the one rule I must never break with this setup.
```

**What came back:** the builder, plus the rule: **never hand-edit `notebook.ipynb`** — any manual
change is destroyed on the next build. That rule is recorded at the top of the builder and in
`CLAUDE.md`, because breaking it silently loses work.

**Trade-off I accepted:** a full rebuild means re-executing everything (~7 minutes), so even a
one-word markdown fix costs a full run. Worth it for reviewable diffs.

---

## Prompt 2 — project memory across sessions

```text
This project runs across several sessions, days apart, and I will forget the details between them.
Set up documentation so I - or an assistant with no memory of previous sessions - can resume
quickly.

Write two files:

1. CLAUDE.md: orientation. What this project is, the locked decisions (dataset, algorithm, metric,
   scope) in a table, the submission facts, the conventions that must be followed (git identity,
   the never-hand-edit-the-notebook rule, what scikit-learn may and may not be used for), and the
   hard requirements that are easy to miss. At the top, a short CURRENT STATUS block saying what is
   done and what the next action is.
2. docs/PROGRESS.md: a living checklist - milestones with checkboxes, a decisions log with dates
   and the reasoning behind each decision, and any open questions.

Both must be updated at the END of every working session, and CLAUDE.md must say so explicitly.

The decisions log matters most: record WHY, not just what. In three weeks I will not remember why I
rejected the alternative dataset.
```

**What came back:** both files. They earned themselves back at the start of this session — after
several days away, they were what reconstructed the state of the work in a couple of minutes.

**Lesson learned the hard way:** they were allowed to go stale mid-project — the status block still
said "next: add Part 2" long after Part 2 and the grid search were finished. Documentation that is
not updated at the moment the work happens becomes actively misleading, which is worse than absent.

---

## Prompt 3 — repository hygiene

```text
Prepare the repository for submission. It is public on GitHub and a grader will open it without
downloading anything.

- Write a README aimed at a human landing on the repo cold: the problem, the approach, what is in
  each file, how to run it, dataset credit with the licence and the academic citation, and a
  submission-links table.
- Write .gitignore so the virtualenv, the downloaded data, notebook checkpoints and any Kaggle
  credentials never get committed. The notebook itself, WITH its outputs, must be committed - that
  is the deliverable.
- Confirm my full ID number appears nowhere in the repository, only the last four digits.
- Set the git identity for this repo locally to my personal account, since the machine's global
  identity is a different work account, and never add AI co-authorship trailers to the commits.
```

**What came back:** the `README.md`, the `.gitignore`, and the repo-local git identity. The
data-exclusion rule is why `data/` contains only a `README.md` explaining how the notebook fetches
the files.
