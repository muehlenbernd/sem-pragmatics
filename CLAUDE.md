# CLAUDE.md

Standing instructions for Claude Code in this repository. Session-specific
tasks live in `revision/` (untracked; see below).

## What this repo is

The Python implementation of the Social Evaluation Model (SEM): a listener
model that derives social evaluations of a speaker (competent, likeable,
pedantic) from inferences about the speaker's knowledge state and motivation.
It is the code and data companion of a journal paper that is **under
anonymous review**. The repo is mirrored to an anonymous review link, so
everything committed here is visible to reviewers.

## Layout

```
sem/            core package: model.py (SEMScenario), fit.py (metrics),
                comparison.py (variant comparison, parameter sampling),
                impact_derivation.py (phi tables from Task 2), plotting.py
case_studies/   cs1_imprecision.py (main case study), cs2_pragmatic_violations.py
scripts/        runnable analyses (run from the repo root)
scripts/dev/    helpers that (re)build the notebooks; not part of the analysis
data/cs1_experiment/
    processed/  experiment1.csv (N = 362), experiment2.csv (N = 390), derived JSON
    codebook.csv
notebooks/      sem_paper.ipynb (reproduces the paper), sem_demo.ipynb
tests/          pytest suite
docs/           design notes
revision/       UNTRACKED (gitignored): revision planning files, session briefs,
                results reports. Read them; never commit them.
```

## How to run

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q          # must pass before every commit
PYTHONPATH=. python scripts/<script>.py   # all scripts run from the repo root
```

Long runs (the .02 parameter grid has ~2.6 million settings) go in the
background or behind a `slow` pytest marker; never block the test suite on them.

## Conventions

- **Notation.** Code uses `v_prc` / `v_apx` for utterances; the paper writes
  `u`. Keep the identifiers; in prose and docs write "u (in code: v)" once.
  Knowledge states `k_p`, `k_np`; strategies `m_max`, `m_sit`, `m_min`;
  contexts `c_HP`, `c_LP`; attributes `a_comp`, `a_like`, `a_ped`.
- **Data values.** `form`: `precise` / `approx`; `context`: `highPr` / `lowPr`;
  `knowledge`: `unknown` (Experiment 1) / `knowl` (Experiment 2).
- **Equation numbers (revised paper).** (1) P(k|u), (2) P(m|k,u,c),
  (3) impact function I(a|k,m;ω), (4) evaluation E_L(a|u,c).
- **Randomness.** Seed 42 everywhere (`numpy.random.default_rng(42)`). The
  "100 sampled parameter combinations" of the paper are exactly
  `sem.comparison.sample_parameters("full", n_samples=100, seed=42)`.
- **Results already reported must not change.** The staged listener is the
  default and its numbers are in the paper (balanced model; sampled RMSE
  0.30 ± 0.10, CDS 0.94 ± 0.43; .02 grid 99.87 %). Any refactor must
  reproduce them exactly; add a test before refactoring if none covers it.
- **Do not edit committed derived data** (`empirical_means_cs1.json`,
  `impact_tables_cs1.json`, `variant_comparison_cs1.json`) unless regenerating
  them produces identical content. New outputs get new file names.
- Python 3.9+, numpy/pandas/scipy/statsmodels. No new heavy dependencies
  without asking.

## Anonymity (while under review)

- No author names, affiliations, personal URLs, OSF links or e-mail addresses
  in anything you add or change (code, comments, docs, commit messages).
  Exception already decided: the metric suite is cited as "Mühlenbernd (2026,
  CMCL)" (third-person citation, as in the paper). Leave `LICENSE` alone.
- Refer to the companion experimental paper as "Author (2025)" and to the data
  archive as "archived publicly; link added after peer review".

## Git

- Work on the branch `meaning-revision`, never on `main`.
- One commit per task step, with a clear message. Run the tests first.
- **Never push, merge, force-push or rewrite history.** The owner reviews and
  pushes.
- Ask before deleting anything not explicitly listed in the session brief.
