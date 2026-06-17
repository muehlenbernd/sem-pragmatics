# SEM Revision Project — Working Plan

> **✅ COMPLETE — historical planning document.** The project shipped as a
> **single case study (CS1: numerical (im)precision)**. All code underlying the
> current manuscript is implemented and tested (**41 tests passing**): impact
> derivation (`sem/impact_derivation.py`), quantitative fit (`sem/fit.py`), and
> the nested-ablation variant comparison (`sem/comparison.py`).
>
> The following items below were **intentionally descoped** relative to this
> plan and are *not* part of the final paper:
> - **RSA baseline** (`baselines/rsa_baseline.py`, Issue 1 / Step B) — RSA is
>   addressed discursively in the paper (§"Relation to the RSA Framework"), not
>   as code; `baselines/` is empty.
> - **Norm-violation baseline** (`baselines/norm_violation_baseline.py`,
>   Issue 2 / Step C) — replaced by the variant comparison (paper Table 2).
> - **Programmatic Bayesian-network figure** (`figures/generate_bayesian_network.py`,
>   Step E) — `semNet.pdf` is hand-drawn.
> - **Case Study 2** (`case_studies/cs2_pragmatic_violations.py`) — exists in
>   code and tests but is *not* in the paper; it now appears only as future work
>   (§"Extending SEM…"). The paper is **CS1-only**.
>
> Repo-structure, test-count, and "to be created" notes below are stale.
> This file (with `TASKS.md` and `README.md`) is slated for the deferred
> repo cleanup gated on manuscript finalization; it is retained as-is for
> historical context.

## Context

This repository contains the Python implementation of the **Social Evaluation
Model (SEM)**, supporting the paper:

> "Modeling Pragmatic Reasoning behind Social Meaning"  
> Target journal: *Meaning: A Journal of Linguistics and Philosophy*  
> (Resubmission after rejection from *Open Mind*)

The paper formalizes how listeners form social judgments of speakers (competence,
likeability, pedantry) based on pragmatic reasoning about utterance choices.
Two case studies are implemented: numerical (im)precision and pragmatic norm
violations (relevance + informativeness).

---

## Repository Structure

```
sem_project/
├── sem/
│   ├── __init__.py          # Package init
│   ├── model.py             # Core SEM engine (SEMScenario class)
│   └── plotting.py          # Reusable figure generation
├── case_studies/
│   ├── cs1_imprecision.py          # Case Study 1: precise vs. approximate
│   └── cs2_pragmatic_violations.py # Case Study 2: relevance & informativeness
├── tests/
│   └── test_case_studies.py        # Pytest suite (19 tests, all passing)
├── figures/                         # Output directory for generated figures
├── PLAN.md                          # This file
└── requirements.txt                 # (to be created)
```

---

## Key Model Facts (for Cursor context)

The core evaluation function (Eq. 1 in paper):

    E_L(a | v, c) = Σ_k Σ_m  P(k|v) · P(m|k,v,c) · I(a|k,m; ω)

- `v` = utterance, `c` = context, `a` = social attribute  
- `k` = speaker knowledge state, `m` = motivation-based strategy  
- `ω` ∈ (0,1) = weight for knowledge vs. motivation  
- Output ∈ [-1, 1]

The `SEMScenario` class in `sem/model.py` implements all components.
New scenarios are created by subclassing or calling the factory functions
`build_scenario()` in each case study file.

---

## Revision Goals (from peer review)

The paper was rejected by *Open Mind* with these main criticisms:

### Issue 1 — RSA relationship [CRITICAL, must address first]
> Reviewer 1 & Editor: "It's not clear a new model is needed — SEM appears to
> be a special case of RSA. Why not formulate this within RSA?"

**Plan:** Implement a minimal RSA-with-social-attributes extension and show
formally where predictions diverge. Either (a) reframe SEM as an RSA extension
with social evaluation as a novel inference target, or (b) argue SEM is simpler/
more tractable and makes distinct predictions.

**Files to create:** `baselines/rsa_baseline.py`

---

### Issue 2 — Quantitative model evaluation [HIGH PRIORITY]
> Reviewer 1: "The 2×2 designs don't generate fine-grained predictions.
> What does SEM predict that a simpler model would not?"

**Plan:**
- Implement a "norm violation heuristic" baseline (simple additive penalty model)
- Compare both models' predictions on the same test cases quantitatively
- Plot model predictions alongside empirical data in the SAME figure

**Files to create:** `baselines/norm_violation_baseline.py`  
**Files to modify:** `sem/plotting.py` — add joint empirical+model plots

---

### Issue 3 — Formalization accessibility [MEDIUM PRIORITY]
> Reviewer 2: "Hard to work through the equations. Add subheadings, intuitions
> before equations, and a Bayesian network diagram."

**Plan:**
- Generate a Bayesian network diagram programmatically (using graphviz or
  matplotlib)
- The diagram should show: v → K → M → E_L(a), with ω and priors as parameters

**Files to create:** `figures/generate_bayesian_network.py`

---

### Issue 4 — "Social meaning" terminology [MEDIUM PRIORITY]
> Reviewer 1: "The term 'social meaning' is used too broadly. In sociolinguistics
> it refers to identity/group membership, but here it means speaker trait
> inference (competence, likeability)."

**Plan:** Purely a writing/framing fix. No code changes needed.  
Terminology to adopt: "evaluative inference" or "social attribute inference"
as the paper's specific contribution, distinct from identity-indexical social
meaning (Labov, Eckert, SMG framework).

---

### Issue 5 — Transparency & reproducibility [HIGH PRIORITY]
> All reviewers: Missing GitHub link (footnote 1 placeholder), no data sharing,
> no preregistration statement.

**Plan:**
- Set up public GitHub repository
- Add experimental data files (anonymized)
- Add R analysis scripts for the imprecision experiment
- Add clear README with reproduction instructions
- Add honest framing in paper: the model fitting in CS1 is post-hoc (derived
  from data), while CS2 is a prospective application (extrapolated from CS1)

**Files to create:** `data/README.md`, `analysis/` (R scripts)

---

### Issue 6 — Failure case analysis [LOW-MEDIUM PRIORITY]
> Editor: "What about the 0.3% / 3.5% parameter combinations that fail?
> Were those plausible configurations?"

**Plan:**
- Use `SEMScenario.robustness_test()` to collect the failure cases
- Analyze which parameter configurations cause failures
- Check if they correspond to plausible/implausible real-world priors

**Files to modify:** add analysis notebook or script for failure cases

---

## Step-by-Step Work Order

### Step A — Clean up existing code ✅ DONE
- Refactored 3 original notebooks into modular Python package
- `sem/model.py`: clean, documented `SEMScenario` class matching paper equations
- `case_studies/cs1_imprecision.py`: Case Study 1 with all definitions
- `case_studies/cs2_pragmatic_violations.py`: Case Study 2 with all definitions
- `sem/plotting.py`: reusable figure functions
- `tests/test_case_studies.py`: 19 tests, all passing

**One finding from Step A:** Under the balanced model, v_B and v_b score
identically on competence in CS2. This is because φ_K is identical for k_aB
and k_ab on a_comp. Worth discussing in the paper.

---

### Step 0 — Empirical impact matrix derivation ✅ READY FOR CLAUDE CODE
**See `TASKS.md` Task 0 for full specification.**

Key outputs:
- `sem/impact_derivation.py`: reusable derivation functions (also used in notebook)
- `data/cs1_experiment/processed/impact_tables_cs1.json`: lookup tables at all
  four thresholds (p<.001, p<.01, p<.05, p<.10) — for later robustness checks
- `scripts/derive_impact_tables.py`: standalone runnable derivation script
- Updated `IMPACT_K` and `IMPACT_M` in `cs1_imprecision.py` (p<.01 values)
- Updated tests

**p<.01 impact matrices (target values after update):**

    IMPACT_K:
      k_p:  a_comp=+1, a_like=+1, a_ped= 0   (a_like: 0 → +1)
      k_np: a_comp=-1, a_like=-1, a_ped=-1   (unchanged)
    IMPACT_M:
      m_max: a_comp= 0, a_like= 0, a_ped=+1  (a_like: -1 → 0)
      m_sit: a_comp=+1, a_like=+1, a_ped= 0  (a_ped: -1 → 0)
      m_min: a_comp=-1, a_like= 0, a_ped= 0  (a_like: -1 → 0)

**Proxy-to-SEM mapping:**
- k_p  → t2_knewExact
- k_np → t2_notKnow
- m_max → t2_fussyPerson
- m_sit → t2_needed, t2_purpose, t2_realWorld  (majority vote)
- m_min → t2_easySpeaker, t2_usualApx          (majority vote)

After updating impact matrices, verify all four CS1 target effects still hold.

**Future robustness check (not now):** The JSON lookup table at all thresholds
enables running the SEM with matrices derived at each threshold and comparing
model performance across threshold choices.

---

### Step 1 — Fit metrics (`sem/fit.py`) [AFTER STEP 0]
**See `TASKS.md` Task 1 for full specification (to be written after Step 0).**

Implement the metric set from Mühlenbernd (2026):
- `normalize_likert()`, `das()`, `iss()`, `esr_per_effect()`, `cds()`
- `spearman_r()`, `rmse()`

These enable quantitative model evaluation and comparison against baselines.

---

### Step B — RSA comparison [AFTER STEP 1]
1. Implement `baselines/rsa_baseline.py`:
   - A standard RSA listener (L1) that infers speaker meaning
   - Extended with a social attribute layer: E_RSA(a|v,c) derived from the
     RSA speaker distribution
2. Run both SEM and RSA-extended on CS1 and CS2
3. Find at least one prediction where they diverge
4. Document: what does SEM capture that RSA-extended does not (or vice versa)?

---

### Step C — Baseline comparison + joint figures
1. Implement `baselines/norm_violation_baseline.py`:
   - Simple rule: each norm violation applies a fixed penalty per attribute
   - Fit penalty weights to CS1 data (same post-hoc procedure as SEM)
   - Compare prediction accuracy on CS2 (held-out, in a sense)
2. Update `sem/plotting.py`:
   - `plot_cs1_with_empirical()`: model predictions + experimental bar chart
     in the same figure (dual axis or normalised to same scale)
   - `plot_cs2_with_empirical()`: same for CS2

---

### Step D — Robustness & failure analysis
1. Run `robustness_test()` on both case studies (use step=0.05 for speed,
   step=0.02 for final paper-quality run)
2. Collect failure cases
3. Analyse: what parameter configurations fail? Are they realistic?
4. Generate robustness summary figures

---

### Step E — Bayesian network diagram + new figures
1. `figures/generate_bayesian_network.py`:
   - Programmatic diagram of the SEM generative model
   - Nodes: v, K, M, a, ω, P(k), P(m)
   - Directed edges following the plate notation
2. Review all figures for consistency with *Meaning* house style
   (accessible to philosophy-of-language readers)

---

## Notes for Cursor / Claude Code Sessions

- **Task list:** See `TASKS.md` for concrete, ordered implementation tasks
- **Always run tests before and after changes:** `PYTHONPATH=. python -m pytest tests/ -v`
- **Core equation is in `sem/model.py`** — do not change the mathematical
  structure without updating the docstring and the paper draft simultaneously
- **The paper draft is not in this repo** — changes to predictions or
  interpretations should be flagged for manual update in the manuscript
- **Naming convention:** knowledge states use `k_`, strategies use `m_`,
  attributes use `a_`, utterances use `v_`, contexts use `c_`
- **Check-in with main Claude.ai chat** at the end of each step (A–E) for
  review against the scientific/rhetorical goals of the revision

---

## Key References

- Frank & Goodman (2012): RSA framework — main model to compare against
- Burnett (2017, 2019): Social Meaning Game — key contrast model
- Beltrama & Papafragou (2023): Source of CS2 empirical data
- Beltrama, Solt & Burnett (2022): Source of CS1 empirical patterns
- Kao et al. (2014): RSA for numerical imprecision (editor recommended this)

---

## Journal Target: *Meaning*

- Diamond Open Access, hosted at Ruhr University Bochum
- Scope: intersection of philosophy of language + linguistic semantics/pragmatics
- Key requirement: **accessibility** — formalisms must be readable by
  philosophers who are not specialists in computational pragmatics
- Allows appendices; GitHub/data links encouraged (CC BY 4.0)
- Fast turnaround is a stated goal of the journal
