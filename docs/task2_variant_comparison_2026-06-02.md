# Task 2 — SEM Variant Comparison: Findings (2026-06-02)

## Purpose
Address the reviewer concern *"does the full SEM's inference machinery add value
beyond simpler alternatives?"* by comparing the full model against three nested
ablations, each evaluated across **100 sampled parameter combinations** (so the
comparison is robust to the choice of any single parameter setting).

All four variants reuse the existing `SEMScenario` infrastructure with
constrained parameters — **no separate baseline model**.

| Variant | Constraint | Removes |
|---|---|---|
| SEM-full | none | — |
| SEM-context-blind | P(m_sit) = 0 | context-sensitive motivation inference |
| SEM-knowledge-only | ω = 1.0 | motivation inference |
| SEM-motivation-only | ω = 0.0 | knowledge-state inference |

Sampling: ω ~ Beta(2,2) clipped to (0.05, 0.95) for the free variants; P(k) ~
Dirichlet(1,1); P(m) ~ Dirichlet(1,1,1) (Dirichlet(1,1) over {m_max, m_min} for
context-blind). `numpy.random.default_rng(seed=42)`. Metrics from `sem/fit.py`
(Mühlenbernd 2026); φ derived at **p < .01** (the current committed matrices).

## Results (n = 100 samples per variant)

| Metric | SEM-full | Context-blind | Know-only | Motiv-only |
|---|---|---|---|---|
| DAS | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.96 ± 0.13 |
| **ISS** | **1.00 ± 0.00** | **0.00 ± 0.00** | **0.04 ± 0.14** | **1.00 ± 0.00** |
| Spearman ρ | 0.53 ± 0.28 | 0.26 ± 0.31 | **0.75 ± 0.09** | −0.08 ± 0.27 |
| RMSE | **0.30 ± 0.10** | 0.42 ± 0.12 | 0.39 ± 0.14 | 0.51 ± 0.09 |
| CCC | 0.52 ± 0.27 | 0.25 ± 0.29 | 0.60 ± 0.13 | 0.02 ± 0.25 |
| CDS (main) | 1.06 ± 0.71 | 1.46 ± 0.72 | 1.65 ± 1.25 | 1.07 ± 0.56 |
| CDS (inter) | 0.82 ± 0.62 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.79 ± 1.44 |
| CDS (overall) | **0.94 ± 0.43** | 1.23 ± 0.36 | 1.32 ± 0.62 | 1.43 ± 0.67 |
| ESR main a_comp | 2.32 ± 1.16 | 2.85 ± 0.99 | 3.28 ± 1.95 | 1.78 ± 1.10 |
| ESR main a_ped | 1.69 ± 0.48 | 2.05 ± 0.58 | 1.63 ± 0.97 | 2.03 ± 0.48 |
| ESR inter a_comp | 1.37 ± 0.98 | 0.00 ± 0.00 | 0.00 ± 0.00 | 2.69 ± 1.67 |
| ESR inter a_like | 1.35 ± 0.95 | 0.00 ± 0.00 | 0.00 ± 0.00 | 2.65 ± 1.50 |

Source: `data/cs1_experiment/processed/variant_comparison_cs1.json`
(reproduce: `PYTHONPATH=. python scripts/run_comparison.py`).

## Calibration — the central result of this section

The purpose of this comparison is **calibration performance on the four
significant CS1 effects** (two main effects: a_comp, a_ped; two interactions:
a_comp, a_like), measured by the Effect Size Ratio (ESR) and the Calibration
Deviation Score (CDS = mean |ESR − 1|), plus RMSE as a calibration measure over
all 12 data points.

**ESR per effect** (|model| / |human|; 1.0 = perfect):

| | SEM-full | Context-blind | Know-only | Motiv-only |
|---|---|---|---|---|
| *Main effects* | | | | |
| ESR a_comp | 2.32 ± 1.16 | 2.85 ± 0.99 | 3.28 ± 1.95 | 1.78 ± 1.10 |
| ESR a_ped | 1.69 ± 0.48 | 2.05 ± 0.58 | 1.63 ± 0.97 | 2.03 ± 0.48 |
| **ESR main (aggregate)** | **2.00 ± 0.78** | 2.45 ± 0.75 | 2.46 ± 1.46 | 1.90 ± 0.73 |
| *Interactions* | | | | |
| ESR a_comp | 1.37 ± 0.98 | 0.00 ± 0.00 | 0.00 ± 0.00 | 2.69 ± 1.67 |
| ESR a_like | 1.35 ± 0.95 | 0.00 ± 0.00 | 0.00 ± 0.00 | 2.65 ± 1.50 |
| **ESR inter (aggregate)** | **1.36 ± 0.96** | 0.00 ± 0.00 | 0.00 ± 0.00 | 2.67 ± 1.57 |

**CDS by category and RMSE** (lower = better):

| | SEM-full | Context-blind | Know-only | Motiv-only |
|---|---|---|---|---|
| **CDS (main)** | **1.06 ± 0.71** | 1.46 ± 0.72 | 1.65 ± 1.25 | 1.07 ± 0.56 |
| **CDS (interaction)** | **0.82 ± 0.62** | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.79 ± 1.44 |
| **CDS (overall)** | **0.94 ± 0.43** | 1.23 ± 0.36 | 1.32 ± 0.62 | 1.43 ± 0.67 |
| **RMSE (all 12 cells)** | **0.30 ± 0.10** | 0.42 ± 0.12 | 0.39 ± 0.14 | 0.51 ± 0.09 |

**Result: the full model is the best-calibrated variant on every measure.** It has
the lowest CDS for main effects, for interactions, and overall, and the lowest
RMSE across all data points. No ablation matches it. This is the headline of the
section.

Reading the detail:
- The full model's **interactions are better calibrated (CDS 0.82) than its main
  effects (CDS 1.06)** — it captures relative context-modulation better than
  absolute effect size. The main-effect over-shoot (ESR ≈ 2×) is the dominant
  residual miscalibration, and is shared by all variants (it reflects the
  trivalent impact scale, not a full-model-specific flaw).
- The ablations expose distinct calibration failures on the interactions:
  context-blind and knowledge-only **cannot represent interactions at all**
  (ESR = 0 → CDS pinned at 1.00); motivation-only **massively over-shoots** them
  (ESR ≈ 2.7, CDS 1.79).
- Honesty note: on CDS-main the full model (1.06) only marginally beats
  motivation-only (1.07) — effectively a tie on main-effect calibration. The
  full model's clear advantages are on the **interaction** and **overall** CDS
  and on **RMSE**.

## Key findings (structure)

1. **The interaction effects require motivation inference via `m_sit`.** ISS
   collapses to ≈0 for the two variants that lose `m_sit` (context-blind 0.00,
   knowledge-only 0.04) and stays at 1.00 for the two that retain it (full,
   motivation-only). Since 2 of the 4 significant empirical effects are
   form×context interactions, this is the central evidence that the contextual
   motivation mechanism is **necessary**, not decorative.

2. **Each ablation fails on a distinct dimension; none dominates the full model
   on both structure and calibration.**
   - *Context-blind*: loses interactions (ISS 0) **and** degrades globally
     (Spearman 0.53→0.26, RMSE 0.30→0.42, CCC 0.52→0.25, CDS 0.94→1.23).
   - *Motivation-only*: keeps interaction direction (ISS 1.00) but is otherwise
     the worst — Spearman −0.08, CCC 0.02, and it overshoots interaction
     magnitude (ESR ≈ 2.7), because without knowledge inference the competence
     ordering breaks.
   - *Knowledge-only*: see nuance below.

3. **The full model is the only variant that captures interaction structure
   without sacrificing global fit** — ISS 1.00, best RMSE (0.30), lowest CDS
   (0.94).

## Nuance to flag (do not bury)

**Knowledge-only has the *highest* global metrics**: Spearman ρ = 0.75 (vs full
0.53) and CCC = 0.60 (vs full 0.52). The CS1 main-effect ordering is largely
epistemic, so an ω=1 model that ignores motivation actually tracks the overall
ranking slightly better. Its fatal flaw is **ISS ≈ 0** — it cannot represent the
interactions at all.

So the argument for the full model is **not** "best on every metric." It is:
*the full model is the only variant that reproduces the interaction structure
(ISS 1.00) while remaining competitive globally; every simpler variant that
matches or beats it on one axis fails badly on the other.* We should frame this
deliberately rather than letting a reviewer surface the knowledge-only Spearman
edge first.

## Threshold robustness (resolved)
The φ functions are derived at **p < .01**. We re-ran the full comparison with
φ derived at **p < .05** (which adds three non-zero cells: k_p→a_ped +1,
m_max→a_comp +1, m_max→a_like −1). The looser threshold made the full model
**worse on every calibration/global metric** — e.g. CDS overall 0.94 → 1.45,
RMSE 0.30 → 0.51, Spearman 0.53 → 0.25 — while ISS was unchanged. This is a
post-hoc justification for the p < .01 choice and preempts a "why not .05?"
review question. Going the other way, **p < .001** marginally lowers overall CDS
(0.94 → 0.89), main-effect CDS (1.06 → 0.89) and RMSE (0.30 → 0.29) by trimming
the over-shoot, but it degrades structure — it drops the borderline m_sit→a_comp
and m_min→a_comp cells, so the full model can no longer reproduce the competence
interaction (ISS 1.00 → 0.57, CDS-interaction 0.82 → 0.90, ESR-interaction a_comp
→ 0); p < .01 therefore remains the best threshold, the only one that keeps full
interaction coverage while staying well-calibrated. (Exploratory script:
`scripts/_explore_threshold.py`.)

We also checked **Pearson r** (alongside Spearman): full 0.59, context-blind
0.30, knowledge-only 0.79, motivation-only 0.02 — same ordering as Spearman, so
no change to conclusions. (Pearson is not part of the committed `fit.py` suite.)

## Implementation
- `sem/comparison.py` — sampling, evaluation, summarization, table printing.
- `scripts/run_comparison.py` — runner; saves the summary JSON.
- `TestVariantComparison` in `tests/test_case_studies.py` (9 tests).
- Runner wired into `case_studies/cs1_imprecision.py` `__main__`.
- Test suite: 41 passing. Runtime ~1.6 s for the full comparison.
