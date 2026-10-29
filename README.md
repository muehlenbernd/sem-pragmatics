# Social Evaluation Model (SEM)

Python implementation of the **Social Evaluation Model (SEM)**, a formal
framework for understanding how pragmatic reasoning shapes listeners' social
judgments of speakers.

> **Paper:** "Modeling Pragmatic Reasoning behind Social Meaning"  
> *Manuscript under review*

---

## What is SEM?

When we hear someone speak, we don't just decode what they said — we also
form impressions of *who they are*. A speaker who gives an unnecessarily
vague answer might seem less competent. One who answers with excessive
precision might seem pedantic. SEM formalizes how these social evaluations
arise from pragmatic inference.

The core evaluation function (Eq. 4 in the paper) is:

$$E_L(a \mid u, c) = \sum_{k \in K} \sum_{m \in M} P(k \mid u) \cdot P(m \mid k, u, c) \cdot I(a \mid k, m;\, \omega)$$

where:
- $u$ = utterance (written `v` in the code), $c$ = situational context, $a$ = social attribute
- $k$ = inferred speaker knowledge state, $m$ = inferred motivation strategy
- $\omega \in (0,1)$ = weight for knowledge vs. motivation
- Output $E_L \in [-1, 1]$

The three components each correspond to one equation in the paper:
- **Eq. 1** $P(k \mid u)$: knowledge-state posterior (staged or joint inference)
- **Eq. 2** $P(m \mid k, u, c)$: strategy posterior
- **Eq. 3** $I(a \mid k, m;\, \omega)$: impact function

The model is applied to Case Study 1: **numerical (im)precision** — how
precise vs. approximate expressions affect ratings of competence, likeability,
and pedantry.

---

## Repository Structure

```
sem-pragmatics/
├── sem/                          # Core SEM package
│   ├── model.py                  # SEMScenario class (Equations 1–4)
│   ├── fit.py                    # Fit metrics (DAS, ISS, ESR, CDS, RMSE, CCC)
│   ├── comparison.py             # Variant comparison and parameter sampling
│   ├── impact_derivation.py      # φ tables from Task 2 motivation data
│   └── plotting.py               # Figure generation
├── case_studies/                 # Scenario configurations
│   ├── cs1_imprecision.py        # Case Study 1: precise vs. approximate
│   └── cs2_pragmatic_violations.py
├── scripts/                      # Runnable analyses (run from repo root)
│   ├── run_comparison.py         # Reproduce variant-comparison results
│   ├── joint_listener.py         # Joint-listener comparison
│   ├── within_condition_check.py # Within-condition robustness check
│   ├── compute_empirical_means.py
│   ├── exp2_knowledge_test.py    # Exp. 2 held-out test (D1)
│   └── dev/                      # Helpers that rebuild notebooks
├── data/                         # Experimental data
│   ├── README.md                 # Data description
│   └── cs1_experiment/           # Data for Case Study 1
├── docs/                         # Design notes
│   ├── staged_vs_joint.md        # Staged vs. joint listener
│   └── impact_mapping_rationale.md
├── notebooks/                    # Interactive examples
│   └── sem_demo.ipynb
├── tests/                        # Pytest suite (57 tests, 2 slow)
├── figures/                      # Generated figures (gitignored)
├── requirements.txt
└── README.md                     # This file
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, numpy, pandas, matplotlib, scipy, pytest

---

## Quick Start

### Run the balanced model (ω = 0.5, uniform priors)

```python
from case_studies.cs1_imprecision import build_scenario, target_effects

# Build the Case Study 1 scenario
scenario = build_scenario(omega=0.5)

# Compute all evaluation scores
scores = scenario.evaluate_all()

# Score for 'competent' given precise utterance in high-precision context
print(scores["a_comp"]["v_prc"]["c_HP"])  # → 0.75

# Check which empirical effects the model predicts correctly
effects = target_effects(scores)
print(effects)  # → [True, True, True, True]
```

### Use custom parameters

```python
scenario = build_scenario(
    omega=0.7,
    priors_K={"k_p": 0.7, "k_np": 0.3},
    priors_M={"m_max": 0.5, "m_sit": 0.4, "m_min": 0.1},
)
scores = scenario.evaluate_all()
```

### Run the robustness test

```python
from case_studies.cs1_imprecision import build_scenario, target_effects

scenario = build_scenario()
result = scenario.robustness_test(effects_fn=target_effects, step=0.05)
print(f"All effects correct: {result['rate_all']*100:.1f}% of parameter combinations")
```

---

## Staged and Joint Listener

The default listener infers the speaker's knowledge state from the utterance
alone (`inference="staged"`). An alternative joint listener additionally
marginalizes over strategies when computing the knowledge posterior:

```python
staged = build_scenario(inference="staged")   # default
joint  = build_scenario(inference="joint")

staged.prob_k("k_p", "v_apx", "c_LP")  # 0.50 (context ignored)
joint.prob_k( "k_p", "v_apx", "c_LP")  # 0.40 (context shifts inference)
```

Both listeners reproduce all four target effects of Case Study 1. See
`docs/staged_vs_joint.md` for the formulae, flat-prior examples, and a
discussion of where the paper addresses the two options (Section 5.5).

---

## Running Tests

```bash
PYTHONPATH=. python -m pytest -q          # 57 tests, 2 slow skipped
PYTHONPATH=. python -m pytest -q --slow   # includes the .02 parameter grid
```

The slow tests sweep 2.6 million parameter settings each and take 30–60 minutes.

---

## Experimental Data

`data/cs1_experiment/processed/` contains:

- `experiment1.csv` — N = 362 participants (base knowledge condition)
- `experiment2.csv` — N = 390 participants (established knowledge condition)
- Pre-computed means and derived JSON files

See `data/README.md` for a full description. The processed data are included
here. The full materials, raw data and preregistration are archived publicly;
the link will be added after peer review.

---

## Use of Generative AI

The model was first implemented without AI assistance. Claude (Anthropic) was
used to revise and reorganize the code and to set up the repository. The
authors checked all code, analyses and results and take full responsibility
for them.

---

## License

MIT License — see `LICENSE` file.
