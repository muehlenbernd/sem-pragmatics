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

The core evaluation function is:

$$E_L(a \mid v, c) = \sum_{k \in K} \sum_{m \in M} P(k \mid v) \cdot P(m \mid k, v, c) \cdot I(a \mid k, m;\, \omega)$$

where:
- $v$ = utterance, $c$ = situational context, $a$ = social attribute
- $k$ = inferred speaker knowledge state, $m$ = inferred motivation strategy
- $\omega \in (0,1)$ = weight for knowledge vs. motivation
- Output $E_L \in [-1, 1]$

The model is applied to two case studies:

1. **Numerical (im)precision** — how precise vs. approximate expressions
   affect ratings of competence, likeability, and pedantry
2. **Pragmatic norm violations** — how violations of relevance and
   informativeness affect social evaluations

---

## Repository Structure

```
sem-model/
├── sem/                          # Core SEM package
│   ├── model.py                  # SEMScenario class (Equations 1–4)
│   └── plotting.py               # Figure generation
├── case_studies/                 # Scenario configurations
│   ├── cs1_imprecision.py        # Case Study 1: precise vs. approximate
│   └── cs2_pragmatic_violations.py  # Case Study 2: relevance & informativeness
├── baselines/                    # Comparison models
│   └── rsa_baseline.py           # RSA-extended baseline (in progress)
├── notebooks/                    # Interactive examples
│   └── sem_demo.ipynb            # Demo: run SEM with custom parameters
├── data/                         # Experimental data
│   ├── README.md                 # Data description and access info
│   └── cs1_experiment/           # Raw and processed data for Case Study 1
├── tests/                        # Pytest suite
│   └── test_case_studies.py      # 19 tests covering both case studies
├── figures/                      # Generated figures (output directory)
├── requirements.txt              # Python dependencies
├── PLAN.md                       # Internal development plan
└── README.md                     # This file
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/[username]/sem-model.git
cd sem-model

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, numpy, matplotlib, pytest

Optional: LaTeX installation for publication-quality figures
(falls back to default matplotlib fonts if unavailable).

---

## Quick Start

### Run the balanced model (ω = 0.5, uniform priors)

```python
from case_studies.cs1_imprecision import build_scenario, target_effects

# Build the Case Study 1 scenario
scenario = build_scenario(omega=0.5)

# Compute all evaluation scores
scores = scenario.evaluate_all()

# Print score for 'competent' given precise utterance in high-precision context
print(scores["a_comp"]["v_prc"]["c_HP"])  # → 0.75

# Check which empirical effects the model predicts correctly
effects = target_effects(scores)
print(effects)  # → [True, True, True, True]
```

### Use custom parameters

```python
scenario = build_scenario(
    omega=0.7,                          # weight knowledge more heavily
    priors_K={"k_p": 0.7, "k_np": 0.3},  # speaker more likely to know exact value
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

### Interactive demo

Open `notebooks/sem_demo.ipynb` in Jupyter for a step-by-step walkthrough
of both case studies with editable parameters.

---

## Reproducing Paper Figures

```bash
# Case Study 1 — balanced model predictions (Figure 6 in paper)
PYTHONPATH=. python case_studies/cs1_imprecision.py

# Case Study 2 — balanced model predictions (Figure 8 in paper)
PYTHONPATH=. python case_studies/cs2_pragmatic_violations.py
```

Figures are saved to the `figures/` directory.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 19 tests should pass. Tests cover:
- Correct prediction of all empirical effects (both case studies)
- Knowledge state inference (Equation 2)
- Strategy inference (Equation 3)
- Score range validity (outputs ∈ [−1, 1])

---

## Experimental Data

The `data/` folder contains the experimental data from the matched-guise
study reported in Case Study 1. See `data/README.md` for details on the
data structure, variable coding, and access conditions.

The R analysis scripts used for the statistical analyses in the paper are
in `data/cs1_experiment/analysis/`.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{sem2025,
  author  = {[Authors]},
  title   = {Modeling Pragmatic Reasoning behind Social Meaning},
  journal = {Meaning: A Journal of Linguistics and Philosophy},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT License — see `LICENSE` file.

---

## Contact

For questions about the model or paper, please open a GitHub issue or
contact the corresponding author at [email].
