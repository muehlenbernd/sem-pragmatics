# Staged and joint listener

SEM's listener evaluates a speaker in two stages: first it infers the speaker's
knowledge state from the utterance, then it infers the motivation, and finally it
combines the two via the impact function. The knowledge-state inference step admits
two formulations, controlled by `SEMScenario(inference=...)`.

## Staged listener (default)

The staged listener uses only the truthfulness constraint to update beliefs about
the speaker's knowledge state:

    P(k | u) ∝ P(k) · χ_t(k, u)

where χ_t(k, u) = 1 if utterance u is truthful under knowledge state k, else 0.
The context c does not enter this stage. The strategy distribution is then inferred
conditional on the inferred knowledge state (Eq. 2 in the paper):

    P(m | k, u, c) ∝ P(m) · χ_m(c, k, u)

This is the listener described throughout the paper and used for all reported
results.

## Joint listener

The joint listener marginalizes over strategies when computing the knowledge
posterior, so the context c also shifts the knowledge inference:

    P(k | u, c) ∝ P(k) · Σ_m P(m) · χ_m(c, k, u)

The strategy posterior P(m | k, u, c) is unchanged. Because every strategy
selects only truthful utterances (χ_m implies χ_t), the explicit truthfulness
factor is redundant and is omitted.

## Flat-prior example

With uniform priors P(k_p) = P(k_np) = 0.5 and P(m_max) = P(m_sit) = P(m_min) = 1/3:

| Listener | u       | c    | P(k_p \| u, c) |
|----------|---------|------|----------------|
| staged   | v_apx   | any  | 0.50           |
| joint    | v_apx   | c_LP | 0.40           |
| joint    | v_apx   | c_HP | 0.25           |
| both     | v_prc   | any  | 1.00           |

Under c_LP, m_sit also produces v_apx (precision not required), so there are more
ways for k_p to produce v_apx than under c_HP, and P(k_p) rises relative to c_HP.

With a maximally-precise-speaker prior P(m_max) = 0.8, P(m_sit) = P(m_min) = 0.1:
the joint listener gives P(k_p | v_apx, c_LP) = 1/6 ≈ 0.167, because m_max would
rarely produce v_apx given k_p.

## Empirical results

Both listeners reproduce all four target effects of Case Study 1:

- Balanced model (ω = 0.5, flat priors): DAS = ISS = 1.00 for both.
- 100 sampled (ω, P(k), P(m)) combinations (seed 42): staged 100/100, joint 99/100.
- Robustness sweep (step .05, 61,731 settings): staged 99.9 %, joint 98.8 %.
- Robustness sweep (step .02, 2,593,080 settings): staged 99.87 %, joint 98.86 %.

The joint listener shows slightly more main-effect overshoot (higher CDS_main) and
slightly better interaction calibration (lower CDS_interaction). The data do not
decide between the two; the staged listener is the paper's primary analysis.

## Where the paper discusses it

Section 5.5 discusses the staged listener as SEM's default design choice and
includes a footnote comparing the staged and joint posteriors, with the flat-prior
example above. The joint listener is presented as a possible alternative that
preserves all qualitative predictions.

## Usage

```python
from case_studies.cs1_imprecision import build_scenario

staged = build_scenario(inference="staged")   # default
joint  = build_scenario(inference="joint")

# Knowledge posterior
staged.prob_k("k_p", "v_apx", "c_LP")  # 0.5
joint.prob_k( "k_p", "v_apx", "c_LP")  # 0.4

# Full evaluation is automatic
joint.evaluate_all()
joint.robustness_test(target_effects, step=0.05)
```
