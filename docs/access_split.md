# Access-split listener

The access-split listener extends SEM by separating the speaker's knowledge
into two variables that play different roles in the model.

## Two knowledge variables

- **access** s ∈ {s_p, s_¬p}: what the speaker has at hand when speaking.
  Drives the truthfulness function t(s) and the strategy indicators χ_m(c, s, u).
- **prior knowledge** h ∈ {h_p, h_¬p}: whether the speaker knew the exact value
  beforehand ("by heart"). Drives the impact function φ_K(h, a).

Constraint: h_p implies s_p (you cannot know something without having access to it).

## Experiment 1 reduction

When the source is left open (Exp. 1), the two variables coincide: s = h.
The model reduces exactly to vanilla SEM with P(k_p) = P(h_p).

## Experiment 2 setting

When the speaker is shown consulting a precise source (Exp. 2), every speaker
has access: s = s_p.  The prior knowledge h retains its own prior π = P(h_p).

Because the truthfulness function and strategies depend on s (not h), and s is
fixed to s_p, the knowledge stage gives P(h | u) = π (unchanged by the
utterance).  The evaluation simplifies to:

    E_L(a | u, c) = Σ_h P(h) · Σ_m P(m | s_p, u, c) · I(a | h, m; ω)

Note: the staged and joint listeners are equivalent in Exp. 2, because the
joint update P(h | u, c) is also π (h does not appear in χ_t or χ_m).

## Implementation

```python
from case_studies.cs1_imprecision import build_scenario
from sem.access_split import AccessSplitScenario

base  = build_scenario()          # balanced SEM, staged listener
split = AccessSplitScenario(base, pi=0.5)

e1 = split.evaluate_all_exp1()    # Exp. 1: s = h, P(k_p) = 0.5
e2 = split.evaluate_all_exp2()    # Exp. 2: s = s_p, P(h_p) = 0.5
```

The `pi` parameter can be supplied at construction time or overridden
per call: `split.evaluate_all_exp2(pi=0.9)`.

## Consistent-prior baseline (Step 4d)

The original Step 4 analysis held Exp. 1 at P(k_p) = 0.5 while varying π in
Exp. 2.  Step 4d uses one listener per π: Exp. 1 = vanilla SEM with P(k_p) = π
(same prior as Exp. 2).  This makes the comparison consistent.

Because v_prc forces the inference P(k_p | v_prc) = 1 whenever π > 0, the shift
Δ_pred(π) = α_i · (1 − π) for every cell, where the α_i are given analytically.
The least-squares solution is π* = 1 − ⟨obs, α⟩ / ⟨α, α⟩.

Two models are compared:
- **(i) uniform split**: φ_K on h for all attributes (α_comp ≠ 0 for v_prc cells).
- **(ii) attribute-specific**: comp uses φ_K on s (access = s_p in Exp. 2) → α_comp = 0
  for v_prc, positive for v_apx; ped/like use φ_K on h.

## Where this analysis appears

`scripts/exp2_access_split.py` runs the full analysis (Steps 1, 4, 4d, 5 from
the Session G brief), comparing uniform split (4a), attribute-specific split
(4b), consistent-prior sweep (4d), and least-squares π against Experiment 2 data.
