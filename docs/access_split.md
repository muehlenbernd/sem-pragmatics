# Extended model: three knowledge states

The paper's §4.4 analysis introduces an *extended model* with three knowledge
states that replaces the two-state {k_p, k_¬p} baseline.

## Three states

- **k_mem** — the speaker knew the exact value beforehand (from memory).
- **k_src** — the speaker reads the value off a consulted source at speech time.
- **k_none** — the speaker has no access; only the approximate form is truthful.

Constraint: k_mem and k_src both allow the precise and approximate forms;
k_none allows only the approximate form. The truthfulness function χ_t and
strategy indicators χ_m depend on access (k_mem/k_src have access, k_none does
not), not on the epistemically finer distinction between k_mem and k_src.

## Impact of k_src

§3.3's derivation procedure assigns φ_K by attribute:

| attribute   | k_mem | k_src | k_none |
|-------------|-------|-------|--------|
| competence  | +1    | +1    | −1     |
| likeability |  0    |  0    |  0     |
| pedantry    | −1    | +1    | +1     |

k_src counts as k_p (access ↔ competence) but as k_¬p (not self-knowledge ↔
pedantry). This is set by argument, not by fitting. See §4.4 for the
justification; Step 2b of `scripts/exp2_section44.py` reports what the §3.3
test yields empirically.

## Experiment 1 and 2 as shifts of the same prior π

One prior π = P(k_mem) is shared across both experiments.

- **Exp. 1** (source unobserved): listeners give k_src no weight; mass is split
  k_mem (π) + k_none (1 − π). This reduces exactly to vanilla SEM with
  P(k_p) = π.
- **Exp. 2** (speaker seen consulting a source): the manipulation moves k_none's
  mass to k_src — k_mem (π) + k_src (1 − π). The same π in both experiments
  makes the comparison consistent.

## Which variant the paper uses

The paper fits π to the four pedantry cells (π* ≈ .70) and uses the
*attribute-specific* split: competence applies φ_K to access (k_mem and k_src
both count as k_p), pedantry and likeability apply φ_K to self-knowledge
(k_src counts as k_¬p). The uniform split (same impact table for all attributes)
is kept in the code as a diagnostic but is not used in the paper's main analysis.

## Code mapping

Code names: access variable `s`, prior-knowledge variable `h`, prior `π` (pi).
`AccessSplitScenario` in `sem/access_split.py`; called from
`scripts/exp2_access_split.py` (full sweep and fit) and
`scripts/exp2_section44.py` (number reproduction).

```python
from case_studies.cs1_imprecision import build_scenario
from sem.access_split import AccessSplitScenario

split = AccessSplitScenario(build_scenario(), pi=0.698)
e1 = split.evaluate_all_exp1()   # Exp. 1: vanilla SEM with P(k_p) = 0.698
e2 = split.evaluate_all_exp2()   # Exp. 2: k_src shift, same π
```
