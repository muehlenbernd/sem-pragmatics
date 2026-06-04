# TASKS.md — Claude Code Task List

This file contains concrete, ordered tasks for Claude Code to execute on the
`sem-pragmatics` project. Work through tasks sequentially. Run the test suite
after every task that modifies Python files.

**Always run tests with:**
```bash
PYTHONPATH=. python -m pytest tests/ -v
```

All 19 existing tests must continue to pass after every task unless a task
explicitly states that impact matrix changes may alter predictions (Task 0,
step 4 handles this explicitly).

---

## Task 0 — Derive empirical impact matrices from data

### Background

The SEM's trivalent impact functions φ_K and φ_M encode how speaker knowledge
states and motivation strategies affect social attribute ratings. These values
(−1, 0, +1) should be derived empirically from the experimental data.

The derivation procedure is:
1. Treat each t2_ motivation checkbox as a proxy for a SEM variable (K or M)
2. For each proxy × raw attribute pair, run an independent-samples t-test
   (checked vs. not-checked participants)
3. Apply a significance threshold: if p < α and diff > 0 → +1; diff < 0 → −1;
   otherwise → 0
4. Map raw attributes directly to SEM attributes (one-to-one, no aggregation):
   - a_comp = competent  (direct)
   - a_like = likeable   (direct)
   - a_ped  = pedantic   (direct)
   Rationale: these are the three attributes the SEM explicitly models.
   knowledgeable, well_prepared, and helpful are not used, avoiding
   the theoretical assumption that they proxy the same construct.
5. Where multiple proxies map to one SEM variable, take majority vote across
   proxies

### Proxy-to-SEM mapping (paper version, use this exactly)

| SEM variable | t2 proxies | n (approx) |
|---|---|---|
| k_p  | knewExact, infoAvail | 38, 17 |
| k_np | notKnow, infoNotAvail | 94, 29 |
| m_max | fussyPerson | 29 |
| m_sit | needed, purpose, realWorld | 40, 23, 53 |
| m_min | easySpeaker, usualApx | 9, 20 |

### Step 0.1 — Create `sem/impact_derivation.py`

Create a new file `sem/impact_derivation.py` with the following functions.
This file should be fully self-contained and importable independently of the
rest of the SEM package — it will later be used in the demo notebook.

```
derive_trivalent_matrix(
    data_path: str,
    sem_mapping: dict,
    sem_attr_mapping: dict,
    alpha: float
) -> dict
```

- Loads the CSV from `data_path`
- For each SEM variable and each SEM attribute, computes the majority-vote
  trivalent value as described above
- Returns a nested dict: `result[sem_var][sem_attr]` → int in {-1, 0, +1}

```
derive_all_thresholds(
    data_path: str,
    sem_mapping: dict,
    sem_attr_mapping: dict,
    alphas: list = [0.001, 0.01, 0.05, 0.10]
) -> dict
```

- Calls `derive_trivalent_matrix` for each alpha
- Returns `result[alpha][sem_var][sem_attr]`

```
save_impact_tables(
    tables: dict,
    output_path: str
) -> None
```

- Saves the output of `derive_all_thresholds` as a JSON file to `output_path`
- JSON structure: `{"0.001": {...}, "0.01": {...}, "0.05": {...}, "0.10": {...}}`
- Each inner dict has structure `{sem_var: {sem_attr: value}}`

```
print_impact_table(
    matrix: dict,
    current_model: dict = None,
    title: str = ""
) -> None
```

- Pretty-prints a single threshold's matrix
- If `current_model` is provided, marks cells that differ with ■
- Columns: a_comp, a_like, a_ped
- Rows: k_p, k_np, m_max, m_sit, m_min

Use `scipy.stats.ttest_ind` for t-tests. The data file is at:
`data/cs1_experiment/processed/experiment1.csv`

Column names in the CSV:
- Social rating columns: `competent`, `well_prepared`, `knowledgeable`,
  `helpful`, `likeable`, `pedantic`  (integer, 1–7 Likert scale)
- Motivation columns: `t2_knewExact`, `t2_infoAvail`, `t2_notKnow`,
  `t2_infoNotAvail`, `t2_fussyPerson`, `t2_needed`, `t2_purpose`,
  `t2_realWorld`, `t2_easySpeaker`, `t2_usualApx`  (binary 0/1)
- Skip rows where t2 column sum < 5 (insufficient data)

The DEFAULT mappings to use (define as module-level constants so they can be
imported by the notebook):

```python
CS1_SEM_MAPPING = {
    "k_p":   ["knewExact", "infoAvail"],
    "k_np":  ["notKnow", "infoNotAvail"],
    "m_max": ["fussyPerson"],
    "m_sit": ["needed", "purpose", "realWorld"],
    "m_min": ["easySpeaker", "usualApx"],
}

CS1_SEM_ATTR_MAPPING = {
    "a_comp": ["competent"],
    "a_like": ["likeable"],
    "a_ped":  ["pedantic"],
}
```

### Step 0.2 — Run the derivation and save lookup tables

Create a script `scripts/derive_impact_tables.py` that:
1. Imports from `sem/impact_derivation.py`
2. Runs `derive_all_thresholds` on `data/cs1_experiment/processed/experiment1.csv`
3. Saves the result to `data/cs1_experiment/processed/impact_tables_cs1.json`
4. Prints all four threshold tables to stdout with the current model shown for
   comparison

Create the `scripts/` directory if it does not exist.
Add a `scripts/__init__.py` (empty).

Run the script and verify it produces output matching the expected p<.01 table:

```
k_p  : a_comp=+1, a_like= 0, a_ped= 0
k_np : a_comp=-1, a_like= 0, a_ped=-1
m_max: a_comp= 0, a_like= 0, a_ped=+1
m_sit: a_comp=+1, a_like=+1, a_ped= 0
m_min: a_comp=-1, a_like= 0, a_ped= 0
```

Note: adding infoAvail to k_p and infoNotAvail to k_np does not change
the expected table at p<.01. infoAvail shows only p<.05 signals (not
reaching p<.01) and infoNotAvail shows no significant effects at any
threshold — both are included for coverage of the full motive space and
for robustness checks at looser thresholds, but they do not alter the
p<.01 impact matrices.

If the output does not match this table exactly, stop and report the
discrepancy rather than proceeding.

### Step 0.3 — Update impact matrices in `cs1_imprecision.py`

Update `IMPACT_K` and `IMPACT_M` in
`case_studies/cs1_imprecision.py` to reflect the p<.01 derived values:

**Current → New:**
```python
# IMPACT_K
"k_p":  {"a_comp":  1, "a_like":  0, "a_ped":  0}   # unchanged
"k_np": {"a_comp": -1, "a_like": -1, "a_ped": -1}   # a_like: -1 → 0

# IMPACT_M
"m_max": {"a_comp":  0, "a_like": -1, "a_ped":  1}  # a_like: -1 → 0
"m_sit": {"a_comp":  1, "a_like":  1, "a_ped": -1}  # a_ped: -1 → 0
"m_min": {"a_comp": -1, "a_like": -1, "a_ped":  0}  # a_like: -1 → 0
```

**New values:**
```python
IMPACT_K: Dict[str, Dict[str, int]] = {
    "k_p":  {"a_comp":  1, "a_like":  0, "a_ped":  0},  # unchanged
    "k_np": {"a_comp": -1, "a_like":  0, "a_ped": -1},  # a_like: -1 → 0
}

IMPACT_M: Dict[str, Dict[str, int]] = {
    "m_max": {"a_comp":  0, "a_like":  0, "a_ped":  1},  # a_like: -1 → 0
    "m_sit": {"a_comp":  1, "a_like":  1, "a_ped":  0},  # a_ped: -1 → 0
    "m_min": {"a_comp": -1, "a_like":  0, "a_ped":  0},  # a_like: -1 → 0
}
```

Also update the docstring in `cs1_imprecision.py` under
"Trivalent impact functions" to reflect the new values and add a note:

```
Trivalent impact functions — derived from Experiment 1 data at p < .01
(see data/cs1_experiment/processed/impact_tables_cs1.json for all thresholds)
```

### Step 0.4 — Check and update tests

Run the full test suite:
```bash
PYTHONPATH=. python -m pytest tests/ -v
```

Some tests in `tests/test_case_studies.py` for CS1 may now fail because the
impact matrices have changed. For each failing test:

1. Recompute the expected value analytically or by running the model
2. If the effect direction is unchanged (the empirical effect still holds with
   the new matrices), update the test to match the new expected value
3. If an effect is no longer predicted by the model (direction reversed or
   disappeared), DO NOT silently update the test — instead add a comment
   `# NOTE: effect weakened after p<.01 impact matrix update` and report
   which effects changed in a summary at the end

Specifically re-verify these four target effects with the new matrices:
- Effect 1: PRECISE > APPROX on a_comp (across contexts)
- Effect 2: PRECISE > APPROX on a_ped (across contexts)
- Effect 3: a_comp gap (prc−apx) larger in c_HP than c_LP
- Effect 4: a_like gap (prc−apx) larger in c_HP than c_LP

Run `PYTHONPATH=. python case_studies/cs1_imprecision.py` and confirm all
four effects are printed as ✓. Report the new balanced-model prediction
scores in a comment at the top of the updated test.

### Step 0.5 — Add a new test class for the derivation

Add a new test class `TestImpactDerivation` to `tests/test_case_studies.py`:

```python
class TestImpactDerivation:
    """Verify that derive_trivalent_matrix reproduces the expected
    p<.01 impact matrices from the CS1 data."""

    def test_p001_derivation_runs(self):
        # Just check it runs without error
        ...

    def test_p01_matches_cs1_impact_matrices(self):
        # Derive at p<.01 and check each cell matches IMPACT_K and IMPACT_M
        # Skip if data file not present (CI environment)
        ...

    def test_all_thresholds_saved(self):
        # Check that impact_tables_cs1.json exists and contains all 4 thresholds
        ...
```

Use `pytest.importorskip` or a file-existence check to skip gracefully if the
data CSV is not present (for CI environments where data is not committed).

---

## Task 1 — Implement `sem/fit.py` and empirical means pipeline

### Background

The SEM currently reports only whether it predicts the correct direction of
each significant empirical effect (qualitative check). Task 1 adds quantitative
model evaluation by comparing SEM predictions to empirical human means using
a metric suite from Mühlenbernd (2026, CMCL). This enables direct reporting of
how well SEM is calibrated, not just directionally correct.

**Key design decision:** Empirical means are pre-computed from the CS1 data
and stored as a JSON lookup file. `sem/fit.py` then works only with these
pre-computed means (not the raw CSV), keeping the fit module self-contained and
fast. The means computation is a one-time preprocessing step.

**Metric scope:** CS1 only (CS2 has no accessible empirical data).

**Which effects are evaluated:**

The CS1 experiment has 3 SEM-modeled attributes: a_comp (competent),
a_like (likeable), a_ped (pedantic). For each, we evaluate:
- 1 main effect of form: precise vs. approximate (aggregated across contexts)
- 1 form × context interaction: does the precise−approx gap differ between c_HP
  and c_LP?

This gives 6 effects total (3 main + 3 interaction). Of these, 4 are
statistically significant in the human data (p < .001):
- Significant main effects: a_comp, a_ped
- Significant interactions: a_comp, a_like
- Non-significant (near-zero ΔH): main effect of a_like, interaction of a_ped

**Metric assignment by effect scope:**

- DAS, ISS, ESR, CDS → computed over the 4 SIGNIFICANT effects only (DAS over
  the 2 significant main effects {a_comp, a_ped}; ISS over the 2 significant
  interactions {a_comp, a_like}). Testing directional agreement on near-zero ΔH
  would be scoring noise; consistent with Mühlenbernd 2026.
- Spearman ρ, RMSE, CCC → computed over ALL 12 condition cells (3 attributes ×
  2 forms × 2 contexts) for a global calibration picture

---

### Step 1.1 — Compute and save empirical means

Create `scripts/compute_empirical_means.py` that reads from the CS1 processed
data and writes a JSON lookup file for use by `sem/fit.py`.

**Input:** `data/cs1_experiment/processed/experiment1.csv`

**Relevant columns:**
- `form`: `"precise"` or `"approximate"` (utterance form)
- `context`: `"HP"` or `"LP"` (high/low precision context)
- `competent`, `likeable`, `pedantic`: integer Likert ratings 1–7

**Output:** `data/cs1_experiment/processed/empirical_means_cs1.json`

**JSON structure:**
```json
{
  "a_comp": {
    "v_prc": {"c_HP": 5.21, "c_LP": 4.87},
    "v_apx": {"c_HP": 4.10, "c_LP": 4.35}
  },
  "a_like": {
    "v_prc": {"c_HP": 4.12, "c_LP": 4.05},
    "v_apx": {"c_HP": 4.20, "c_LP": 4.31}
  },
  "a_ped": {
    "v_prc": {"c_HP": 3.98, "c_LP": 4.10},
    "v_apx": {"c_HP": 3.21, "c_LP": 3.30}
  },
  "_meta": {
    "scale": "7-point Likert (1-7, raw, not normalized)",
    "source": "data/cs1_experiment/processed/experiment1.csv",
    "n_total": 362,
    "n_per_condition": {"v_prc_c_HP": 90, "v_prc_c_LP": 91, "v_apx_c_HP": 89, "v_apx_c_LP": 92}
  }
}
```

Notes:
- Store raw 1–7 means (not normalized). Normalization to [−1, 1] happens inside
  `sem/fit.py` at computation time so the JSON is human-readable.
- The `_meta` key stores provenance information; include actual n values from
  the data.
- Column name mapping: `form` values `"precise"` → `v_prc`, `"approximate"` →
  `v_apx`; `context` values `"HP"` → `c_HP`, `"LP"` → `c_LP`. Verify these
  column values by inspecting the CSV before assuming them.
- Print a summary table to stdout showing the 12 computed means (3 attrs × 2
  forms × 2 contexts) so the output can be manually sanity-checked.

---

### Step 1.2 — Create `sem/fit.py`

Create `sem/fit.py`. This module is the quantitative evaluation engine for SEM.
It must be fully self-contained and importable independently of the rest of the
SEM package.

**Module-level docstring** should explain: this module computes quantitative
fit metrics comparing SEM predictions (in [−1, 1]) to empirical human means
(raw 1–7 Likert, loaded from JSON and normalized at call time). Metrics follow
Mühlenbernd (2026, CMCL).

---

#### Function: `normalize_likert`

```python
def normalize_likert(value: float, scale_min: int = 1, scale_max: int = 7) -> float:
```

Maps a raw Likert mean to [−1, 1] using the formula:
```
normalized = 2 * (value - scale_min) / (scale_max - scale_min) - 1
```

For a 1–7 scale: `normalized = (value - 4) / 3`

This is the same transformation used by the SEM's evaluation function output
range, making model and human values directly comparable.

---

#### Function: `load_empirical_means`

```python
def load_empirical_means(
    json_path: str,
    attributes: list = None
) -> dict:
```

- Loads the JSON produced by Step 1.1
- Normalizes all means from 1–7 to [−1, 1] using `normalize_likert`
- Returns a dict with the same structure as the JSON (minus `_meta`), but with
  normalized float values
- If `attributes` is provided (e.g. `["a_comp", "a_like"]`), filters to those
  keys only
- `_meta` is preserved under the same key for reference

---

#### Function: `extract_effects`

```python
def extract_effects(
    means: dict,
    attributes: list = None
) -> dict:
```

Given a means dict (structure: `{attr: {utterance: {context: score}}}`),
computes the derived quantities needed for all metrics.

For each attribute `a` in `means` (or `attributes` if filtered):
- `delta_HP[a]` = means[a]["v_prc"]["c_HP"] − means[a]["v_apx"]["c_HP"]
- `delta_LP[a]` = means[a]["v_prc"]["c_LP"] − means[a]["v_apx"]["c_LP"]
- `main_effect[a]` = (delta_HP[a] + delta_LP[a]) / 2  (main effect of form)
- `interaction[a]` = delta_HP[a] − delta_LP[a]  (form × context interaction)

Returns a dict:
```python
{
  "a_comp": {
    "delta_HP": ..., "delta_LP": ...,
    "main_effect": ..., "interaction": ...
  },
  ...
}
```

This function is used by all metric functions below. It works identically on
human means and model scores (both are in [−1, 1] after normalization).

---

#### Function: `das`

```python
def das(
    human_effects: dict,
    model_effects: dict,
    attributes: list = None
) -> float:
```

**Directional Agreement Score** for main effects.

For each attribute `a`: check whether `sign(model_effects[a]["main_effect"])`
equals `sign(human_effects[a]["main_effect"])`.

- Score = (number of matching signs) / (number of attributes)
- Range: [0, 1]; 1 = perfect directional agreement on all main effects
- Skip attributes where `human_effects[a]["main_effect"] == 0` (undefined sign)
- `attributes` parameter filters which attributes to include

---

#### Function: `iss`

```python
def iss(
    human_effects: dict,
    model_effects: dict,
    attributes: list = None
) -> float:
```

**Interaction Sensitivity Score** for form × context interactions.

Identical logic to `das` but applied to `"interaction"` instead of
`"main_effect"`.

- Score = (number of matching signs on interaction) / (number of attributes)
- Range: [0, 1]; 1 = perfect directional agreement on all interactions

---

#### Function: `esr`

```python
def esr(
    human_effects: dict,
    model_effects: dict,
    sig_main: list,
    sig_interaction: list
) -> dict:
```

**Effect Size Ratio** per significant effect.

`sig_main` and `sig_interaction` are lists of attribute names for which the
human effect is statistically significant (and thus has non-negligible |ΔH|).

For each `a` in `sig_main`:
```
ESR_main[a] = |model_effects[a]["main_effect"]| / |human_effects[a]["main_effect"]|
```

For each `a` in `sig_interaction`:
```
ESR_interaction[a] = |model_effects[a]["interaction"]| / |human_effects[a]["interaction"]|
```

Returns:
```python
{
  "main": {"a_comp": 0.82, "a_ped": 1.10},
  "interaction": {"a_comp": 0.65, "a_like": 0.91}
}
```

Raise `ValueError` if any |ΔH| == 0 for a listed significant attribute (this
would indicate a data error, not a near-zero non-significant effect).

---

#### Function: `cds`

```python
def cds(
    esr_values: dict
) -> dict:
```

**Calibration Deviation Score.**

Takes the output of `esr()` directly.

```
CDS_main        = mean(|ESR_main[a] - 1| for a in sig_main)
CDS_interaction = mean(|ESR_interaction[a] - 1| for a in sig_interaction)
CDS             = mean(CDS_main, CDS_interaction)
```

Returns:
```python
{
  "cds_main": 0.14,
  "cds_interaction": 0.22,
  "cds": 0.18
}
```

Lower is better; 0 = perfect magnitude calibration.

---

#### Function: `global_metrics`

```python
def global_metrics(
    human_means: dict,
    model_scores: dict,
    attributes: list = None
) -> dict:
```

Computes Spearman ρ, RMSE, and CCC across all conditions for the specified
attributes. These metrics operate on the flat vector of all (attr, utterance,
context) pairs — 12 values total for 3 attributes × 2 utterances × 2 contexts.

- **Spearman ρ**: rank correlation between human and model vectors
  (`scipy.stats.spearmanr`)
- **RMSE**: `sqrt(mean((H_i - M_i)^2))`
- **CCC** (Concordance Correlation Coefficient):
  ```
  CCC = 2 * cov(H, M) / (var(H) + var(M) + (mean(H) - mean(M))^2)
  ```
  where cov, var, mean are computed over the flat vector. Implement directly
  (do not use an external CCC library).

Returns:
```python
{
  "spearman_r": 0.94,
  "rmse": 0.18,
  "ccc": 0.87
}
```

---

#### Function: `evaluate`

```python
def evaluate(
    model_scores: dict,
    empirical_json_path: str,
    sig_main: list = None,
    sig_interaction: list = None,
    attributes: list = None
) -> dict:
```

**Top-level evaluation runner.** This is the primary entry point for computing
all metrics in one call.

Default values (CS1 case):
```python
sig_main        = ["a_comp", "a_ped"]
sig_interaction = ["a_comp", "a_like"]
attributes      = ["a_comp", "a_like", "a_ped"]
```

Steps:
1. Load and normalize empirical means from `empirical_json_path`
2. Extract effects for both human means and model scores
3. Compute DAS (2 significant main effects), ISS (2 significant interactions)
4. Compute ESR (sig attributes only)
5. Compute CDS from ESR
6. Compute global metrics (Spearman ρ, RMSE, CCC) across all 12 values
7. Return a flat dict:

```python
{
  "das": 1.0,
  "iss": 1.0,
  "esr_main": {"a_comp": 0.82, "a_ped": 1.10},
  "esr_interaction": {"a_comp": 0.65, "a_like": 0.91},
  "cds_main": 0.14,
  "cds_interaction": 0.22,
  "cds": 0.18,
  "spearman_r": 0.94,
  "rmse": 0.18,
  "ccc": 0.87
}
```

`model_scores` has the same structure as the empirical means dict
(`{attr: {utterance: {context: score}}}`), but values are already in [−1, 1]
(SEM output). No normalization is applied to model scores.

---

### Step 1.3 — Add runner to `cs1_imprecision.py`

At the bottom of `case_studies/cs1_imprecision.py`, after the existing effect
checks, add a `run_fit_evaluation()` function:

```python
def run_fit_evaluation(omega=0.5, priors_K=None, priors_M=None,
                       empirical_json_path=None):
    """Run quantitative fit evaluation for CS1 balanced model."""
```

Default `empirical_json_path`:
`"data/cs1_experiment/processed/empirical_means_cs1.json"`

The function should:
1. Build the scenario with given parameters
2. Call `scenario.evaluate_all()` to get model scores
3. Call `sem.fit.evaluate(model_scores, empirical_json_path)`
4. Print a formatted summary table to stdout, e.g.:

```
=== CS1 Fit Evaluation (balanced model: ω=0.5, flat priors) ===

Global metrics:
  Spearman ρ : 0.94
  RMSE       : 0.18
  CCC        : 0.87

Structural alignment:
  DAS (main effects)    : 1.00  (2/2)
  ISS (interactions)    : 1.00  (2/2)

Magnitude calibration (significant effects only):
  ESR main effects:
    a_comp : 0.82
    a_ped  : 1.10
  ESR interactions:
    a_comp : 0.65
    a_like : 0.91
  CDS (main)        : 0.14
  CDS (interaction) : 0.22
  CDS (overall)     : 0.18
```

5. Return the result dict from `evaluate()` so it can be used
   programmatically.

Update the `if __name__ == "__main__":` block at the bottom of
`cs1_imprecision.py` to also call `run_fit_evaluation()` after the existing
effect checks.

---

### Step 1.4 — Add tests

Add a new test class `TestFitMetrics` to `tests/test_case_studies.py`:

```python
class TestFitMetrics:
    """Verify sem/fit.py metric functions on known inputs."""
```

Include at minimum:

```python
def test_normalize_likert_midpoint(self):
    # normalize_likert(4.0) == 0.0

def test_normalize_likert_endpoints(self):
    # normalize_likert(1.0) == -1.0
    # normalize_likert(7.0) == 1.0

def test_das_perfect(self):
    # Both human and model have same-sign main effects → DAS = 1.0

def test_das_all_wrong(self):
    # All signs flipped → DAS = 0.0

def test_iss_partial(self):
    # 2 of 3 interactions correct → ISS = 0.67 (approx)

def test_esr_perfect_calibration(self):
    # Model effects equal human effects → all ESR = 1.0

def test_esr_double_magnitude(self):
    # Model effect twice as large → ESR = 2.0

def test_cds_zero(self):
    # All ESR = 1.0 → CDS = 0.0

def test_global_metrics_identical(self):
    # H == M → spearman_r = 1.0, rmse = 0.0, ccc = 1.0

def test_evaluate_runs_on_cs1(self):
    # Run evaluate() on actual CS1 balanced model output
    # Skip if empirical_means_cs1.json not present
    # Check: result["das"] in [0, 1], result["cds"] >= 0, etc.
```

Use `pytest.mark.skipif` or a file-existence check to skip
`test_evaluate_runs_on_cs1` gracefully if the JSON is not present.

All 19 existing tests must continue to pass.

---

### Completion check

After all steps:

1. Run `PYTHONPATH=. python scripts/compute_empirical_means.py` — verify JSON
   is created and printed means look plausible (precise > approximate on
   competent and pedantic; small differences on likeable)
2. Run `PYTHONPATH=. python case_studies/cs1_imprecision.py` — verify effect
   checks print ✓ for all 4 effects AND the fit table is printed with
   sensible values
3. Run `PYTHONPATH=. python -m pytest tests/ -v` — all tests pass (19 original
   + new TestFitMetrics tests)

Report: `TASK 1 COMPLETE: sem/fit.py implemented, empirical means saved,
runner added to cs1_imprecision.py, N tests passing`

---

## Task 2 — SEM variant comparison via parameter sampling

### Background

Task 1 established that the balanced SEM (ω=0.5, flat priors) reproduces the
qualitative structure of the CS1 empirical effects but overshoots effect
magnitudes (CDS=0.75). A key reviewer concern is whether the full SEM's formal
inference machinery adds explanatory value beyond simpler alternatives. Task 2
addresses this by comparing four nested SEM variants — the full model and three
ablations that remove specific inferential components — using a parameter
sampling procedure.

Rather than reporting metrics for a single parameter setting (which is
sensitive to the choice), each variant is evaluated across 100 sampled
parameter combinations, yielding a distribution over metrics per variant. This
makes the comparison robust to parameter choice and directly addresses the
CDS sensitivity concern: if a variant consistently scores worse across the
full sample, the difference is structural, not an artifact of a specific
parameter setting.

**This task does NOT implement a separate baseline model.** All four variants
use the existing SEMScenario infrastructure with constrained parameters.

---

### The four SEM variants

| Variant | Constraint | What it removes |
|---|---|---|
| SEM-full | none — all parameters free | — |
| SEM-context-blind | P(m_sit) fixed at 0 | context-sensitive motivation inference |
| SEM-knowledge-only | ω fixed at 1.0 | motivation inference entirely |
| SEM-motivation-only | ω fixed at 0.0 | knowledge state inference entirely |

**Rationale for each ablation:**
- **SEM-context-blind**: operationalizes the reviewer's "violation of
  expectation" account — the model can still infer knowledge states and
  non-contextual motivations, but loses the mechanism that makes evaluations
  context-sensitive (m_sit). Expected to drop sharply on ISS (the interaction
  metric), since without m_sit the model has no mechanism for context to
  modulate evaluations positively.
- **SEM-knowledge-only**: tests whether epistemic inference alone (without
  motivational inference) is sufficient. φ_K has no a_like signal, so this
  variant should struggle on likeable-related effects.
- **SEM-motivation-only**: tests whether motivational inference alone is
  sufficient. The knowledge inference drives the main effect of form on
  competence (precise speakers are inferred to have k_p), so this variant
  should weaken the a_comp main effect.

---

### Sampling procedure

For each variant, draw 100 independent parameter combinations as follows.

**ω sampling:**
- SEM-full: ω ~ Beta(2, 2), clipped to (0.05, 0.95)
- SEM-context-blind: ω ~ Beta(2, 2), same clip
- SEM-knowledge-only: ω = 1.0 (fixed)
- SEM-motivation-only: ω = 0.0 (fixed)

**P(k) sampling** (for all variants):
- Draw (p_kp, p_knp) ~ Dirichlet(1, 1) — uniform over the 2-simplex
- Equivalent to: p_kp ~ Uniform(0, 1), p_knp = 1 − p_kp

**P(m) sampling:**
- SEM-full, SEM-knowledge-only, SEM-motivation-only:
  Draw (p_mmax, p_msit, p_mmin) ~ Dirichlet(1, 1, 1) — uniform over the
  3-simplex
- SEM-context-blind:
  Fix p_msit = 0; draw (p_mmax, p_mmin) ~ Dirichlet(1, 1), then set
  p_msit = 0.0 explicitly. The two drawn values already sum to 1.0 and
  are passed directly as p_mmax and p_mmin.

Use `numpy.random.default_rng(seed=42)` for reproducibility. Draw all 100
samples at once using `rng.dirichlet()` and `rng.beta()`.

Each drawn combination must satisfy: all priors > 0 (or == 0 for the fixed
zero in SEM-context-blind), sum-to-1 within K and M. Add an assertion to
verify this for each sample. With Dirichlet(1,...) priors, degenerate all-zero
draws should not occur, but assert defensively.

---

### Step 2.1 — Create `sem/comparison.py`

Create `sem/comparison.py`. This module implements the sampling, evaluation,
and summarization logic for the variant comparison. It must be fully
self-contained and importable independently.

**Module docstring:** explain that this module compares four nested SEM
variants via parameter sampling to assess which inferential components are
necessary for reproducing the CS1 empirical effects.

---

#### Function: `sample_parameters`

```python
def sample_parameters(
    variant: str,
    n_samples: int = 100,
    seed: int = 42
) -> list[dict]:
```

`variant` is one of: `"full"`, `"context_blind"`, `"knowledge_only"`,
`"motivation_only"`.

Returns a list of `n_samples` dicts, each with keys:
```python
{
    "omega": float,
    "priors_K": {"k_p": float, "k_np": float},
    "priors_M": {"m_max": float, "m_sit": float, "m_min": float}
}
```

Implements the sampling procedure described above exactly.

---

#### Function: `evaluate_variant`

```python
def evaluate_variant(
    variant: str,
    empirical_json_path: str,
    n_samples: int = 100,
    seed: int = 42
) -> list[dict]:
```

For each of the `n_samples` parameter combinations sampled for `variant`:
1. Build the CS1 scenario using `build_scenario()` from
   `case_studies.cs1_imprecision` with the sampled parameters
2. Call `scenario.evaluate_all()` to get model scores
3. Call `sem.fit.evaluate(model_scores, empirical_json_path)` to get all
   metrics
4. Append the result dict (augmented with the sampled parameters) to a list

Returns the list of per-sample result dicts.

**Important — zero prior handling for SEM-context-blind:** passing
`priors_M["m_sit"] = 0.0` to `build_scenario()` must be verified to work
correctly — the SEMScenario χ_m indicator function should gate out m_sit
naturally. Before running the full 100 samples, run one test call with
p_msit=0.0 and assert that the resulting evaluation scores differ from
the full model (i.e. the constraint is actually applied). If `build_scenario()`
rejects zero priors, clamp to 1e-9 instead and document the choice in a
comment.

---

#### Function: `summarize_results`

```python
def summarize_results(
    results: list[dict]
) -> dict:
```

Given the list of per-sample result dicts from `evaluate_variant`, compute
mean and standard deviation across samples for each scalar metric:
`das`, `iss`, `spearman_r`, `rmse`, `ccc`, `cds_main`, `cds_interaction`,
`cds`.

For ESR values (dicts keyed by attribute), compute mean and SD per attribute
separately.

Returns:
```python
{
    "das":             {"mean": 1.00, "sd": 0.00},
    "iss":             {"mean": 1.00, "sd": 0.00},
    "spearman_r":      {"mean": 0.71, "sd": 0.09},
    "rmse":            {"mean": 0.28, "sd": 0.04},
    "ccc":             {"mean": 0.63, "sd": 0.07},
    "cds_main":        {"mean": 1.21, "sd": 0.34},
    "cds_interaction": {"mean": 0.33, "sd": 0.11},
    "cds":             {"mean": 0.77, "sd": 0.19},
    "esr_main": {
        "a_comp": {"mean": 2.4, "sd": 0.8},
        "a_ped":  {"mean": 1.9, "sd": 0.6}
    },
    "esr_interaction": {
        "a_comp": {"mean": 1.3, "sd": 0.4},
        "a_like": {"mean": 1.2, "sd": 0.3}
    }
}
```

(Numbers above are illustrative — use actual computed values.)

---

#### Function: `compare_all_variants`

```python
def compare_all_variants(
    empirical_json_path: str,
    n_samples: int = 100,
    seed: int = 42
) -> dict:
```

Runs `evaluate_variant` and `summarize_results` for all four variants.

Returns:
```python
{
    "full":            {summary dict},
    "context_blind":   {summary dict},
    "knowledge_only":  {summary dict},
    "motivation_only": {summary dict}
}
```

---

#### Function: `print_comparison_table`

```python
def print_comparison_table(comparison: dict) -> None:
```

Prints a formatted comparison table to stdout. Format:

```
=== SEM Variant Comparison (n=100 samples per variant) ===

Metric           SEM-full       Context-blind  Know-only      Motiv-only
──────────────────────────────────────────────────────────────────────────
DAS              1.00 ± 0.00    1.00 ± 0.00    1.00 ± 0.00    0.96 ± 0.13
ISS              1.00 ± 0.00    0.00 ± 0.00    0.04 ± 0.14    1.00 ± 0.00
Spearman ρ       0.71 ± 0.09    0.54 ± 0.11    0.65 ± 0.10    0.62 ± 0.11
RMSE             0.28 ± 0.04    0.38 ± 0.06    0.31 ± 0.05    0.33 ± 0.05
CCC              0.63 ± 0.07    0.48 ± 0.09    0.57 ± 0.08    0.53 ± 0.09
CDS (main)       1.21 ± 0.34    1.45 ± 0.41    1.18 ± 0.36    1.52 ± 0.43
CDS (inter)      0.33 ± 0.11    0.68 ± 0.22    0.35 ± 0.13    0.39 ± 0.14
CDS (overall)    0.77 ± 0.19    1.07 ± 0.28    0.77 ± 0.20    0.96 ± 0.24

ESR main effects (significant only):
  a_comp         2.4 ± 0.8      ...            ...            ...
  a_ped          1.9 ± 0.6      ...            ...            ...
ESR interactions (significant only):
  a_comp         1.3 ± 0.4      ...            ...            ...
  a_like         1.2 ± 0.3      ...            ...            ...
```

---

### Step 2.2 — Create `scripts/run_comparison.py`

Create a standalone runner script `scripts/run_comparison.py`:

1. Imports `compare_all_variants` and `print_comparison_table` from
   `sem.comparison`
2. Runs the full comparison with default settings (n=100, seed=42)
3. Prints the comparison table to stdout
4. Saves the full results dict as
   `data/cs1_experiment/processed/variant_comparison_cs1.json`
   (use `json.dump` with `indent=2`; convert numpy floats to Python floats
   via explicit `float()` casting before serialization)

Run the script and verify it completes without error. Expected runtime: under
60 seconds for n=100. If slower, profile and report — do not silently reduce
n or coarsen parameters.

---

### Step 2.3 — Add tests

Add a new test class `TestVariantComparison` to `tests/test_case_studies.py`:

```python
class TestVariantComparison:
    """Verify sem/comparison.py sampling and summarization logic."""

    def test_sample_parameters_count(self):
        # sample_parameters("full", n_samples=10) returns list of length 10

    def test_sample_parameters_omega_range(self):
        # all sampled omega in (0, 1) for "full" variant

    def test_sample_parameters_priors_sum_to_one(self):
        # priors_K values sum to 1.0 for each sample (within float tolerance)
        # priors_M values sum to 1.0 for each sample

    def test_context_blind_msit_zero(self):
        # all samples for "context_blind" have priors_M["m_sit"] == 0.0

    def test_knowledge_only_omega_one(self):
        # all samples for "knowledge_only" have omega == 1.0

    def test_motivation_only_omega_zero(self):
        # all samples for "motivation_only" have omega == 0.0

    def test_summarize_results_keys(self):
        # summarize_results output contains all expected metric keys:
        # das, iss, spearman_r, rmse, ccc, cds_main, cds_interaction, cds,
        # esr_main, esr_interaction

    def test_summarize_results_sd_nonnegative(self):
        # all sd values >= 0 for scalar metrics

    def test_compare_all_variants_runs(self):
        # compare_all_variants completes and returns dict with exactly 4 keys:
        # "full", "context_blind", "knowledge_only", "motivation_only"
        # Skip if empirical_means_cs1.json not present
```

Use a file-existence check to skip `test_compare_all_variants_runs` gracefully
if the JSON is not present (CI environment).

All 32 existing tests must continue to pass.

---

### Step 2.4 — Update `cs1_imprecision.py` runner

Add a call to `compare_all_variants` in the `if __name__ == "__main__":` block
of `case_studies/cs1_imprecision.py`, after the existing fit evaluation:

```python
from sem.comparison import compare_all_variants, print_comparison_table

comparison = compare_all_variants(
    empirical_json_path="data/cs1_experiment/processed/empirical_means_cs1.json"
)
print_comparison_table(comparison)
```

Guard with a try/except that catches a missing JSON file gracefully and prints
a warning instead of crashing.

---

### Completion check

1. Run `PYTHONPATH=. python scripts/run_comparison.py` — verify table prints
   and JSON is saved to `data/cs1_experiment/processed/variant_comparison_cs1.json`
2. Run `PYTHONPATH=. python case_studies/cs1_imprecision.py` — verify effect
   checks ✓, fit table, and comparison table all print in sequence
3. Run `PYTHONPATH=. python -m pytest tests/ -v` — all tests pass
   (32 original + new TestVariantComparison tests)
4. **Scientific sanity check — ISS:** inspect the ISS column in the printed
   table. SEM-context-blind must show a notably lower ISS mean than SEM-full.
   If ISS is identical or higher for context-blind, stop and report — this
   would indicate the P(m_sit)=0 constraint is not being applied correctly.

Report: `TASK 2 COMPLETE: sem/comparison.py implemented, variant comparison
run and saved, N tests passing, ISS: full=X ± Y vs context-blind=A ± B`

---

## Notes for all tasks

- Never modify `sem/model.py` core equations without updating the docstring
- Variable naming convention: knowledge states `k_`, strategies `m_`,
  attributes `a_`, utterances `v_`, contexts `c_`
- All new files need docstrings explaining their purpose
- After completing all steps in a task, print a one-line summary:
  `TASK N COMPLETE: <what was done, what tests pass>`
