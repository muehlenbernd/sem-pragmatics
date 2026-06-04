# SEM Revision — Progress Report (2026-06-02)

Prepared for review in the main Claude.ai planning chat. This session covered:
(1) updating manuscript Section 3 text to the revised analysis, (2) restructuring
the notebooks and building a full paper-reproduction notebook, (3) regenerating
all data figures and wiring them into the manuscript, (4) updating Figure 7's
source and closing the Section 4 robustness numbers, and (5) implementing Task 1
(quantitative fit metrics, `sem/fit.py`).

All code runs; the manuscript compiles cleanly; the test suite is at **32 passing**.

---

## 1. Key methodological decision (needs your sign-off in the science chat)

The original paper described the motivation→rating analysis (the basis for
deriving φ_K/φ_M) as a **linear mixed-effects model**. The revised code derives
φ via **independent-samples t-tests** at **p < .01**. We resolved this as follows,
after explicit discussion:

- **§3.2 significant effects (Task 1 ratings)** → keep **LMM** (matches the
  original `lmerTest` analysis; re-verified, all hold at p<.001).
- **§3.3 φ-derivation (Task 2 motives)** → use **t-tests** at p<.01, because the
  trivalent derivation is a coarse discretization heuristic and (crucially) the
  model's conclusions are **robust to the choice**.

**The borderline cell.** The one place the two methods disagree is
`needed × competent`: significant under a t-test (**p = .010** → `m_sit` gets
`a_comp = +1`) but not under an LMM with scenario random intercept
(**p = .012** → would be 0). We **retain the t-test value**. Justification, now in
a manuscript footnote: all four CS1 target effects hold under *either* value, so
the choice does not drive any conclusion.

> **Reviewer-risk note for the science chat:** mixing methods invites a "why
> different methods?" question. The defense is (a) discretization heuristic and
> (b) robustness — *not* "it preserves our parameters." We should make sure the
> footnote frames it that way (it currently does). Do not argue the t-test is
> *more correct* than the LMM (it isn't, statistically); argue it's *adequate and
> the result is robust*.

---

## 2. Manuscript Section 3 text changes (done, compiles clean)

- **§3.2 Task-2 method sentence**: "linear mixed effects model…" → "independent-
  samples *t*-tests (Experiment 1)".
- **Threshold**: p < .05 → **p < .01** (text + Figure 7 caption).
- **Significant-effect count**: 14/30 → **7/30**.
- **§3.3 mapping**: now `k_p ← {knewExact, infoAvail}`, `k_¬p ← {notKnow,
  infoNotAvail}`, `m_sit ← {needed, realWorld, purpose}` (infoAvail/infoNotAvail
  moved from m_sit to the knowledge states).
- **§3.3 φ-derivation sentence**: rewritten for t-tests + p<.01 + majority-vote
  aggregation, with the borderline-cell footnote.
- **§3.4 robustness**: "2,823,576 combinations … 99.7% … 0.3%" → "**2,593,080**
  valid combinations … **99.87%** … **0.13%**", ω range corrected to 0.05–0.95.
- **Unchanged (re-verified, correct as written)**: the §3.2 four-effects sentence
  (all p<.001) and the §3.2 lmerTest methods sentence.

---

## 3. Reproduced results (all match the manuscript)

### §3.2 — Experimental effects (LMM, Experiment 1)
| Effect | b | p | holds? |
|---|---|---|---|
| form → competent (precise>approx) | +0.434 | 1.02e-12 | ✓ p<.001 |
| form → pedantic (precise>approx) | +0.442 | 8.96e-07 | ✓ p<.001 |
| form → likeable (no main effect) | +0.074 | 0.237 | ✓ n.s. |
| form × context on competent | +0.276 | 5.82e-06 | ✓ p<.001 |
| form × context on likeable | +0.212 | 7.22e-04 | ✓ p<.001 |

### §3.3 — φ derivation (t-tests, p<.01): 7 of 30 significant
**φ_K** (columns: a_comp, a_like, a_ped):
`k_p (+1, 0, 0)`, `k_¬p (−1, 0, −1)`
**φ_M**:
`m_max (0, 0, +1)`, `m_sit (+1, +1, 0)`, `m_min (−1, 0, 0)`
These are the committed `IMPACT_K`/`IMPACT_M`, and the notebook confirms they
equal the live derivation.

### §3.4 — Balanced model (ω=0.5, uniform priors), E_L
| attr | v_prc·c_HP | v_prc·c_LP | v_apx·c_HP | v_apx·c_LP |
|---|---|---|---|---|
| a_comp | +0.750 | +0.500 | −0.250 | 0.000 |
| a_like | +0.250 | 0.000 | +0.083 | +0.208 |
| a_ped | +0.250 | +0.500 | −0.167 | −0.167 |

All 4 target effects ✓. **Robustness (step 0.02): 2,593,080 combos → 99.87%
all-four**, 0.13% three-of-four; the only failing effect is Effect 1, in the
corner ω low / P(k_p) low / P(m_sit) high (matches footnote).

### §4 — CS2 (pragmatic violations), balanced model
Predictions: `a_comp: v_A +1.0, v_a +0.5, v_B −0.5, v_b −0.5`;
`a_like: v_A +0.5, v_a −0.125, v_B −0.75, v_b −0.75`. All 6 effects ✓.
**Robustness (step 1/14): 1,063,348 combos → 96.50% all-six**, 3.5% five-of-six;
misses concentrated in Effects 3 & 4 (matches footnote). *(Count corrected from
the manuscript's prior 572,572; rate unchanged.)*

### Empirical means (raw 1–7), Experiment 1, N=362
n per condition: v_prc_c_HP 90, v_prc_c_LP 87, v_apx_c_HP 92, v_apx_c_LP 93.
| attr | v_prc·c_HP | v_prc·c_LP | v_apx·c_HP | v_apx·c_LP |
|---|---|---|---|---|
| competent | 6.23 | 5.82 | 4.78 | 5.51 |
| likeable | 5.22 | 4.87 | 4.64 | 5.15 |
| pedantic | 4.18 | 4.69 | 3.54 | 3.55 |

---

## 4. Task 1 — Quantitative fit metrics (`sem/fit.py`) — COMPLETE

Implements the Mühlenbernd (2026) metric suite. Human means are normalized to
[−1, 1] via `(x−4)/3`; the model is already in [−1, 1]; **all metrics use the
normalized human data** (essential for ESR/CDS/RMSE/CCC; sign/rank metrics are
unaffected). CS1 only (CS2 has no accessible Task-2 data).

**Files:** `scripts/compute_empirical_means.py` → `empirical_means_cs1.json`;
`sem/fit.py` (normalize_likert, load_empirical_means, extract_effects, das, iss,
esr, cds, global_metrics, evaluate); `run_fit_evaluation()` in
`cs1_imprecision.py`; `TestFitMetrics` (10 tests).

### Fit results — CS1 balanced model
```
Global:   Spearman ρ 0.55 · RMSE 0.26 · CCC 0.61
Structure: DAS 1.00 (2/2)   ISS 1.00 (2/2)
ESR main:        a_comp 2.55,  a_ped 1.83
ESR interaction: a_comp 1.32,  a_like 1.31
CDS: main 1.19 · interaction 0.31 · overall 0.75
```

**Interpretation (for the paper narrative):**
- **Structure is captured perfectly.** Both significant form×context interactions
  match in sign (ISS = 1.00, 2/2) and both significant main effects match in sign
  (DAS = 1.00, 2/2). DAS and ISS are now restricted to the four statistically
  significant effects (per the paper's definition), so the *non-significant*
  a_like main effect (human slightly +, model slightly −) — previously the lone
  DAS miss — is correctly excluded rather than scored as noise.
- **Magnitudes are over-shot.** On the significant main effects the model's effect
  sizes are ~1.8–2.6× the human ones (ESR), driving CDS up to 0.75. The model is
  directionally/structurally right but **not magnitude-calibrated** under the
  balanced model with uniform priors.
- This is an honest, reportable limitation and a natural motivation for parameter
  fitting (the priors and ω are currently fixed, not fitted). Worth a sentence in
  the discussion.

---

## 5. Figures (regenerated; driven by the notebook)

`notebooks/sem_paper.ipynb` is the single source for the data figures, saved to
`figures/generated/` (PDF + PNG). Design choices made this session:

- **Fig 4** (Task-1 empirical): mean ± SD bars by form×context; dashed ceiling
  line at 7; y-label "mean (sd)"; taller aspect; no in-figure title.
- **Fig 5** (Task-2 frequencies): **y-axis = % of participants (N=362), 5% steps**;
  stacked by form; taller aspect; no title.
- **Fig 8** (model vs empirical): two rows (model E_L / empirical normalized via
  `(x−4)/3`); **bars anchored at the scale floor −1** (0 treated as a
  median/intermediate score, not a neutral point); no title.
- **Fig 10** (CS2 predictions): bars anchored at −1; **paired (v_A,v_B) then
  (v_a,v_b)** to mirror Fig 8's coral–blue pairs; colored by topic; no title.
- **Fig 7** (`overviewTask2.tex`): panels (b) p-values [7 bold at p<.01], (c)
  mappings, (d) φ matrices all updated; recompiled to `overviewTask2.pdf`.
- **Integration**: `\graphicspath{{figures/}{../figures/generated/}}` so the
  manuscript pulls Figs 4/5/8/10 straight from notebook output (auto-refresh on
  re-run). Old figure files are no longer referenced.

**Notebook robustness defaults** were coarsened for runtime — CS1 `STEP = 0.1`,
CS2 `STEP2 = 1/7` — each with a note that the paper uses 0.02 / (1/14) giving
99.87% / 96.5%, and a one-line change reproduces the exact figure.

---

## 6. State of the repo

- `manuscript/sem_paper.tex` — compiles clean (0 undefined refs, no missing
  graphics); Figs 4/5/8/10 embedded from `figures/generated/`; Fig 7 from updated
  `overviewTask2.pdf`.
- `notebooks/sem_demo.ipynb` — restored to original 18 cells (Section 3 repro
  block removed).
- `notebooks/sem_paper.ipynb` — full reproduction notebook (numbers, effects,
  figures), runs end-to-end in ~20 s.
- `sem/fit.py`, `scripts/compute_empirical_means.py`,
  `scripts/_build_paper_notebook.py` (notebook builder) — new.
- Tests: **32 passing** (22 prior + 10 `TestFitMetrics`).

---

## 7. Open items / next steps

1. **Fill the two `[NOTE FOR REVISION]` fit-measure placeholders** (§3.4 and
   §4.2) with the Task-1 numbers. Note: fit is **CS1-only** — §4.2 likely just
   gets a sentence stating CS2 has no accessible empirical data for quantitative
   fit. *(Deferred to next session.)*
2. Optionally surface the fit metrics in `sem_paper.ipynb`.
3. **Task 2 — RSA baseline** (`baselines/rsa_baseline.py`), per PLAN.md Step B.
4. **Submission housekeeping**: the `../figures/generated/` relative path is great
   for the revision loop but should be flattened (copy PDFs into the manuscript
   folder) at journal upload; and `overviewTask2.tex` must be recompiled by hand
   after any φ/mapping change (it is the one figure not auto-refreshed).

---

## 8. Questions for the science chat
- Is the LMM-for-effects / t-test-for-φ split acceptable as framed (heuristic +
  robustness footnote)? Any preferred wording?
- How hard do we lean on the **calibration finding** (good structure, over-shot
  magnitudes)? Frame as a limitation, or as motivation for a fitted-parameter
  follow-up — or both?
- For §4.2, is a one-sentence "no quantitative fit (no accessible data)" the right
  call, or do we want to source the Beltrama & Papafragou means for a CS2 fit?
