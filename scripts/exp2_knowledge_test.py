"""
scripts/exp2_knowledge_test.py
==============================
D1: Experiment 2 as a held-out test of the P(k|u) stage.

Analysis plan: TODO_meaning_revision.md → Details → D1.
Design principle: nothing is refitted.  φ_K, φ_M stay as derived from Exp. 1
(Task 2 motivation classes); ω = .5 and flat P(m) in the balanced setting;
the only quantity that differs between models is P(k_p).

Run:
    PYTHONPATH=. python scripts/exp2_knowledge_test.py

Output:
    data/cs1_experiment/processed/exp2_knowledge_test_cs1.json
    revision/SESSION-F-RESULTS.md  (untracked)
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from case_studies.cs1_imprecision import build_scenario
from sem.comparison import sample_parameters
from sem.fit import normalize_likert, load_empirical_means, extract_effects

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

E1_PATH  = "data/cs1_experiment/processed/experiment1.csv"
E2_PATH  = "data/cs1_experiment/processed/experiment2.csv"
E1_MEANS = "data/cs1_experiment/processed/empirical_means_cs1.json"
E2_MEANS = "data/cs1_experiment/processed/empirical_means_cs1_exp2.json"
OUT_JSON   = "data/cs1_experiment/processed/exp2_knowledge_test_cs1.json"
OUT_REPORT = "revision/SESSION-F-RESULTS.md"

ATTRS   = ["a_comp", "a_like", "a_ped"]
FORMS   = ["v_prc", "v_apx"]
CTXS    = ["c_HP", "c_LP"]
FORM_MAP = {"precise": "v_prc", "approx": "v_apx"}
CTX_MAP  = {"highPr": "c_HP", "lowPr": "c_LP"}
RATTR_MAP = {"competent": "a_comp", "likeable": "a_like", "pedantic": "a_ped"}

# CS1 significant effects (from Exp. 1, used unchanged for all metric calls)
SIG_MAIN   = ["a_comp", "a_ped"]
SIG_INTER  = ["a_comp", "a_like"]


# ---------------------------------------------------------------------------
# Helper: normalized cell means from a DataFrame
# ---------------------------------------------------------------------------

def cell_means_norm(df: pd.DataFrame) -> Dict:
    """Normalized cell means: result[a][v][c] ∈ [-1, 1]."""
    out: Dict = {}
    for raw_col, sem_a in RATTR_MAP.items():
        out[sem_a] = {}
        for form_val, v in FORM_MAP.items():
            out[sem_a][v] = {}
            for ctx_val, c in CTX_MAP.items():
                sub = df[(df.form == form_val) & (df.context == ctx_val)][raw_col]
                out[sem_a][v][c] = round(normalize_likert(float(sub.mean())), 6)
    return out


# ---------------------------------------------------------------------------
# Q1a: Manipulation check and cell means
# ---------------------------------------------------------------------------

def q1_descriptives(e1: pd.DataFrame, e2: pd.DataFrame) -> Dict:
    """Cell means (raw + normalized) for both experiments."""
    out: Dict = {"exp1": {}, "exp2": {}}
    for exp, df in [("exp1", e1), ("exp2", e2)]:
        for raw_col, sem_a in RATTR_MAP.items():
            out[exp][sem_a] = {}
            for form_val, v in FORM_MAP.items():
                out[exp][sem_a][v] = {}
                for ctx_val, c in CTX_MAP.items():
                    sub = df[(df.form == form_val) & (df.context == ctx_val)][raw_col]
                    out[exp][sem_a][v][c] = {
                        "n": int(len(sub)),
                        "mean_raw": round(float(sub.mean()), 4),
                        "sd_raw":   round(float(sub.std(ddof=1)), 4),
                        "mean_norm": round(normalize_likert(float(sub.mean())), 4),
                    }
    return out


def q1_manipulation_check(e1: pd.DataFrame, e2: pd.DataFrame) -> Dict:
    """Task 3 manipulation check for approximate form."""
    result = {}
    for col, label in [("t3_didntKnow", "didnt_know"), ("t3_easierSpeaker", "easier_speaker")]:
        apx1 = e1[e1.form == "approx"][col].mean()
        apx2 = e2[e2.form == "approx"][col].mean()
        result[label] = {
            "exp1_approx": round(float(apx1), 3),
            "exp2_approx": round(float(apx2), 3),
        }
    return result


def q1_delta_ci(e1: pd.DataFrame, e2: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """Exp2 − Exp1 per normalized cell with 95% CI (independent-samples SE)."""
    z = sp_stats.norm.ppf(1 - alpha / 2)
    out: Dict = {}
    for raw_col, sem_a in RATTR_MAP.items():
        out[sem_a] = {}
        for form_val, v in FORM_MAP.items():
            out[sem_a][v] = {}
            for ctx_val, c in CTX_MAP.items():
                c1 = e1[(e1.form == form_val) & (e1.context == ctx_val)][raw_col]
                c2 = e2[(e2.form == form_val) & (e2.context == ctx_val)][raw_col]
                # normalize_likert derivative = 1/3 for a 1-7 scale
                scale = 1.0 / 3.0
                m1 = normalize_likert(float(c1.mean()))
                m2 = normalize_likert(float(c2.mean()))
                se1 = float(c1.std(ddof=1)) * scale / math.sqrt(len(c1))
                se2 = float(c2.std(ddof=1)) * scale / math.sqrt(len(c2))
                se_d = math.sqrt(se1 ** 2 + se2 ** 2)
                d = round(m2 - m1, 4)
                out[sem_a][v][c] = {
                    "delta": d,
                    "ci_lo": round(d - z * se_d, 4),
                    "ci_hi": round(d + z * se_d, 4),
                }
    return out


def q1_exp2_effects_alone(e2: pd.DataFrame) -> Dict:
    """Four target effects in Exp. 2 data alone (sign from cell means)."""
    means = cell_means_norm(e2)
    efx = extract_effects(means)
    return {
        "comp_main_positive":  efx["a_comp"]["main_effect"] > 0,
        "ped_main_positive":   efx["a_ped"]["main_effect"] > 0,
        "comp_inter_positive": efx["a_comp"]["interaction"] > 0,
        "like_inter_positive": efx["a_like"]["interaction"] > 0,
        "comp_main_effect": round(efx["a_comp"]["main_effect"], 4),
        "ped_main_effect":  round(efx["a_ped"]["main_effect"], 4),
        "comp_interaction": round(efx["a_comp"]["interaction"], 4),
        "like_interaction": round(efx["a_like"]["interaction"], 4),
    }


# ---------------------------------------------------------------------------
# Q1b: Mixed model via R/lmerTest
# ---------------------------------------------------------------------------

R_SCRIPT = r"""
suppressMessages(library(lmerTest))
suppressMessages(library(jsonlite))
df <- read.csv("{csv_path}")
df$knowledge <- factor(df$knowledge)
df$form      <- factor(df$form)
df$context   <- factor(df$context)
# sum coding
contrasts(df$knowledge) <- contr.sum(2)
contrasts(df$form)      <- contr.sum(2)
contrasts(df$context)   <- contr.sum(2)
out <- list()
for (attr in c("competent", "likeable", "pedantic")) {{
  df$rating <- df[[attr]]
  m <- lmer(rating ~ form * context * knowledge + (1|scenario), data=df,
            REML=FALSE)
  s <- summary(m)$coefficients
  row_names <- rownames(s)
  out[[attr]] <- list(
    coef        = as.numeric(s[, "Estimate"]),
    se          = as.numeric(s[, "Std. Error"]),
    t_value     = as.numeric(s[, "t value"]),
    p_value     = as.numeric(s[, "Pr(>|t|)"]),
    row_names   = row_names,
    n_obs       = nrow(df)
  )
}}
cat(toJSON(out, pretty=FALSE))
"""


def q1_mixed_model(e1: pd.DataFrame, e2: pd.DataFrame) -> Dict:
    """Fit rating ~ form * context * knowledge + (1|scenario) via lmerTest."""
    merged = pd.concat([e1.copy(), e2.copy()], ignore_index=True)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as fcsv:
        csv_path = fcsv.name
        merged.to_csv(fcsv, index=False)

    with tempfile.NamedTemporaryFile(suffix=".R", delete=False, mode="w") as fR:
        r_path = fR.name
        fR.write(R_SCRIPT.format(csv_path=csv_path))

    try:
        proc = subprocess.run(
            ["Rscript", r_path],
            capture_output=True, text=True, check=True
        )
        result = json.loads(proc.stdout)
    except subprocess.CalledProcessError as exc:
        print("R stderr:", exc.stderr[:500], file=sys.stderr)
        result = {"error": exc.stderr[:200]}
    finally:
        os.unlink(csv_path)
        os.unlink(r_path)

    # Round numbers
    out: Dict = {}
    for attr, data in result.items():
        if not isinstance(data, dict):
            continue
        out[attr] = []
        for i, name in enumerate(data["row_names"]):
            out[attr].append({
                "term":    name,
                "coef":    round(data["coef"][i], 4),
                "se":      round(data["se"][i], 4),
                "t":       round(data["t_value"][i], 3),
                "p":       round(data["p_value"][i], 4),
            })
    return out


# ---------------------------------------------------------------------------
# Q2: Calibration flatness across P(k_p)
# ---------------------------------------------------------------------------

def q2_calibration(e1_means: Dict, samples: List[Dict]) -> Dict:
    """
    DAS, ISS, RMSE, CDS at each P(k_p) ∈ {.05, .10, ..., .95}:
      (i)  ω=.5, flat P(m), balanced setting
      (ii) mean ± SD over 100 sampled (ω, P(m)) with P(k_p) fixed
    """
    from sem.fit import das, iss, global_metrics, esr, cds, extract_effects

    p_vals = [round(x * 0.05, 2) for x in range(1, 20)]  # .05 to .95
    human_means = e1_means
    human_effects = extract_effects(human_means)

    balanced_rows = []
    for p in p_vals:
        s = build_scenario(priors_K={"k_p": p, "k_np": 1 - p})
        scores = s.evaluate_all()
        model_effects = extract_effects(scores)
        esr_vals = esr(human_effects, model_effects, SIG_MAIN, SIG_INTER)
        cds_vals = cds(esr_vals)
        glob = global_metrics(human_means, scores)
        balanced_rows.append({
            "pk_p":  p,
            "das":   round(das(human_effects, model_effects, SIG_MAIN), 3),
            "iss":   round(iss(human_effects, model_effects, SIG_INTER), 3),
            "rmse":  round(glob["rmse"], 4),
            "cds":   round(cds_vals["cds"], 4),
            "cds_main": round(cds_vals["cds_main"], 4),
            "cds_interaction": round(cds_vals["cds_interaction"], 4),
        })

    sampled_rows = []
    for p in p_vals:
        metrics_per_sample = {"das": [], "iss": [], "rmse": [], "cds": []}
        for samp in samples:
            priors_K = {"k_p": p, "k_np": 1 - p}
            sc = build_scenario(
                omega=samp["omega"],
                priors_K=priors_K,
                priors_M=samp["priors_M"],
            )
            scores = sc.evaluate_all()
            model_effects = extract_effects(scores)
            esr_vals = esr(human_effects, model_effects, SIG_MAIN, SIG_INTER)
            cds_vals = cds(esr_vals)
            glob = global_metrics(human_means, scores)
            metrics_per_sample["das"].append(das(human_effects, model_effects, SIG_MAIN))
            metrics_per_sample["iss"].append(iss(human_effects, model_effects, SIG_INTER))
            metrics_per_sample["rmse"].append(glob["rmse"])
            metrics_per_sample["cds"].append(cds_vals["cds"])
        sampled_rows.append({
            "pk_p": p,
            **{k: {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4)}
               for k, v in metrics_per_sample.items()},
        })

    return {"balanced": balanced_rows, "sampled": sampled_rows}


# ---------------------------------------------------------------------------
# D1 Step 1: Structural prediction
# ---------------------------------------------------------------------------

def d1_step1(inference: str = "staged") -> Dict:
    """P(k_p|v_prc) = 1 for any prior, under both listeners."""
    results = {}
    for p in [0.1, 0.5, 0.9]:
        sc = build_scenario(priors_K={"k_p": p, "k_np": 1 - p}, inference=inference)
        pk_prc_HP = round(sc.prob_k("k_p", "v_prc", "c_HP"), 6)
        pk_prc_LP = round(sc.prob_k("k_p", "v_prc", "c_LP"), 6)
        pk_apx_HP = round(sc.prob_k("k_p", "v_apx", "c_HP"), 6)
        pk_apx_LP = round(sc.prob_k("k_p", "v_apx", "c_LP"), 6)
        results[f"pk_p={p}"] = {
            "P(k_p|v_prc,c_HP)": pk_prc_HP,
            "P(k_p|v_prc,c_LP)": pk_prc_LP,
            "P(k_p|v_apx,c_HP)": pk_apx_HP,
            "P(k_p|v_apx,c_LP)": pk_apx_LP,
        }
    all_prc_are_one = all(
        v["P(k_p|v_prc,c_HP)"] == 1.0 and v["P(k_p|v_prc,c_LP)"] == 1.0
        for v in results.values()
    )
    return {"samples": results, "all_precise_cells_P_kp_equals_1": all_prc_are_one}


# ---------------------------------------------------------------------------
# D1 Step 2: Two models (base P(k_p)=.5 vs knowledge P(k_p)=1.0)
# ---------------------------------------------------------------------------

def d1_step2(e1_means: Dict, inference: str = "staged") -> Dict:
    """All 12 cells under base (P(k_p)=.5) and knowledge (P(k_p)=1.0) models."""
    base = build_scenario(priors_K={"k_p": 0.5, "k_np": 0.5}, inference=inference)
    know = build_scenario(priors_K={"k_p": 1.0, "k_np": 0.0}, inference=inference)
    base_scores = base.evaluate_all()
    know_scores = know.evaluate_all()

    cells = {}
    for a in ATTRS:
        cells[a] = {}
        for v in FORMS:
            cells[a][v] = {}
            for c in CTXS:
                e_base = round(base_scores[a][v][c], 4)
                e_know = round(know_scores[a][v][c], 4)
                delta  = round(e_know - e_base, 4)
                cells[a][v][c] = {
                    "E_base":      e_base,
                    "E_knowledge": e_know,
                    "delta_predicted": delta,
                }
    return cells


# ---------------------------------------------------------------------------
# D1 Step 3: Motivation shift P(m|v_apx, c) under base vs knowledge
# ---------------------------------------------------------------------------

def d1_step3(inference: str = "staged") -> Dict:
    """Marginal P(m|v_apx, c) under base (P(k_p)=.5) vs knowledge (P(k_p)=1.0)."""
    base = build_scenario(priors_K={"k_p": 0.5, "k_np": 0.5}, inference=inference)
    know = build_scenario(priors_K={"k_p": 1.0, "k_np": 0.0}, inference=inference)

    def marginal_pm(sc, v: str, c: str) -> Dict:
        total: Dict[str, float] = {m: 0.0 for m in sc.strategies}
        for k in sc.knowledge_states:
            p_k = sc.prob_k(k, v, c)
            for m in sc.strategies:
                total[m] += p_k * sc.prob_m(m, k, c, v)
        return {m: round(total[m], 4) for m in sc.strategies}

    return {
        "base": {
            "v_apx_c_HP": marginal_pm(base, "v_apx", "c_HP"),
            "v_apx_c_LP": marginal_pm(base, "v_apx", "c_LP"),
        },
        "knowledge": {
            "v_apx_c_HP": marginal_pm(know, "v_apx", "c_HP"),
            "v_apx_c_LP": marginal_pm(know, "v_apx", "c_LP"),
        },
    }


# ---------------------------------------------------------------------------
# D1 Step 4: Model vs empirical Exp.2 − Exp.1 differences
# ---------------------------------------------------------------------------

def d1_step4(delta_ci: Dict, step2_scores: Dict) -> Dict:
    """Per-cell: predicted Δ vs observed Δ (normalized), sign agreement."""
    out: Dict = {}
    approx_sign_agree = []
    for a in ATTRS:
        out[a] = {}
        for v in FORMS:
            out[a][v] = {}
            for c in CTXS:
                obs = delta_ci[a][v][c]["delta"]
                pred = step2_scores[a][v][c]["delta_predicted"]
                sign_agree = (obs > 0) == (pred > 0) if obs != 0 else None
                out[a][v][c] = {
                    "observed_delta":  obs,
                    "predicted_delta": pred,
                    "sign_agree":      sign_agree,
                }
                if v == "v_apx":
                    approx_sign_agree.append(sign_agree)
    n_agree = sum(1 for x in approx_sign_agree if x is True)
    n_disagree = sum(1 for x in approx_sign_agree if x is False)
    out["_summary"] = {
        "approx_cells_sign_agree": n_agree,
        "approx_cells_sign_disagree": n_disagree,
        "approx_cells_obs_zero": sum(1 for x in approx_sign_agree if x is None),
    }
    return out


# ---------------------------------------------------------------------------
# D1 Step 5: Base-prior sweep (primary analysis)
# ---------------------------------------------------------------------------

def _sweep_predicted_delta(inference: str = "staged") -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute predicted Δ for each P(k_p) in [0.01, ..., 1.00] (100 values).
    Returns (p_vals, delta_array) where delta_array has shape (100, 3, 2)
    representing (n_p, n_attrs, n_ctxs) for approximate cells only.
    """
    p_vals = np.arange(0.01, 1.001, 0.01)[:100]
    know = build_scenario(priors_K={"k_p": 1.0, "k_np": 0.0}, inference=inference)
    E_know_apx = np.array([
        [know.evaluate_all()[a]["v_apx"][c] for c in CTXS]
        for a in ATTRS
    ])  # shape (3, 2)

    delta_arr = np.zeros((len(p_vals), len(ATTRS), len(CTXS)))
    for i, p in enumerate(p_vals):
        base = build_scenario(priors_K={"k_p": float(p), "k_np": float(1 - p)}, inference=inference)
        scores = base.evaluate_all()
        E_base_apx = np.array([
            [scores[a]["v_apx"][c] for c in CTXS]
            for a in ATTRS
        ])
        delta_arr[i] = E_know_apx - E_base_apx

    return p_vals, delta_arr


def _sweep_predicted_delta_precise(inference: str = "staged") -> np.ndarray:
    """E_knowledge - E_base for v_prc cells (shape: (3, 2)), should be all ~0."""
    know = build_scenario(priors_K={"k_p": 1.0, "k_np": 0.0}, inference=inference)
    base = build_scenario(priors_K={"k_p": 0.5, "k_np": 0.5}, inference=inference)
    know_scores = know.evaluate_all()
    base_scores = base.evaluate_all()
    return np.array([
        [know_scores[a]["v_prc"][c] - base_scores[a]["v_prc"][c] for c in CTXS]
        for a in ATTRS
    ])


def _sse(predicted: np.ndarray, observed: np.ndarray) -> float:
    """SSE over cells (predicted and observed have same shape)."""
    return float(np.sum((predicted - observed) ** 2))


def d1_step5(
    e1: pd.DataFrame,
    e2: pd.DataFrame,
    delta_ci: Dict,
    n_bootstrap: int = 2000,
    seed: int = 42,
    inference: str = "staged",
) -> Dict:
    """
    Base-prior sweep: find P0* = argmin SSE(predicted Δ, observed Δ).
    Primary criterion: 6 approximate cells.
    Secondary criterion: all 12 cells (precise cells predicted to have Δ=0).
    Bootstrap CI: resample participants within experiment × (form × context), B=2000.
    """
    p_vals, delta_pred_apx = _sweep_predicted_delta(inference=inference)
    # delta_pred_apx: (100, 3, 2) — approximate cells only

    # Observed Δ matrix (3 attrs × 2 ctxs) for approx form
    obs_apx = np.array([
        [delta_ci[a]["v_apx"][c]["delta"] for c in CTXS]
        for a in ATTRS
    ])  # shape (3, 2)

    # Observed Δ for all 12 cells
    obs_all = np.array([
        [[delta_ci[a][v][c]["delta"] for c in CTXS] for v in FORMS]
        for a in ATTRS
    ])  # shape (3, 2, 2) — [attr, form, ctx]

    # Predicted Δ for precise cells ≈ 0
    pred_prc = _sweep_predicted_delta_precise(inference=inference)  # (3, 2)

    # Primary P0* (observed Δ, 6 approx cells)
    sse_by_p = np.sum((delta_pred_apx - obs_apx[np.newaxis]) ** 2, axis=(1, 2))
    p0_primary_idx = int(np.argmin(sse_by_p))
    p0_primary = round(float(p_vals[p0_primary_idx]), 2)

    # Secondary P0* (all 12 cells: approx + precise with predicted Δ=0)
    obs_all_apx = obs_all[:, FORMS.index("v_apx"), :]   # (3, 2)
    obs_all_prc = obs_all[:, FORMS.index("v_prc"), :]   # (3, 2)
    sse_12 = sse_by_p + np.sum((pred_prc[np.newaxis] - obs_all_prc[np.newaxis]) ** 2, axis=(1, 2))
    p0_secondary_idx = int(np.argmin(sse_12))
    p0_secondary = round(float(p_vals[p0_secondary_idx]), 2)

    # Bootstrap CI for primary P0*
    rng = np.random.default_rng(seed)
    # Pool rows per (experiment × form × context)
    groups: Dict[str, pd.DataFrame] = {}
    for exp_name, df in [("exp1", e1), ("exp2", e2)]:
        for form_val, v in FORM_MAP.items():
            for ctx_val, c in CTX_MAP.items():
                key = f"{exp_name}_{v}_{c}"
                groups[key] = df[
                    (df.form == form_val) & (df.context == ctx_val)
                ].reset_index(drop=True)

    boot_p0 = []
    for _ in range(n_bootstrap):
        # Resample within each group
        delta_boot = np.zeros((len(ATTRS), len(CTXS)))
        for ai, a_sem in enumerate(ATTRS):
            raw_col = [k for k, v in RATTR_MAP.items() if v == a_sem][0]
            for ci, c in enumerate(CTXS):
                e1_key = f"exp1_v_apx_{c}"
                e2_key = f"exp2_v_apx_{c}"
                g1 = groups[e1_key]
                g2 = groups[e2_key]
                s1 = g1.sample(n=len(g1), replace=True, random_state=rng.integers(2**31))[raw_col]
                s2 = g2.sample(n=len(g2), replace=True, random_state=rng.integers(2**31))[raw_col]
                delta_boot[ai, ci] = (
                    normalize_likert(float(s2.mean())) - normalize_likert(float(s1.mean()))
                )
        sse_boot = np.sum((delta_pred_apx - delta_boot[np.newaxis]) ** 2, axis=(1, 2))
        boot_p0.append(float(p_vals[int(np.argmin(sse_boot))]))

    boot_p0_arr = np.array(boot_p0)
    ci_lo = round(float(np.percentile(boot_p0_arr, 2.5)), 2)
    ci_hi = round(float(np.percentile(boot_p0_arr, 97.5)), 2)

    # Spread of P0* over 100 sampled (ω, P(m)) combinations
    samples = sample_parameters("full", n_samples=100, seed=42)
    p0_sampled = []
    for samp in samples:
        know_samp = build_scenario(
            omega=samp["omega"], priors_K={"k_p": 1.0, "k_np": 0.0},
            priors_M=samp["priors_M"], inference=inference,
        )
        E_know_apx_samp = np.array([
            [know_samp.evaluate_all()[a]["v_apx"][c] for c in CTXS]
            for a in ATTRS
        ])
        sse_samp = np.zeros(len(p_vals))
        for i, p in enumerate(p_vals):
            base_samp = build_scenario(
                omega=samp["omega"],
                priors_K={"k_p": float(p), "k_np": float(1 - p)},
                priors_M=samp["priors_M"],
                inference=inference,
            )
            E_base_apx_samp = np.array([
                [base_samp.evaluate_all()[a]["v_apx"][c] for c in CTXS]
                for a in ATTRS
            ])
            sse_samp[i] = np.sum((E_know_apx_samp - E_base_apx_samp - obs_apx) ** 2)
        p0_sampled.append(round(float(p_vals[int(np.argmin(sse_samp))]), 2))

    return {
        "p0_primary":   p0_primary,
        "p0_secondary": p0_secondary,
        "bootstrap_ci": {"lo": ci_lo, "hi": ci_hi, "n": n_bootstrap},
        "boot_p0_mean": round(float(boot_p0_arr.mean()), 3),
        "boot_p0_sd":   round(float(boot_p0_arr.std()), 3),
        "p0_sampled_100": p0_sampled,
        "p0_sampled_mean": round(float(np.mean(p0_sampled)), 3),
        "p0_sampled_sd":   round(float(np.std(p0_sampled)), 3),
        "sse_curve": {
            "p_vals":  [round(float(p), 2) for p in p_vals],
            "sse_primary": [round(float(x), 6) for x in sse_by_p],
        },
    }


# ---------------------------------------------------------------------------
# Pedantry risk
# ---------------------------------------------------------------------------

def pedantry_risk(e1: pd.DataFrame, e2: pd.DataFrame, delta_ci: Dict) -> Dict:
    """
    SEM predicts zero change for precise cells.
    Test: Exp.2 − Exp.1 on pedantic rating for precise form per context.
    """
    out: Dict = {}
    raw_col = "pedantic"
    for ctx_val, c in CTX_MAP.items():
        c1 = e1[(e1.form == "precise") & (e1.context == ctx_val)][raw_col]
        c2 = e2[(e2.form == "precise") & (e2.context == ctx_val)][raw_col]
        t_stat, p_val = sp_stats.ttest_ind(c2, c1, equal_var=False)
        m1 = normalize_likert(float(c1.mean()))
        m2 = normalize_likert(float(c2.mean()))
        out[c] = {
            "exp1_norm": round(m1, 4),
            "exp2_norm": round(m2, 4),
            "delta_obs": delta_ci["a_ped"]["v_prc"][c]["delta"],
            "delta_ci_lo": delta_ci["a_ped"]["v_prc"][c]["ci_lo"],
            "delta_ci_hi": delta_ci["a_ped"]["v_prc"][c]["ci_hi"],
            "t_stat": round(float(t_stat), 3),
            "p_value": round(float(p_val), 4),
            "sem_prediction": 0.0,
        }
    # Pooled (both contexts combined)
    c1_all = e1[e1.form == "precise"][raw_col]
    c2_all = e2[e2.form == "precise"][raw_col]
    t_stat_all, p_val_all = sp_stats.ttest_ind(c2_all, c1_all, equal_var=False)
    m1_all = normalize_likert(float(c1_all.mean()))
    m2_all = normalize_likert(float(c2_all.mean()))
    out["pooled"] = {
        "exp1_norm": round(m1_all, 4),
        "exp2_norm": round(m2_all, 4),
        "delta_obs": round(m2_all - m1_all, 4),
        "t_stat":   round(float(t_stat_all), 3),
        "p_value":  round(float(p_val_all), 4),
        "sem_prediction": 0.0,
    }
    return out


# ---------------------------------------------------------------------------
# Figure: Step 5 SSE curve
# ---------------------------------------------------------------------------

def save_step5_figure(s5: Dict, s5_joint: Dict, obs_apx: np.ndarray,
                      out_path: str = "figures/step5_prior_sweep.pdf") -> None:
    """Plot SSE(P0) for staged and joint listeners, mark P0*, overlay observed Δ."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    p_vals = s5["sse_curve"]["p_vals"]
    sse_st = s5["sse_curve"]["sse_primary"]
    sse_jt = s5_joint["sse_curve"]["sse_primary"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(p_vals, sse_st, label="staged", color="steelblue")
    ax.plot(p_vals, sse_jt, label="joint",  color="darkorange", linestyle="--")
    ax.axvline(s5["p0_primary"],       color="steelblue", linestyle=":", alpha=0.7,
               label=f"P0*(staged)={s5['p0_primary']}")
    ax.axvline(s5_joint["p0_primary"], color="darkorange", linestyle=":", alpha=0.7,
               label=f"P0*(joint)={s5_joint['p0_primary']}")
    ax.set_xlabel("Base P(k_p)")
    ax.set_ylabel("SSE (6 approximate cells)")
    ax.set_title("Step 5: Base-prior sweep")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved to {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pstar(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return "n.s."


def write_report(results: Dict, path: str) -> None:
    lines = []
    def h(title): lines.append(f"\n## {title}\n")
    def row(*cols): lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    def sep(n): lines.append("| " + " | ".join(["---"] * n) + " |")

    lines.append("# Session F Results\n")
    lines.append("Generated by `scripts/exp2_knowledge_test.py`. "
                 "All model scores ∈ [−1,1] (normalized Likert).\n")

    # Q1: Manipulation check
    h("Q1a: Manipulation Check (Task 3, approximate form)")
    mc = results["q1_manipulation_check"]
    row("measure", "Exp. 1", "Exp. 2", "direction")
    sep(4)
    row("t3_didntKnow",
        mc["didnt_know"]["exp1_approx"],
        mc["didnt_know"]["exp2_approx"],
        "↓ (expected)" if mc["didnt_know"]["exp2_approx"] < mc["didnt_know"]["exp1_approx"] else "↑ (unexpected)")
    row("t3_easierSpeaker",
        mc["easier_speaker"]["exp1_approx"],
        mc["easier_speaker"]["exp2_approx"],
        "↑ (expected)" if mc["easier_speaker"]["exp2_approx"] > mc["easier_speaker"]["exp1_approx"] else "↓ (unexpected)")
    lines.append("\n*Author (2025): didntKnow 70% → 42%, easierSpeaker 33% → 55%.*\n")

    # Q1: Exp.2 cell means and delta
    h("Q1b: Cell Means and Exp.2 − Exp.1 Differences")
    row("attr", "form", "context", "E1 (norm)", "E2 (norm)", "Δ", "95% CI")
    sep(7)
    for a in ATTRS:
        for v in FORMS:
            for c in CTXS:
                d = results["q1_descriptives"]
                e1v = d["exp1"][a][v][c]["mean_norm"]
                e2v = d["exp2"][a][v][c]["mean_norm"]
                delta_d = results["q1_delta_ci"][a][v][c]
                lo, hi = delta_d["ci_lo"], delta_d["ci_hi"]
                ci_str = f"[{lo}, {hi}]"
                row(a, v, c, e1v, e2v, delta_d["delta"], ci_str)

    # Q1: four effects in Exp. 2
    h("Q1c: Four Target Effects in Exp. 2 Alone (from cell means)")
    ef = results["q1_exp2_effects_alone"]
    row("effect", "Exp. 2 sign", "value", "matches Exp. 1?")
    sep(4)
    row("comp main (precise > approx)", "+" if ef["comp_main_positive"] else "−",
        ef["comp_main_effect"], "✓" if ef["comp_main_positive"] else "✗")
    row("ped main (precise > approx)", "+" if ef["ped_main_positive"] else "−",
        ef["ped_main_effect"], "✓" if ef["ped_main_positive"] else "✗")
    row("comp interaction (gap larger c_HP)", "+" if ef["comp_inter_positive"] else "−",
        ef["comp_interaction"], "✓" if ef["comp_inter_positive"] else "✗")
    row("like interaction (gap larger c_HP)", "+" if ef["like_inter_positive"] else "−",
        ef["like_interaction"], "✓" if ef["like_inter_positive"] else "✗")

    # Q1: Mixed model
    h("Q1d: Mixed Model  rating ~ form * context * knowledge + (1|scenario)  [lmerTest, sum-coded]")
    if "error" in results.get("q1_mixed_model", {}):
        lines.append(f"\n*R error: {results['q1_mixed_model']['error']}*\n")
    else:
        for a_raw, a_sem in [("competent", "a_comp"), ("likeable", "a_like"), ("pedantic", "a_ped")]:
            if a_raw not in results["q1_mixed_model"]:
                continue
            lines.append(f"\n**{a_raw}**\n")
            row("term", "β", "SE", "t", "p", "")
            sep(6)
            for entry in results["q1_mixed_model"][a_raw]:
                row(entry["term"], entry["coef"], entry["se"], entry["t"],
                    entry["p"], _pstar(entry["p"]))

    # Q2: Calibration flatness
    h("Q2: Calibration Flatness across P(k_p)  [ω=.5, flat P(m); Exp. 1 data]")
    row("P(k_p)", "DAS", "ISS", "RMSE", "CDS", "CDS_main", "CDS_inter")
    sep(7)
    for r in results["q2_calibration"]["balanced"]:
        row(r["pk_p"], r["das"], r["iss"], r["rmse"], r["cds"],
            r["cds_main"], r["cds_interaction"])
    lines.append("\n*(sampled mean ± SD over 100 combinations in JSON output)*\n")

    # D1 Step 1
    h("D1 Step 1: Structural Prediction  P(k_p|v_prc) = 1 regardless of prior")
    s1 = results["d1_step1_staged"]
    lines.append(f"\nSatisfied: {s1['all_precise_cells_P_kp_equals_1']} "
                 f"(staged); {results['d1_step1_joint']['all_precise_cells_P_kp_equals_1']} (joint).\n")
    lines.append("Precise cells identical in Exp. 1 and Exp. 2 is the direct test.\n")

    # D1 Step 2
    h("D1 Step 2: Predicted Δ = E_knowledge − E_base  [staged; P(k_p): .5 → 1.0]")
    row("attr", "form", "context", "E_base", "E_knowledge", "Δ_predicted")
    sep(6)
    s2 = results["d1_step2_staged"]
    for a in ATTRS:
        for v in FORMS:
            for c in CTXS:
                ce = s2[a][v][c]
                row(a, v, c, ce["E_base"], ce["E_knowledge"], ce["delta_predicted"])

    # D1 Step 3
    h("D1 Step 3: Motivation Shift  P(m|v_apx, c)  [staged]")
    s3 = results["d1_step3_staged"]
    row("condition", "m_max", "m_sit", "m_min")
    sep(4)
    for cond_label, cond_key in [("base c_HP", "v_apx_c_HP"), ("base c_LP", "v_apx_c_LP"),
                                  ("know c_HP", "v_apx_c_HP"), ("know c_LP", "v_apx_c_LP")]:
        model_key = "base" if "base" in cond_label else "knowledge"
        pm = s3[model_key][cond_key]
        row(f"{model_key} {cond_key}", pm.get("m_max"), pm.get("m_sit"), pm.get("m_min"))

    # D1 Step 4
    h("D1 Step 4: Model vs Empirical Δ  [staged; base P(k_p)=.5]")
    row("attr", "form", "context", "Δ_obs", "Δ_pred", "sign agree")
    sep(6)
    s4 = results["d1_step4_staged"]
    for a in ATTRS:
        for v in FORMS:
            for c in CTXS:
                ce = s4[a][v][c]
                agree_str = "✓" if ce["sign_agree"] else ("✗" if ce["sign_agree"] is False else "–")
                row(a, v, c, ce["observed_delta"], ce["predicted_delta"], agree_str)
    sm = s4["_summary"]
    lines.append(f"\n*Approx cells: {sm['approx_cells_sign_agree']} agree, "
                 f"{sm['approx_cells_sign_disagree']} disagree, "
                 f"{sm['approx_cells_obs_zero']} obs≈0.*\n")

    # D1 Step 5
    h("D1 Step 5: Base-Prior Sweep — Primary Analysis")
    s5_st = results["d1_step5_staged"]
    s5_jt = results["d1_step5_joint"]
    row("listener", "P0* (primary)", "P0* (secondary)", "95% boot CI", "sampled mean ± SD")
    sep(5)
    row("staged",
        s5_st["p0_primary"], s5_st["p0_secondary"],
        f"[{s5_st['bootstrap_ci']['lo']}, {s5_st['bootstrap_ci']['hi']}]",
        f"{s5_st['p0_sampled_mean']} ± {s5_st['p0_sampled_sd']}")
    row("joint",
        s5_jt["p0_primary"], s5_jt["p0_secondary"],
        f"[{s5_jt['bootstrap_ci']['lo']}, {s5_jt['bootstrap_ci']['hi']}]",
        f"{s5_jt['p0_sampled_mean']} ± {s5_jt['p0_sampled_sd']}")
    lines.append("\n**Interpretation:** P0* is the base prior most consistent with "
                 "the observed near-null Exp.2 − Exp.1 shifts.\n")
    lines.append("> **Roland's judgment needed:** how to frame P0* relative to the grid-search "
                 "optimum (§4.2 footnote); whether the CI is tight enough to make a strong claim.\n")

    # Pedantry risk
    h("Pedantry Risk  [SEM predicts Δ = 0 for precise form]")
    row("context", "Δ_obs (norm)", "95% CI", "t", "p", "misprediction?")
    sep(6)
    ped = results["pedantry_risk"]
    for c_label, c_key in [("c_HP", "c_HP"), ("c_LP", "c_LP"), ("pooled", "pooled")]:
        ce = ped[c_key]
        if c_key != "pooled":
            lo = results["q1_delta_ci"]["a_ped"]["v_prc"][c_key]["ci_lo"]
            hi = results["q1_delta_ci"]["a_ped"]["v_prc"][c_key]["ci_hi"]
            ci_str = f"[{lo}, {hi}]"
        else:
            ci_str = "—"
        mispred = "YES" if (ce["delta_obs"] < 0 and ce["p_value"] < 0.05) else (
                  "marginal" if (ce["delta_obs"] < 0 and ce["p_value"] < 0.10) else "no")
        row(c_label, ce["delta_obs"], ci_str, ce["t_stat"], ce["p_value"], mispred)
    lines.append(
        "\n*Author (2025) reports lower pedantry in knowledge condition for both forms; "
        "SEM predicts zero change for precise form.*\n"
    )
    lines.append("> **Roland's judgment needed:** whether the precise-cell pedantry difference "
                 "is the ELM effect or an artefact; how to frame it in §4 / §5.4.\n")

    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Report written to {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    e1 = pd.read_csv(E1_PATH)
    e2 = pd.read_csv(E2_PATH)
    e1_means = load_empirical_means(E1_MEANS)
    samples_100 = sample_parameters("full", n_samples=100, seed=42)

    print("Q1: descriptives, manipulation check, delta CI...")
    q1_desc   = q1_descriptives(e1, e2)
    q1_manip  = q1_manipulation_check(e1, e2)
    q1_delta  = q1_delta_ci(e1, e2)
    q1_exp2   = q1_exp2_effects_alone(e2)

    print("Q1: mixed model (lmerTest)...")
    q1_mm = q1_mixed_model(e1, e2)

    print("Q2: calibration flatness...")
    q2 = q2_calibration(e1_means, samples_100)

    print("D1 Step 1: structural prediction (staged + joint)...")
    s1_st = d1_step1(inference="staged")
    s1_jt = d1_step1(inference="joint")

    print("D1 Step 2: two models (staged + joint)...")
    s2_st = d1_step2(e1_means, inference="staged")
    s2_jt = d1_step2(e1_means, inference="joint")

    print("D1 Step 3: motivation shift (staged + joint)...")
    s3_st = d1_step3(inference="staged")
    s3_jt = d1_step3(inference="joint")

    print("D1 Step 4: model vs data (staged + joint)...")
    s4_st = d1_step4(q1_delta, s2_st)
    s4_jt = d1_step4(q1_delta, s2_jt)

    print("D1 Step 5: base-prior sweep + bootstrap (staged)...")
    s5_st = d1_step5(e1, e2, q1_delta, n_bootstrap=2000, seed=42, inference="staged")
    print(f"  staged P0*={s5_st['p0_primary']}, CI=[{s5_st['bootstrap_ci']['lo']}, {s5_st['bootstrap_ci']['hi']}]")

    print("D1 Step 5: base-prior sweep + bootstrap (joint)...")
    s5_jt = d1_step5(e1, e2, q1_delta, n_bootstrap=2000, seed=42, inference="joint")
    print(f"  joint  P0*={s5_jt['p0_primary']}, CI=[{s5_jt['bootstrap_ci']['lo']}, {s5_jt['bootstrap_ci']['hi']}]")

    print("Pedantry risk...")
    ped = pedantry_risk(e1, e2, q1_delta)

    results = {
        "q1_descriptives":       q1_desc,
        "q1_manipulation_check": q1_manip,
        "q1_delta_ci":           q1_delta,
        "q1_exp2_effects_alone": q1_exp2,
        "q1_mixed_model":        q1_mm,
        "q2_calibration":        q2,
        "d1_step1_staged":       s1_st,
        "d1_step1_joint":        s1_jt,
        "d1_step2_staged":       s2_st,
        "d1_step2_joint":        s2_jt,
        "d1_step3_staged":       s3_st,
        "d1_step3_joint":        s3_jt,
        "d1_step4_staged":       s4_st,
        "d1_step4_joint":        s4_jt,
        "d1_step5_staged":       s5_st,
        "d1_step5_joint":        s5_jt,
        "pedantry_risk":         ped,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results JSON written to {OUT_JSON}")

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    write_report(results, OUT_REPORT)

    print("Saving Step 5 figure...")
    obs_apx_arr = np.array([
        [q1_delta[a]["v_apx"][c]["delta"] for c in CTXS]
        for a in ATTRS
    ])
    save_step5_figure(s5_st, s5_jt, obs_apx_arr)


if __name__ == "__main__":
    main()
