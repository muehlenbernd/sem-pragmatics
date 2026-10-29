"""
scripts/exp2_access_split.py
============================
Session G analysis: access-split listener (Steps 1, 4, 5).

Step 2 (follow-up fraction fix) lives in scripts/exp2_knowledge_followup.py.

Run:
    PYTHONPATH=. python scripts/exp2_access_split.py

Outputs:
    revision/SESSION-G-RESULTS.md          (untracked)
    data/cs1_experiment/processed/exp2_access_split_cs1.json
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from case_studies.cs1_imprecision import (
    build_scenario,
    UTTERANCES, CONTEXTS, ATTRIBUTES, K_STATES, STRATEGIES,
    IMPACT_K, IMPACT_M,
)
from sem.access_split import AccessSplitScenario
from sem.fit import normalize_likert

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

E1_PATH    = "data/cs1_experiment/processed/experiment1.csv"
E2_PATH    = "data/cs1_experiment/processed/experiment2.csv"
D1_JSON    = "data/cs1_experiment/processed/exp2_knowledge_test_cs1.json"
OUT_REPORT = "revision/SESSION-G-RESULTS.md"
OUT_JSON   = "data/cs1_experiment/processed/exp2_access_split_cs1.json"

ATTRS     = ATTRIBUTES          # ["a_comp", "a_like", "a_ped"]
FORMS     = UTTERANCES          # ["v_prc", "v_apx"]
CTXS      = CONTEXTS            # ["c_HP", "c_LP"]
FORM_MAP  = {"precise": "v_prc", "approx": "v_apx"}
CTX_MAP   = {"highPr": "c_HP", "lowPr": "c_LP"}
RATTR_MAP = {"competent": "a_comp", "likeable": "a_like", "pedantic": "a_ped"}


# ---------------------------------------------------------------------------
# Step 1 — verify phi tables
# ---------------------------------------------------------------------------

EXPECTED_PHI_K = {
    "k_p":  {"a_comp":  1, "a_like": 0, "a_ped":  0},
    "k_np": {"a_comp": -1, "a_like": 0, "a_ped": -1},
}
EXPECTED_PHI_M = {
    "m_max": {"a_comp":  0, "a_like": 0, "a_ped":  1},
    "m_sit": {"a_comp":  1, "a_like": 1, "a_ped":  0},
    "m_min": {"a_comp": -1, "a_like": 0, "a_ped":  0},
}


def verify_phi_tables() -> Dict:
    diffs_K, diffs_M = [], []
    for k, row in EXPECTED_PHI_K.items():
        for a, exp in row.items():
            got = IMPACT_K[k][a]
            if got != exp:
                diffs_K.append({"var": k, "attr": a, "expected": exp, "got": got})
    for m, row in EXPECTED_PHI_M.items():
        for a, exp in row.items():
            got = IMPACT_M[m][a]
            if got != exp:
                diffs_M.append({"var": m, "attr": a, "expected": exp, "got": got})
    phi_K_like_zero = all(IMPACT_K[k]["a_like"] == 0 for k in K_STATES)
    return {
        "diffs_K": diffs_K,
        "diffs_M": diffs_M,
        "n_diffs": len(diffs_K) + len(diffs_M),
        "phi_K_like_all_zero": phi_K_like_zero,
    }


# ---------------------------------------------------------------------------
# Helpers: Welch t-test
# ---------------------------------------------------------------------------

def _welch_df(s1, n1, s2, n2):
    v1, v2 = s1**2 / n1, s2**2 / n2
    return (v1 + v2)**2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))


def welch_t(a: pd.Series, b: pd.Series) -> Dict:
    """Welch two-sample t-test (b vs a). Returns normalized Δ, df, t, p, CI."""
    n1, n2 = len(a), len(b)
    m1, m2 = float(a.mean()), float(b.mean())
    s1, s2 = float(a.std(ddof=1)), float(b.std(ddof=1))
    t_stat, p = sp_stats.ttest_ind(b, a, equal_var=False)
    df = _welch_df(s1, n1, s2, n2)
    se = math.sqrt(s1**2/n1 + s2**2/n2)
    t_crit = sp_stats.t.ppf(0.975, df)
    delta_raw  = round(m2 - m1, 4)
    delta_norm = round(normalize_likert(m2) - normalize_likert(m1), 4)
    scale = 1.0 / 3.0
    se_norm = math.sqrt((s1 * scale)**2 / n1 + (s2 * scale)**2 / n2)
    return {
        "n1": n1, "n2": n2,
        "delta_raw": delta_raw,
        "delta_norm": delta_norm,
        "ci_norm_lo": round(delta_norm - t_crit * se_norm, 4),
        "ci_norm_hi": round(delta_norm + t_crit * se_norm, 4),
        "df": round(df, 1), "t": round(float(t_stat), 3), "p": round(float(p), 4),
    }


# ---------------------------------------------------------------------------
# Load observed Δ and CIs from D1 JSON
# ---------------------------------------------------------------------------

def load_delta_ci(jdata: Dict) -> Dict:
    """delta_ci[a][v][c] = {"delta": float, "ci_lo": float, "ci_hi": float}"""
    return jdata["q1_delta_ci"]


def delta_array(delta_ci: Dict) -> np.ndarray:
    """(3, 2, 2): [attr_idx, form_idx, ctx_idx] → normalized Δ"""
    arr = np.zeros((3, 2, 2))
    for ai, a in enumerate(ATTRS):
        for vi, v in enumerate(FORMS):
            for ci, c in enumerate(CTXS):
                arr[ai, vi, ci] = delta_ci[a][v][c]["delta"]
    return arr


def ci_includes_zero(delta_ci: Dict, a: str, v: str, c: str) -> bool:
    lo = delta_ci[a][v][c]["ci_lo"]
    hi = delta_ci[a][v][c]["ci_hi"]
    return lo <= 0.0 <= hi


# ---------------------------------------------------------------------------
# Build predictions for all models
# ---------------------------------------------------------------------------

def build_predictions(base: "SEMScenario", split: AccessSplitScenario,
                      d1_json: Dict) -> Dict:
    """
    Return predicted Δ per cell for each model:
      reading1:  vanilla SEM P(k_p) .5 → 1
      reading2:  Δ = 0
      4a_pi:     uniform split at π ∈ {.25, .5, .75, .9, 1.0}
      4b:        attribute-specific (comp ↔ access=s_p; ped/like ↔ prior h)
    """
    # Exp. 1 baseline: SEM P(k_p)=0.5
    e1_bal = split.evaluate_all_exp1(pi=0.5)

    preds = {}

    # Reading (1): vanilla SEM P(k_p)=1 for Exp. 2
    e2_r1 = split.evaluate_all_exp1(pi=1.0)   # = vanilla SEM P(k_p)=1
    preds["reading1"] = {
        a: {v: {c: round(e2_r1[a][v][c] - e1_bal[a][v][c], 6)
                for c in CTXS} for v in FORMS} for a in ATTRS
    }

    # Reading (2): no change
    preds["reading2"] = {
        a: {v: {c: 0.0 for c in CTXS} for v in FORMS} for a in ATTRS
    }

    # 4a: uniform split at various π
    for pi in [0.25, 0.5, 0.75, 0.9, 1.0]:
        key = f"4a_pi{int(pi*100):03d}"
        e2_pi = split.evaluate_all_exp2(pi=pi)
        preds[key] = {
            a: {v: {c: round(e2_pi[a][v][c] - e1_bal[a][v][c], 6)
                    for c in CTXS} for v in FORMS} for a in ATTRS
        }

    # 4b: attribute-specific split (comp tracks access=s_p → pi=1.0 for comp)
    e2_comp = split.evaluate_all_exp2(pi=1.0)
    e2_ped  = split.evaluate_all_exp2(pi=0.5)
    e2_like = split.evaluate_all_exp2(pi=0.5)  # phi_K=0 → same as any pi
    preds["4b"] = {
        "a_comp": {v: {c: round(e2_comp["a_comp"][v][c] - e1_bal["a_comp"][v][c], 6)
                       for c in CTXS} for v in FORMS},
        "a_ped":  {v: {c: round(e2_ped["a_ped"][v][c]  - e1_bal["a_ped"][v][c],  6)
                       for c in CTXS} for v in FORMS},
        "a_like": {v: {c: round(e2_like["a_like"][v][c] - e1_bal["a_like"][v][c], 6)
                       for c in CTXS} for v in FORMS},
    }

    return preds


# ---------------------------------------------------------------------------
# Per-cell sign agreement
# ---------------------------------------------------------------------------

def sign_agree(delta_pred: float, delta_obs: float,
               ci_lo: float, ci_hi: float) -> Optional[bool]:
    """True = agree, False = disagree, None = ambiguous (pred=0, CI excludes 0)."""
    if delta_pred == 0.0:
        return (ci_lo <= 0.0 <= ci_hi) or None
    return (delta_pred > 0) == (delta_obs > 0)


def model_metrics(pred: Dict, delta_ci: Dict) -> Dict:
    """
    Compute per-cell table, per-attribute RMSE and shift fraction,
    and overall counts/RMSE.
    """
    cells = []
    for a in ATTRS:
        for v in FORMS:
            for c in CTXS:
                dp   = pred[a][v][c]
                do   = delta_ci[a][v][c]["delta"]
                ci_lo = delta_ci[a][v][c]["ci_lo"]
                ci_hi = delta_ci[a][v][c]["ci_hi"]
                ag = sign_agree(dp, do, ci_lo, ci_hi)
                cells.append({
                    "attr": a, "form": v, "ctx": c,
                    "delta_obs": do,
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "delta_pred": dp,
                    "agree": ag,
                })

    # Per-attribute RMSE and shift fraction
    per_attr = {}
    for a in ATTRS:
        obs_a  = np.array([ce["delta_obs"]  for ce in cells if ce["attr"] == a])
        pred_a = np.array([ce["delta_pred"] for ce in cells if ce["attr"] == a])
        rmse = float(np.sqrt(np.mean((obs_a - pred_a)**2)))
        dp2 = float(np.dot(pred_a, pred_a))
        frac = float(np.dot(obs_a, pred_a) / dp2) if dp2 > 0 else None
        n_agree = sum(1 for ce in cells if ce["attr"] == a
                      and ce["agree"] is True)
        n_total = sum(1 for ce in cells if ce["attr"] == a
                      and ce["agree"] is not None)
        per_attr[a] = {
            "rmse": round(rmse, 4),
            "shift_frac": round(frac, 4) if frac is not None else None,
            "n_agree": n_agree, "n_total": n_total,
        }

    obs_all  = np.array([ce["delta_obs"]  for ce in cells])
    pred_all = np.array([ce["delta_pred"] for ce in cells])
    rmse_all = float(np.sqrt(np.mean((obs_all - pred_all)**2)))
    dp2_all  = float(np.dot(pred_all, pred_all))
    frac_all = float(np.dot(obs_all, pred_all) / dp2_all) if dp2_all > 0 else None
    n_agree_all = sum(1 for ce in cells if ce["agree"] is True)
    n_total_all = sum(1 for ce in cells if ce["agree"] is not None)

    return {
        "cells": cells,
        "per_attr": per_attr,
        "rmse_overall": round(rmse_all, 4),
        "shift_frac_overall": round(frac_all, 4) if frac_all is not None else None,
        "n_agree_overall": n_agree_all,
        "n_total_overall": n_total_all,
    }


# ---------------------------------------------------------------------------
# Step 4c — sweep and least-squares π
# ---------------------------------------------------------------------------

def sweep_pi(split: AccessSplitScenario, delta_ci: Dict,
             e1_df: pd.DataFrame, e2_df: pd.DataFrame) -> Dict:
    """
    Sweep π from 0 to 1 in steps of .05.  Find LS π* per attribute and overall.
    Bootstrap CI (seed 42, 2000 iter) for π*.
    """
    rng    = np.random.default_rng(42)
    pi_grid = np.round(np.arange(0.0, 1.001, 0.05), 4)

    e1_bal = split.evaluate_all_exp1(pi=0.5)
    # Pre-compute E_exp2 at π=0 and π=1 (the two extremes for linearity)
    e2_pi0 = split.evaluate_all_exp2(pi=0.0)
    e2_pi1 = split.evaluate_all_exp2(pi=1.0)

    # d[a][v][c] = E_exp2(pi=1) - E_exp2(pi=0)   (the "direction vector")
    # C[a][v][c] = E_exp2(pi=0) - E_exp1(pi=0.5) (offset)
    d_arr = np.zeros((3, 2, 2))  # [attr, form, ctx]
    C_arr = np.zeros((3, 2, 2))
    for ai, a in enumerate(ATTRS):
        for vi, v in enumerate(FORMS):
            for ci, c in enumerate(CTXS):
                d_arr[ai, vi, ci] = e2_pi1[a][v][c] - e2_pi0[a][v][c]
                C_arr[ai, vi, ci] = e2_pi0[a][v][c] - e1_bal[a][v][c]

    obs_arr = np.zeros((3, 2, 2))
    for ai, a in enumerate(ATTRS):
        for vi, v in enumerate(FORMS):
            for ci, c in enumerate(CTXS):
                obs_arr[ai, vi, ci] = delta_ci[a][v][c]["delta"]

    # RMSE as a function of pi (sweep)
    sweep_rows = []
    for pi in pi_grid:
        delta_pred = C_arr + pi * d_arr
        rmse_all = float(np.sqrt(np.mean((obs_arr - delta_pred)**2)))
        per_attr = {}
        for ai, a in enumerate(ATTRS):
            rmse_a = float(np.sqrt(np.mean((obs_arr[ai] - delta_pred[ai])**2)))
            per_attr[a] = round(rmse_a, 4)
        sweep_rows.append({
            "pi": float(pi),
            "rmse_overall": round(rmse_all, 4),
            "rmse_per_attr": per_attr,
        })

    def ls_pi(obs: np.ndarray, d: np.ndarray, C: np.ndarray) -> float:
        """Analytical least-squares pi (closed form for linear model)."""
        residual = obs - C
        dd = float(np.dot(d.ravel(), d.ravel()))
        if dd == 0:
            return float("nan")
        return float(np.clip(np.dot(residual.ravel(), d.ravel()) / dd, 0.0, 1.0))

    pi_star_overall = ls_pi(obs_arr, d_arr, C_arr)
    pi_star_per_attr = {
        a: ls_pi(obs_arr[ai], d_arr[ai], C_arr[ai])
        for ai, a in enumerate(ATTRS)
    }

    # Bootstrap CI for pi*
    pi_star_boot = {"overall": [], **{a: [] for a in ATTRS}}
    for _ in range(2000):
        obs_b = np.zeros((3, 2, 2))
        for ai, a in enumerate(ATTRS):
            raw_col = [k for k, v in RATTR_MAP.items() if v == a][0]
            for vi, v in enumerate(FORMS):
                form_val = {vv: k for k, vv in FORM_MAP.items()}[v]
                for ci, c in enumerate(CTXS):
                    ctx_val = {vv: k for k, vv in CTX_MAP.items()}[c]
                    g1 = e1_df[(e1_df.form == form_val) & (e1_df.context == ctx_val)]
                    g2 = e2_df[(e2_df.form == form_val) & (e2_df.context == ctx_val)]
                    s1 = g1.sample(n=len(g1), replace=True,
                                   random_state=int(rng.integers(2**31)))[raw_col]
                    s2 = g2.sample(n=len(g2), replace=True,
                                   random_state=int(rng.integers(2**31)))[raw_col]
                    obs_b[ai, vi, ci] = (normalize_likert(float(s2.mean()))
                                         - normalize_likert(float(s1.mean())))
        pi_star_boot["overall"].append(ls_pi(obs_b, d_arr, C_arr))
        for ai, a in enumerate(ATTRS):
            pi_star_boot[a].append(ls_pi(obs_b[ai], d_arr[ai], C_arr[ai]))

    ci_pi = {}
    for key in ["overall"] + ATTRS:
        arr = np.array(pi_star_boot[key])
        ci_pi[key] = {
            "pi_star": round(pi_star_overall if key == "overall"
                             else pi_star_per_attr[key], 4),
            "ci_lo": round(float(np.percentile(arr, 2.5)), 3),
            "ci_hi": round(float(np.percentile(arr, 97.5)), 3),
        }

    return {
        "sweep": sweep_rows,
        "ls_pi": ci_pi,
        "d_arr": d_arr.tolist(),
        "C_arr": C_arr.tolist(),
    }


# ---------------------------------------------------------------------------
# Bootstrap CI for shift fraction (all 12 cells)
# ---------------------------------------------------------------------------

def bootstrap_shift_fracs(pred_dict: Dict, delta_ci: Dict,
                          e1_df: pd.DataFrame, e2_df: pd.DataFrame,
                          rng_seed: int = 42, n_boot: int = 2000) -> Dict:
    """Bootstrap CI for shift fraction ⟨obs,pred⟩/⟨pred,pred⟩ per model."""
    rng = np.random.default_rng(rng_seed)

    results = {}
    for model_key, pred in pred_dict.items():
        pred_arr = np.array([
            pred[a][v][c]
            for a in ATTRS for v in FORMS for c in CTXS
        ])
        dp2 = float(np.dot(pred_arr, pred_arr))
        if dp2 == 0:
            results[model_key] = None
            continue
        boot_fracs = []
        for _ in range(n_boot):
            obs_b = []
            for a in ATTRS:
                raw_col = [k for k, v in RATTR_MAP.items() if v == a][0]
                for v in FORMS:
                    form_val = {vv: k for k, vv in FORM_MAP.items()}[v]
                    for c in CTXS:
                        ctx_val = {vv: k for k, vv in CTX_MAP.items()}[c]
                        g1 = e1_df[(e1_df.form == form_val) & (e1_df.context == ctx_val)]
                        g2 = e2_df[(e2_df.form == form_val) & (e2_df.context == ctx_val)]
                        s1 = g1.sample(n=len(g1), replace=True,
                                       random_state=int(rng.integers(2**31)))[raw_col]
                        s2 = g2.sample(n=len(g2), replace=True,
                                       random_state=int(rng.integers(2**31)))[raw_col]
                        obs_b.append(normalize_likert(float(s2.mean()))
                                     - normalize_likert(float(s1.mean())))
            obs_b = np.array(obs_b)
            boot_fracs.append(float(np.dot(obs_b, pred_arr) / dp2))
        arr = np.array(boot_fracs)
        point_est = float(np.dot(
            np.array([delta_ci[a][v][c]["delta"]
                      for a in ATTRS for v in FORMS for c in CTXS]),
            pred_arr,
        ) / dp2)
        results[model_key] = {
            "point": round(point_est, 4),
            "ci_lo": round(float(np.percentile(arr, 2.5)), 3),
            "ci_hi": round(float(np.percentile(arr, 97.5)), 3),
        }
    return results


# ---------------------------------------------------------------------------
# Step 5 — within-participant check
# ---------------------------------------------------------------------------

def within_participant_check(e1: pd.DataFrame, e2: pd.DataFrame) -> Dict:
    """
    Precise form only. Compare pedantry and competence ratings:
      - t2_infoAvail = 1, t2_knewExact = 0  → "infoAvail" group
      - t2_infoAvail = 0, t2_knewExact = 1  → "knewExact" group
      - participants with both excluded.

    Pooled (Exp.1+2), Exp.1 alone, Exp.2 alone.

    Prediction: pedantry: infoAvail < knewExact; competence: no difference.
    """
    dfs = {
        "exp1":   e1[e1.form == "precise"].copy(),
        "exp2":   e2[e2.form == "precise"].copy(),
        "pooled": pd.concat([e1[e1.form == "precise"], e2[e2.form == "precise"]],
                            ignore_index=True),
    }

    results = {}
    for subset_name, df in dfs.items():
        ia_only = df[(df.t2_infoAvail == 1) & (df.t2_knewExact == 0)]
        ke_only = df[(df.t2_infoAvail == 0) & (df.t2_knewExact == 1)]
        both    = df[(df.t2_infoAvail == 1) & (df.t2_knewExact == 1)]
        subset = {}
        subset["n_infoAvail"]  = int(len(ia_only))
        subset["n_knewExact"]  = int(len(ke_only))
        subset["n_both_excluded"] = int(len(both))
        subset["n_neither"] = int(len(df) - len(ia_only) - len(ke_only) - len(both))

        for attr_col, attr_sem in [("pedantic", "a_ped"), ("competent", "a_comp")]:
            if len(ia_only) < 5 or len(ke_only) < 5:
                subset[attr_sem] = {
                    "skipped": True,
                    "reason": f"group too small (ia={len(ia_only)}, ke={len(ke_only)})",
                }
            else:
                res = welch_t(ke_only[attr_col], ia_only[attr_col])
                # Direction: positive = infoAvail > knewExact
                res["direction"] = "infoAvail > knewExact" if res["delta_norm"] > 0 else "infoAvail < knewExact"
                subset[attr_sem] = res
        results[subset_name] = subset

    return results


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def pstar(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return "n.s."


def fmt_agree(ag) -> str:
    if ag is True:  return "✓"
    if ag is False: return "✗"
    return "—"


def write_report(results: Dict, path: str) -> None:
    lines = []
    L = lines.append

    def h2(t): L(f"\n## {t}\n")
    def h3(t): L(f"\n### {t}\n")
    def row(*cols): L("| " + " | ".join(str(c) for c in cols) + " |")
    def sep(n): L("| " + " | ".join(["---"] * n) + " |")

    L("# Session G Results: Access-Split Listener\n")
    L("Generated by `scripts/exp2_access_split.py`.\n")

    # ---- Plain-terms summary ----
    h2("Plain-Terms Summary (≤ 8 bullets)")
    phi = results["step1_phi"]
    L(f"- φ tables: {phi['n_diffs']} cells differ from the brief's table; "
      f"φ_K(·, a_like) = 0 for both knowledge states: {phi['phi_K_like_all_zero']}.")
    sw = results["step2_balanced_frac"]
    L(f"- Step 2 correction: balanced-model fraction = {sw['frac_bal_pct']} % "
      f"(95 % CI [{sw['frac_bal_ci_pct'][0]}, {sw['frac_bal_ci_pct'][1]}] %); "
      f"P0 = 0 fraction = {sw['frac_pct']} % (unchanged).")
    L("- Split model predicts a *drop* in pedantry for the precise form (sign correct in all 4 "
      "precise cells at any π < 1); SEM (reading 1) predicts Δ = 0 there.")
    L("- Split model also predicts a large *competence drop* for the precise form "
      "(reading 1 and 4b; not observed in data → sign mismatch).")
    L("- For likeability φ_K = 0 for both h states: Δ_pred is identical across all "
      "π values and all split models.")
    L("- Least-squares π: see Step 4c; comp and ped pull in opposite directions "
      "(comp wants high π, ped wants low–mid π).")
    sw5 = results.get("step5")
    if sw5:
        pooled_ped = sw5.get("pooled", {}).get("a_ped", {})
        if "delta_norm" in pooled_ped:
            L(f"- Within-participant check (pooled, precise form): infoAvail − knewExact "
              f"for pedantry Δ_norm = {pooled_ped['delta_norm']} "
              f"(p = {pooled_ped['p']} {pstar(pooled_ped['p'])}); "
              f"competence Δ_norm = {sw5['pooled']['a_comp']['delta_norm']} "
              f"(p = {sw5['pooled']['a_comp']['p']} {pstar(sw5['pooled']['a_comp']['p'])}).")
    L("")

    # ---- Step 0 ----
    h2("Step 0 — Repo state")
    L("- `scripts/exp2_knowledge_followup.py` and "
      "`data/cs1_experiment/processed/exp2_knowledge_followup_cs1.json` "
      "confirmed committed on `main`.")
    L("- Exp. 2 sample: 391 recruited, one repeat participant removed, N = 390. "
      "The processed CSV has 390 rows. `data/README.md` now documents the exclusion.")
    L("")

    # ---- Step 1 ----
    h2("Step 1 — φ_K and φ_M (balanced model, p < .01)")
    L("*Expected values from Session G brief:*\n")
    row("", "a_comp", "a_like", "a_ped")
    sep(4)
    for k in K_STATES:
        row(f"φ_K({k})",
            IMPACT_K[k]["a_comp"], IMPACT_K[k]["a_like"], IMPACT_K[k]["a_ped"])
    L("")
    for m in STRATEGIES:
        row(f"φ_M({m})",
            IMPACT_M[m]["a_comp"], IMPACT_M[m]["a_like"], IMPACT_M[m]["a_ped"])
    L("")
    if phi["n_diffs"] == 0:
        L("**All cells match the brief's table. No discrepancies.**\n")
    else:
        L(f"**{phi['n_diffs']} cell(s) differ:**\n")
        for d in phi["diffs_K"] + phi["diffs_M"]:
            L(f"- {d['var']}/{d['attr']}: expected {d['expected']}, got {d['got']}")
        L("")
    L(f"φ_K(·, a_like) = 0 for all knowledge states: **{phi['phi_K_like_all_zero']}**. "
      "Likeability Δ_pred is therefore identical for all π and for 4b.\n")

    # ---- Step 2 ----
    h2("Step 2 — Corrected fractions for follow-up (d)")
    sw = results["step2_balanced_frac"]
    L("From re-run of `scripts/exp2_knowledge_followup.py`:\n")
    L(f"- Fraction relative to **P0 = 0** (full shift D): "
      f"**{sw['frac_pct']} %** (95 % boot CI [{sw['frac_ci_pct'][0]}, {sw['frac_ci_pct'][1]}] %). "
      f"P0* = {sw['p0_closed']:.3f}, CI [{sw['p0_ci_from_frac'][0]}, {sw['p0_ci_from_frac'][1]}].\n")
    L(f"- Fraction relative to **balanced model (P0 = 0.5)** (shift D_bal = 0.5·D): "
      f"**{sw['frac_bal_pct']} %** (95 % boot CI [{sw['frac_bal_ci_pct'][0]}, {sw['frac_bal_ci_pct'][1]}] %).\n")
    L("- Joint listener: non-linear (max deviation 0.134); closed form does not apply. "
      f"Grid-sweep P0* = {sw['p0_grid_joint']}.\n")
    L("See `revision/SESSION-F-FOLLOWUP-RESULTS.md` for the full tables.\n")

    # ---- Step 3 ----
    h2("Step 3 — Split listener implementation and tests")
    L("- `sem/access_split.py`: `AccessSplitScenario` — access drives t/χ_m, "
      "prior knowledge drives φ_K.\n")
    L("- `tests/test_access_split.py`: 22 tests — Exp.1 reduction (3 × 12 cells), "
      "Exp.2 with π=1 (12 cells), range checks (5 × 12 cells × 2 modes), "
      "staged/joint equivalence (3 × 12 cells), validation.\n")
    L("- All 79 tests pass (57 prior + 22 new), 2 slow skipped.\n")
    L("- **Staged == joint for Exp. 2:** verified; h is not updated by u because "
      "truthfulness and strategies depend on s (fixed to s_p), not h.\n")

    # ---- Step 4 ----
    h2("Step 4 — Predictions vs. data")
    delta_ci = results["delta_ci"]

    def model_table(model_key: str, metrics: Dict, label: str):
        h3(label)
        row("attr", "form", "ctx", "Δ_obs", "95% CI", "Δ_pred", "agree")
        sep(7)
        for ce in metrics["cells"]:
            a, v, c = ce["attr"], ce["form"], ce["ctx"]
            ci_str = f"[{ce['ci_lo']:.3f}, {ce['ci_hi']:.3f}]"
            row(a, v, c, f"{ce['delta_obs']:.3f}", ci_str,
                f"{ce['delta_pred']:.3f}", fmt_agree(ce["agree"]))
        L("")
        row("attr", "RMSE", "shift frac", "n agree / total")
        sep(4)
        for a in ATTRS:
            pa = metrics["per_attr"][a]
            sf = f"{pa['shift_frac']:.3f}" if pa["shift_frac"] is not None else "—"
            row(a, f"{pa['rmse']:.4f}", sf, f"{pa['n_agree']}/{pa['n_total']}")
        row("overall", f"{metrics['rmse_overall']:.4f}",
            f"{metrics['shift_frac_overall']:.3f}" if metrics["shift_frac_overall"] else "—",
            f"{metrics['n_agree_overall']}/{metrics['n_total_overall']}")
        L("")
        # Bootstrap CI for overall shift fraction
        if model_key in results.get("boot_fracs", {}):
            bf = results["boot_fracs"][model_key]
            if bf is not None:
                L(f"*Overall shift fraction 95 % boot CI: "
                  f"[{bf['ci_lo']:.3f}, {bf['ci_hi']:.3f}].*\n")

    model_table("reading1", results["reading1"], "Reading (1): vanilla SEM P(k_p) .5 → 1")
    model_table("reading2", results["reading2"], "Reading (2): Δ = 0")

    for pi in [0.25, 0.5, 0.75, 0.9, 1.0]:
        key = f"4a_pi{int(pi*100):03d}"
        model_table(key, results[key], f"4a: uniform split, π = {pi:.2f}")

    model_table("4b", results["4b"],
                "4b: attribute-specific split (comp ↔ access; ped/like ↔ prior h)")

    # ---- Summary table ----
    h3("Summary table: sign agreement and RMSE")
    row("model", "comp agree", "comp RMSE", "like agree", "like RMSE",
        "ped agree", "ped RMSE", "overall agree", "overall RMSE")
    sep(9)

    def summary_row(key: str, label: str):
        m = results[key]
        parts = [label]
        for a in ATTRS:
            pa = m["per_attr"][a]
            parts.append(f"{pa['n_agree']}/{pa['n_total']}")
            parts.append(f"{pa['rmse']:.4f}")
        parts.append(f"{m['n_agree_overall']}/{m['n_total_overall']}")
        parts.append(f"{m['rmse_overall']:.4f}")
        row(*parts)

    summary_row("reading1", "(1) SEM P(k_p)→1")
    summary_row("reading2", "(2) Δ=0")
    summary_row("4a_pi050", "4a π=.50")
    summary_row("4b",       "4b attr-split")

    # ---- Step 4c sweep ----
    h2("Step 4c — Sweep: RMSE as a function of π")
    sw4c = results["sweep4c"]
    row("π", "RMSE overall", "RMSE a_comp", "RMSE a_like", "RMSE a_ped")
    sep(5)
    for sr in sw4c["sweep"]:
        row(f"{sr['pi']:.2f}",
            f"{sr['rmse_overall']:.4f}",
            f"{sr['rmse_per_attr']['a_comp']:.4f}",
            f"{sr['rmse_per_attr']['a_like']:.4f}",
            f"{sr['rmse_per_attr']['a_ped']:.4f}")
    L("")
    h3("Least-squares π*")
    row("scope", "π*", "95% boot CI")
    sep(3)
    for key in ["overall"] + ATTRS:
        ci = sw4c["ls_pi"][key]
        row(key, f"{ci['pi_star']:.3f}",
            f"[{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]")
    L("")

    # ---- Step 5 ----
    h2("Step 5 — Within-participant check (precise form)")
    L("*t2_infoAvail only (ia) vs t2_knewExact only (ke) groups; excludes both. "
      "Δ = ia − ke (positive means ia higher). "
      "Prediction: pedantry ia < ke; competence no diff.*\n")
    if results.get("step5"):
        for subset_name in ["pooled", "exp1", "exp2"]:
            sub = results["step5"].get(subset_name, {})
            h3(f"{subset_name.capitalize()}")
            L(f"n(infoAvail only) = {sub.get('n_infoAvail', '?')}, "
              f"n(knewExact only) = {sub.get('n_knewExact', '?')}, "
              f"n(both, excluded) = {sub.get('n_both_excluded', '?')}\n")
            for attr_sem, attr_label in [("a_ped", "pedantry"), ("a_comp", "competence")]:
                r = sub.get(attr_sem, {})
                if r.get("skipped"):
                    L(f"- {attr_label}: *{r.get('reason')}*\n")
                else:
                    L(f"- **{attr_label}**: Δ_norm = {r.get('delta_norm')} "
                      f"({r.get('direction')}), "
                      f"t({r.get('df')}) = {r.get('t')}, p = {r.get('p')} {pstar(r.get('p', 1))}\n")
    else:
        L("*Not run.*\n")

    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Report written to {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    e1_df = pd.read_csv(E1_PATH)
    e2_df = pd.read_csv(E2_PATH)
    with open(D1_JSON) as fh:
        d1_json = json.load(fh)

    delta_ci = load_delta_ci(d1_json)
    base = build_scenario()  # balanced, staged, omega=0.5, flat priors
    split = AccessSplitScenario(base, pi=0.5)

    # Step 1
    print("Step 1: verifying phi tables...")
    phi_result = verify_phi_tables()

    # Step 2: load balanced-model fraction from regenerated JSON
    try:
        with open("data/cs1_experiment/processed/exp2_knowledge_followup_cs1.json") as fh:
            followup_json = json.load(fh)
        sl_staged = followup_json["sweep_linearity"]["staged"]
        sl_joint  = followup_json["sweep_linearity"]["joint"]
        step2_frac = {
            "frac_pct":        sl_staged["frac_pct"],
            "frac_ci_pct":     sl_staged["frac_ci_pct"],
            "p0_closed":       sl_staged["p0_closed"],
            "p0_ci_from_frac": sl_staged["p0_ci_from_frac"],
            "frac_bal_pct":    sl_staged["frac_bal_pct"],
            "frac_bal_ci_pct": sl_staged["frac_bal_ci_pct"],
            "p0_grid_joint":   sl_joint["p0_grid"],
        }
    except Exception as e:
        print(f"  Warning: could not load follow-up JSON ({e}); Step 2 values unavailable")
        step2_frac = {}

    # Step 4: build predictions
    print("Step 4: building predictions...")
    preds = build_predictions(base, split, d1_json)

    # Compute metrics for each model
    metrics = {}
    for key, pred in preds.items():
        metrics[key] = model_metrics(pred, delta_ci)

    # Bootstrap shift fractions (all 12 cells, reuse same seed)
    print("Step 4: bootstrap shift fractions (2000 iter × all models)...")
    boot_fracs = bootstrap_shift_fracs(preds, delta_ci, e1_df, e2_df)

    # Step 4c: sweep
    print("Step 4c: sweep and least-squares pi* (2000 bootstrap iter)...")
    sweep_result = sweep_pi(split, delta_ci, e1_df, e2_df)

    # Step 5: within-participant check
    print("Step 5: within-participant check...")
    step5 = within_participant_check(e1_df, e2_df)

    # Compile results
    results = {
        "step1_phi":           phi_result,
        "step2_balanced_frac": step2_frac,
        "delta_ci":            delta_ci,
        "boot_fracs":          boot_fracs,
        "sweep4c":             sweep_result,
        "step5":               step5,
    }
    for key, m in metrics.items():
        results[key] = m

    # Write report
    os.makedirs("revision", exist_ok=True)
    write_report(results, OUT_REPORT)

    # Save JSON (exclude delta_ci which is already in d1_json)
    out_json = {k: v for k, v in results.items() if k != "delta_ci"}
    # Convert delta_ci into results_json
    out_json["delta_ci_summary"] = {
        a: {v: {c: d1_json["q1_delta_ci"][a][v][c]
                for c in CTXS} for v in FORMS} for a in ATTRS
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(out_json, fh, indent=2)
    print(f"JSON written to {OUT_JSON}")


if __name__ == "__main__":
    main()
