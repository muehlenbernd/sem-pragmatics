"""
scripts/exp2_knowledge_followup.py
===================================
SESSION-F-FOLLOWUP tasks (a)–(d).

Run:
    PYTHONPATH=. python scripts/exp2_knowledge_followup.py

Outputs:
    revision/SESSION-F-FOLLOWUP-RESULTS.md   (untracked)
"""

import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import fisher_exact

from case_studies.cs1_imprecision import build_scenario
from sem.fit import normalize_likert, load_empirical_means
from sem.comparison import sample_parameters

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

E1_PATH    = "data/cs1_experiment/processed/experiment1.csv"
E2_PATH    = "data/cs1_experiment/processed/experiment2.csv"
JSON_PATH  = "data/cs1_experiment/processed/exp2_knowledge_test_cs1.json"
OUT_REPORT = "revision/SESSION-F-FOLLOWUP-RESULTS.md"

ATTRS    = ["a_comp", "a_like", "a_ped"]
FORMS    = ["v_prc", "v_apx"]
CTXS     = ["c_HP", "c_LP"]
FORM_MAP = {"precise": "v_prc", "approx": "v_apx"}
CTX_MAP  = {"highPr": "c_HP", "lowPr": "c_LP"}
RATTR_MAP = {"competent": "a_comp", "likeable": "a_like", "pedantic": "a_ped"}

# Task 2 proxy class mapping (from sem/impact_derivation.py)
T2_CLASS_MAP = {
    "k_p":   ["t2_knewExact", "t2_infoAvail"],
    "k_np":  ["t2_notKnow", "t2_infoNotAvail"],
    "m_max": ["t2_fussyPerson"],
    "m_sit": ["t2_needed", "t2_purpose", "t2_realWorld"],
    "m_min": ["t2_easySpeaker", "t2_usualApx"],
}

# Task 3 and Task 2 columns requested for the precise-form diagnostic (c)
T3_PRECISE_COLS = ["t3_easierSpeaker", "t3_soundSmart", "t3_tooMuchInf",
                   "t3_appropriate", "t3_giveMore"]
T2_PRECISE_COLS = ["t2_fussyPerson", "t2_knewExact", "t2_infoAvail",
                   "t2_easySpeaker", "t2_needed"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _welch_df(s1, n1, s2, n2):
    """Welch-Satterthwaite degrees of freedom."""
    v1, v2 = s1**2 / n1, s2**2 / n2
    return (v1 + v2)**2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))


def welch_test_raw(a: pd.Series, b: pd.Series) -> Dict:
    """Welch two-sample t-test; returns raw and normalized Δ, df, t, p, CI."""
    n1, n2 = len(a), len(b)
    m1, m2 = float(a.mean()), float(b.mean())
    s1, s2 = float(a.std(ddof=1)), float(b.std(ddof=1))
    t, p = sp_stats.ttest_ind(b, a, equal_var=False)
    df = _welch_df(s1, n1, s2, n2)
    se = math.sqrt(s1**2/n1 + s2**2/n2)
    t_crit = sp_stats.t.ppf(0.975, df)
    delta_raw  = round(m2 - m1, 4)
    delta_norm = round(normalize_likert(m2) - normalize_likert(m1), 4)
    # CI on raw scale (t-based)
    ci_raw_lo = round(delta_raw - t_crit * se, 4)
    ci_raw_hi = round(delta_raw + t_crit * se, 4)
    # CI on normalized scale
    scale = 1.0 / 3.0
    se_norm = math.sqrt((s1*scale)**2/n1 + (s2*scale)**2/n2)
    ci_norm_lo = round(delta_norm - t_crit * se_norm, 4)
    ci_norm_hi = round(delta_norm + t_crit * se_norm, 4)
    return {
        "n1": n1, "n2": n2,
        "mean1_raw": round(m1, 4), "mean2_raw": round(m2, 4),
        "delta_raw": delta_raw, "ci_raw": (ci_raw_lo, ci_raw_hi),
        "delta_norm": delta_norm, "ci_norm": (ci_norm_lo, ci_norm_hi),
        "df": round(df, 1), "t": round(float(t), 3), "p": round(float(p), 4),
    }


def _prop_z(k1, n1, k2, n2):
    """Two-proportion z-test (manual)."""
    p1, p2 = k1/n1, k2/n2
    p_pool = (k1+k2)/(n1+n2)
    se = math.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
    if se == 0:
        return float("nan"), 1.0
    z = (p2 - p1) / se
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return z, p


def prop_test(k1, n1, k2, n2) -> Dict:
    """Two-proportion z-test (or Fisher's exact if expected count < 5)."""
    if min(k1, n1-k1, k2, n2-k2) < 5:
        ct = [[k2, n2-k2], [k1, n1-k1]]
        odds, p = fisher_exact(ct)
        method = "Fisher"
        z = float("nan")
    else:
        z, p = _prop_z(k1, n1, k2, n2)
        method = "z"
        z = round(float(z), 3)
    return {
        "p1": round(k1/n1, 3), "p2": round(k2/n2, 3),
        "diff": round(k2/n2 - k1/n1, 3),
        "n1": n1, "n2": n2,
        "z_or_or": z, "p": round(float(p), 4), "method": method,
    }


def class_share(df: pd.DataFrame, cols: List[str]) -> Tuple[int, int]:
    """Number of participants who endorsed at least one proxy in `cols`."""
    endorsed = df[cols].any(axis=1)
    return int(endorsed.sum()), int(len(df))


# ---------------------------------------------------------------------------
# (a) Pedantry full table: raw + normalized, df, Welch t-test
# ---------------------------------------------------------------------------

def pedantry_full(e1: pd.DataFrame, e2: pd.DataFrame) -> List[Dict]:
    rows = []
    col = "pedantic"
    for ctx_val, c in CTX_MAP.items():
        a = e1[(e1.form == "precise") & (e1.context == ctx_val)][col]
        b = e2[(e2.form == "precise") & (e2.context == ctx_val)][col]
        r = welch_test_raw(a, b)
        r["context"] = c
        r["form"] = "v_prc"
        rows.append(r)
    # approx form
    for ctx_val, c in CTX_MAP.items():
        a = e1[(e1.form == "approx") & (e1.context == ctx_val)][col]
        b = e2[(e2.form == "approx") & (e2.context == ctx_val)][col]
        r = welch_test_raw(a, b)
        r["context"] = c
        r["form"] = "v_apx"
        rows.append(r)
    # pooled precise
    a = e1[e1.form == "precise"][col]
    b = e2[e2.form == "precise"][col]
    r = welch_test_raw(a, b)
    r["context"] = "pooled"
    r["form"] = "v_prc"
    rows.append(r)
    # pooled approx
    a = e1[e1.form == "approx"][col]
    b = e2[e2.form == "approx"][col]
    r = welch_test_raw(a, b)
    r["context"] = "pooled"
    r["form"] = "v_apx"
    rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# (b1) Task 2 proxy class shares: approximate form, Exp1 vs Exp2
# ---------------------------------------------------------------------------

def t2_class_shares(e1: pd.DataFrame, e2: pd.DataFrame) -> Dict:
    """For each T2 proxy class, by context + pooled: proportion, n, two-prop test."""
    out = {}
    for cls, cols in T2_CLASS_MAP.items():
        out[cls] = {}
        for ctx_label in ["c_HP", "c_LP", "pooled"]:
            if ctx_label == "pooled":
                df1 = e1[e1.form == "approx"]
                df2 = e2[e2.form == "approx"]
            else:
                ctx_val = {v: k for k, v in CTX_MAP.items()}[ctx_label]
                df1 = e1[(e1.form == "approx") & (e1.context == ctx_val)]
                df2 = e2[(e2.form == "approx") & (e2.context == ctx_val)]
            k1, n1 = class_share(df1, cols)
            k2, n2 = class_share(df2, cols)
            res = prop_test(k1, n1, k2, n2)
            out[cls][ctx_label] = res
    return out


def t2_proxies_approx(e1: pd.DataFrame, e2: pd.DataFrame) -> Dict:
    """Per-proxy proportions for approx form."""
    all_cols = [c for cols in T2_CLASS_MAP.values() for c in cols]
    out = {}
    for col in all_cols:
        out[col] = {}
        for ctx_label in ["c_HP", "c_LP", "pooled"]:
            if ctx_label == "pooled":
                df1 = e1[e1.form == "approx"]
                df2 = e2[e2.form == "approx"]
            else:
                ctx_val = {v: k for k, v in CTX_MAP.items()}[ctx_label]
                df1 = e1[(e1.form == "approx") & (e1.context == ctx_val)]
                df2 = e2[(e2.form == "approx") & (e2.context == ctx_val)]
            k1, n1 = int(df1[col].sum()), int(len(df1))
            k2, n2 = int(df2[col].sum()), int(len(df2))
            out[col][ctx_label] = prop_test(k1, n1, k2, n2)
    return out


# ---------------------------------------------------------------------------
# (b2) Joint listener: Steps 2, 3, 4 (read from JSON)
# ---------------------------------------------------------------------------

def joint_steps_from_json(jdata: Dict, delta_ci: Dict) -> Dict:
    s2_jt = jdata["d1_step2_joint"]
    s3_jt = jdata["d1_step3_joint"]
    # Re-derive step 4 for joint with updated precise-cell treatment
    s4_jt = {}
    approx_agree = []
    for a in ATTRS:
        s4_jt[a] = {}
        for v in FORMS:
            s4_jt[a][v] = {}
            for c in CTXS:
                obs  = delta_ci[a][v][c]["delta"]
                pred = s2_jt[a][v][c]["delta_predicted"]
                ci_lo = delta_ci[a][v][c]["ci_lo"]
                ci_hi = delta_ci[a][v][c]["ci_hi"]
                if v == "v_prc":
                    ci_incl_0 = ci_lo <= 0.0 <= ci_hi
                    s4_jt[a][v][c] = {
                        "observed_delta": obs, "predicted_delta": pred,
                        "ci_includes_0": ci_incl_0,
                    }
                else:
                    agree = (obs > 0) == (pred > 0) if obs != 0 else None
                    s4_jt[a][v][c] = {
                        "observed_delta": obs, "predicted_delta": pred,
                        "sign_agree": agree,
                    }
                    approx_agree.append(agree)
    n_agree = sum(1 for x in approx_agree if x is True)
    n_dis   = sum(1 for x in approx_agree if x is False)
    s4_jt["_summary"] = {
        "approx_cells_sign_agree": n_agree,
        "approx_cells_sign_disagree": n_dis,
    }
    return {"step2": s2_jt, "step3": s3_jt, "step4": s4_jt}


# ---------------------------------------------------------------------------
# (b3) Step 4 staged: replace sign-agree for precise cells with CI-includes-0
# ---------------------------------------------------------------------------

def step4_staged_revised(jdata: Dict, delta_ci: Dict) -> Dict:
    s2_st = jdata["d1_step2_staged"]
    out = {}
    approx_agree = []
    for a in ATTRS:
        out[a] = {}
        for v in FORMS:
            out[a][v] = {}
            for c in CTXS:
                obs  = delta_ci[a][v][c]["delta"]
                pred = s2_st[a][v][c]["delta_predicted"]
                ci_lo = delta_ci[a][v][c]["ci_lo"]
                ci_hi = delta_ci[a][v][c]["ci_hi"]
                if v == "v_prc":
                    out[a][v][c] = {
                        "observed_delta": obs, "predicted_delta": pred,
                        "ci_includes_0": ci_lo <= 0.0 <= ci_hi,
                    }
                else:
                    agree = (obs > 0) == (pred > 0) if obs != 0 else None
                    out[a][v][c] = {
                        "observed_delta": obs, "predicted_delta": pred,
                        "sign_agree": agree,
                    }
                    approx_agree.append(agree)
    n_agree = sum(1 for x in approx_agree if x is True)
    n_dis   = sum(1 for x in approx_agree if x is False)
    out["_summary"] = {
        "approx_cells_sign_agree": n_agree,
        "approx_cells_sign_disagree": n_dis,
    }
    return out


# ---------------------------------------------------------------------------
# (c) Precise form: Task 2 and Task 3 proportions
# ---------------------------------------------------------------------------

def precise_motivations(e1: pd.DataFrame, e2: pd.DataFrame) -> Dict:
    out = {}
    for col in T3_PRECISE_COLS + T2_PRECISE_COLS:
        out[col] = {}
        for ctx_label in ["c_HP", "c_LP", "pooled"]:
            if ctx_label == "pooled":
                df1 = e1[e1.form == "precise"]
                df2 = e2[e2.form == "precise"]
            else:
                ctx_val = {v: k for k, v in CTX_MAP.items()}[ctx_label]
                df1 = e1[(e1.form == "precise") & (e1.context == ctx_val)]
                df2 = e2[(e2.form == "precise") & (e2.context == ctx_val)]
            k1, n1 = int(df1[col].sum()), int(len(df1))
            k2, n2 = int(df2[col].sum()), int(len(df2))
            out[col][ctx_label] = prop_test(k1, n1, k2, n2)
    return out


# ---------------------------------------------------------------------------
# (d) Sweep linearity: closed-form P0*, fraction of predicted shift
# ---------------------------------------------------------------------------

def sweep_linearity(jdata: Dict, delta_ci_dict: Dict) -> Dict:
    obs_apx = np.array([
        [delta_ci_dict[a]["v_apx"][c]["delta"] for c in CTXS]
        for a in ATTRS
    ])  # (3, 2)

    results = {}
    for inference in ["staged", "joint"]:
        # D: predicted Δ per approx cell from the stored sweep at P0=0
        # Under staged, Δ(P0) = (1-P0)*D iff linearity holds.
        # Compute E_base(P0=0) and E_know(P0=1)
        know = build_scenario(priors_K={"k_p": 1.0, "k_np": 0.0}, inference=inference)
        base0 = build_scenario(priors_K={"k_p": 0.0, "k_np": 1.0}, inference=inference)
        E_know_apx = np.array([[know.evaluate_all()[a]["v_apx"][c] for c in CTXS] for a in ATTRS])
        E_base0_apx = np.array([[base0.evaluate_all()[a]["v_apx"][c] for c in CTXS] for a in ATTRS])
        D = E_know_apx - E_base0_apx  # (3, 2)

        # Closed-form P0*
        dot_obs_D = float(np.dot(obs_apx.ravel(), D.ravel()))
        dot_D_D   = float(np.dot(D.ravel(), D.ravel()))
        frac = dot_obs_D / dot_D_D  # <obs,D>/<D,D>
        p0_closed = float(np.clip(1.0 - frac, 0.0, 1.0))

        # Check linearity numerically over the sweep grid
        p_vals = np.arange(0.01, 1.001, 0.01)[:100]
        pred_linear = np.array([(1 - p) * D for p in p_vals])  # (100, 3, 2)

        # Actual predicted Δ from sweep
        pred_actual = np.zeros((len(p_vals), 3, 2))
        for i, p in enumerate(p_vals):
            base_i = build_scenario(priors_K={"k_p": float(p), "k_np": float(1-p)},
                                    inference=inference)
            scores_i = base_i.evaluate_all()
            for ai, a in enumerate(ATTRS):
                for ci, c in enumerate(CTXS):
                    pred_actual[i, ai, ci] = E_know_apx[ai, ci] - scores_i[a]["v_apx"][c]

        max_linear_error = float(np.max(np.abs(pred_actual - pred_linear)))
        is_linear = max_linear_error < 1e-9

        # Bootstrap CI for frac (= 1 - P0*)
        rng = np.random.default_rng(42)
        e1 = pd.read_csv(E1_PATH)
        e2 = pd.read_csv(E2_PATH)
        frac_boot = []
        for _ in range(2000):
            delta_b = np.zeros((3, 2))
            for ai, a_sem in enumerate(ATTRS):
                raw_col = [k for k, v in RATTR_MAP.items() if v == a_sem][0]
                for ci, c in enumerate(CTXS):
                    ctx_val = {v: k for k, v in CTX_MAP.items()}[c]
                    g1 = e1[(e1.form == "approx") & (e1.context == ctx_val)]
                    g2 = e2[(e2.form == "approx") & (e2.context == ctx_val)]
                    s1 = g1.sample(n=len(g1), replace=True,
                                   random_state=int(rng.integers(2**31)))[raw_col]
                    s2 = g2.sample(n=len(g2), replace=True,
                                   random_state=int(rng.integers(2**31)))[raw_col]
                    delta_b[ai, ci] = (normalize_likert(float(s2.mean()))
                                       - normalize_likert(float(s1.mean())))
            f_b = float(np.dot(delta_b.ravel(), D.ravel())) / dot_D_D
            frac_boot.append(f_b)

        frac_arr = np.array(frac_boot)
        ci_lo_frac = round(float(np.percentile(frac_arr, 2.5)), 3)
        ci_hi_frac = round(float(np.percentile(frac_arr, 97.5)), 3)
        p0_ci_lo = round(float(np.clip(1.0 - ci_hi_frac, 0.0, 1.0)), 3)
        p0_ci_hi = round(float(np.clip(1.0 - ci_lo_frac, 0.0, 1.0)), 3)

        # Per-cell: D, obs, sign match
        per_cell = []
        for ai, a in enumerate(ATTRS):
            for ci, c in enumerate(CTXS):
                d_val = round(float(D[ai, ci]), 4)
                obs_val = round(float(obs_apx[ai, ci]), 4)
                sign_match = (d_val > 0) == (obs_val > 0) if obs_val != 0 else None
                per_cell.append({
                    "attr": a, "context": c,
                    "D": d_val, "obs_delta": obs_val,
                    "sign_match": sign_match,
                })

        n_sign_match = sum(1 for x in per_cell if x["sign_match"] is True)

        # SSE-grid P0* (from stored JSON)
        p0_grid = jdata[f"d1_step5_{inference}"]["p0_primary"]

        results[inference] = {
            "D": D.tolist(),
            "frac": round(frac, 4),
            "frac_pct": round(frac * 100, 1),
            "frac_ci": (ci_lo_frac, ci_hi_frac),
            "frac_ci_pct": (round(ci_lo_frac*100, 1), round(ci_hi_frac*100, 1)),
            "p0_closed": round(p0_closed, 4),
            "p0_grid": p0_grid,
            "p0_ci_from_frac": (p0_ci_lo, p0_ci_hi),
            "is_linear": is_linear,
            "max_linear_error": round(max_linear_error, 12),
            "per_cell": per_cell,
            "n_sign_match": n_sign_match,
            "upper_bound_at_boundary": p0_ci_hi == 1.0,
        }

    return results


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(out: Dict, path: str) -> None:
    lines = []
    L = lines.append

    def h2(t): L(f"\n## {t}\n")
    def row(*cols): L("| " + " | ".join(str(c) for c in cols) + " |")
    def sep(n): L("| " + " | ".join(["---"] * n) + " |")
    def pstar(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        if p < 0.10:  return "."
        return "n.s."

    L("# Session F Follow-up Results\n")
    L("Generated by `scripts/exp2_knowledge_followup.py`.\n")

    # ---- (a) Pedantry full table ----
    h2("(a) Pedantry Δ: full table (normalized and raw Likert)")
    L("*Welch two-sample t-test (Exp. 2 − Exp. 1) for all pedantic cells.*\n")
    row("form", "context", "n1", "n2",
        "Δ_norm", "95% CI (norm)", "Δ_raw", "95% CI (raw)",
        "df", "t", "p", "")
    sep(12)
    for r in out["pedantry_full"]:
        row(r["form"], r["context"], r["n1"], r["n2"],
            r["delta_norm"],
            f"[{r['ci_norm'][0]}, {r['ci_norm'][1]}]",
            r["delta_raw"],
            f"[{r['ci_raw'][0]}, {r['ci_raw'][1]}]",
            r["df"], r["t"], r["p"], pstar(r["p"]))

    L("\n**Discrepancy resolved (a):** SESSION-F-RESULTS gave precise-form "
      "c_LP Δ_norm = −0.211 (p = .013) and c_HP Δ_norm = −0.097 (p = .293). "
      "HANDOVER-F-to-G.md had −0.262 and −0.066 — those numbers were wrong "
      "(written from memory, not from the script). Correct values: see table above.\n")

    # ---- (b1) Task 2 proxy shares ----
    h2("(b1) Task 2 proxy class shares: approximate form, Exp. 1 vs. Exp. 2")
    L("*Proportion endorsing at least one proxy in the class; two-proportion test.*\n")
    row("class", "proxies", "context", "Exp1 share", "Exp2 share", "diff", "p", "method", "")
    sep(9)
    cls_proxy_labels = {cls: "+".join(c.replace("t2_","") for c in cols)
                        for cls, cols in T2_CLASS_MAP.items()}
    for cls, ctx_rows in out["t2_class_shares"].items():
        for ctx_label, r in ctx_rows.items():
            row(cls, cls_proxy_labels[cls], ctx_label,
                r["p1"], r["p2"], r["diff"], r["p"], r["method"], pstar(r["p"]))

    L("\n**Model P(m|v_apx, c) — staged, base vs. knowledge:**\n")
    s3 = out["json"]["d1_step3_staged"]
    row("condition", "m_max", "m_sit", "m_min")
    sep(4)
    for label, mk, ck in [
        ("base c_HP",  "base", "v_apx_c_HP"),
        ("base c_LP",  "base", "v_apx_c_LP"),
        ("know c_HP",  "knowledge", "v_apx_c_HP"),
        ("know c_LP",  "knowledge", "v_apx_c_LP"),
    ]:
        pm = s3[mk][ck]
        row(label, pm.get("m_max"), pm.get("m_sit"), pm.get("m_min"))

    L("\n**Direction check (approx form, pooled):** "
      "predicted change base→knowledge vs. observed change Exp1→Exp2:\n")
    row("class", "strategy", "model direction", "proxy", "obs direction", "agrees?")
    sep(6)
    s3b = s3["base"]["v_apx_c_HP"]  # base c_HP for reference
    s3k = s3["knowledge"]["v_apx_c_HP"]
    dir_map = {
        "m_max": ("m_max decreases",  "t2_fussyPerson"),
        "m_sit": ("m_sit increases",  "t2_needed+purpose+realWorld"),
        "m_min": ("m_min increases",  "t2_easySpeaker+usualApx"),
    }
    for strat, (model_dir, proxy_label) in dir_map.items():
        # model change direction (c_HP)
        base_val = s3["base"]["v_apx_c_HP"][strat]
        know_val = s3["knowledge"]["v_apx_c_HP"][strat]
        m_dir_str = "↑" if know_val > base_val else ("↓" if know_val < base_val else "=")

        # cls
        cls = "m_max" if strat == "m_max" else ("m_sit" if strat == "m_sit" else "m_min")
        r_pooled = out["t2_class_shares"][cls]["pooled"]
        obs_dir_str = "↑" if r_pooled["diff"] > 0.02 else ("↓" if r_pooled["diff"] < -0.02 else "≈")
        agrees = (m_dir_str == obs_dir_str)
        row(cls, strat, f"base={base_val}→know={know_val} ({m_dir_str})",
            proxy_label, f"Exp1→Exp2 diff={r_pooled['diff']} ({obs_dir_str})",
            "✓" if agrees else "✗")

    # ---- (b2) Joint listener Steps 2, 3, 4 ----
    h2("(b2) Joint listener: Step 2 (predicted Δ)")
    row("attr", "form", "context", "E_base", "E_knowledge", "Δ_predicted")
    sep(6)
    s2_jt = out["joint"]["step2"]
    for a in ATTRS:
        for v in FORMS:
            for c in CTXS:
                ce = s2_jt[a][v][c]
                row(a, v, c, ce["E_base"], ce["E_knowledge"], ce["delta_predicted"])

    h2("(b2) Joint listener: Step 3 (motivation posteriors P(m|v_apx, c))")
    s3_jt = out["joint"]["step3"]
    row("condition", "m_max", "m_sit", "m_min")
    sep(4)
    for label, mk, ck in [
        ("base c_HP",  "base", "v_apx_c_HP"),
        ("base c_LP",  "base", "v_apx_c_LP"),
        ("know c_HP",  "knowledge", "v_apx_c_HP"),
        ("know c_LP",  "knowledge", "v_apx_c_LP"),
    ]:
        pm = s3_jt[mk][ck]
        row(label, pm.get("m_max"), pm.get("m_sit"), pm.get("m_min"))

    h2("(b2) Joint listener: Step 4 (model vs. empirical Δ)")
    row("attr", "form", "context", "Δ_obs", "Δ_pred", "sign agree / CI incl. 0")
    sep(6)
    s4_jt = out["joint"]["step4"]
    for a in ATTRS:
        for v in FORMS:
            for c in CTXS:
                ce = s4_jt[a][v][c]
                if v == "v_prc":
                    col_val = "CI incl. 0: " + ("yes" if ce["ci_includes_0"] else "NO")
                else:
                    col_val = "✓" if ce.get("sign_agree") else ("✗" if ce.get("sign_agree") is False else "–")
                row(a, v, c, ce["observed_delta"], ce["predicted_delta"], col_val)
    sm_jt = s4_jt["_summary"]
    L(f"\n*Approx cells: {sm_jt['approx_cells_sign_agree']} agree, "
      f"{sm_jt['approx_cells_sign_disagree']} disagree.*\n")

    # ---- (b3) Step 4 staged revised ----
    h2("(b3) Step 4 staged: revised (precise cells show CI includes 0)")
    row("attr", "form", "context", "Δ_obs", "Δ_pred", "sign agree / CI incl. 0")
    sep(6)
    s4_st = out["step4_staged_revised"]
    for a in ATTRS:
        for v in FORMS:
            for c in CTXS:
                ce = s4_st[a][v][c]
                if v == "v_prc":
                    col_val = "CI incl. 0: " + ("yes" if ce["ci_includes_0"] else "NO")
                else:
                    col_val = "✓" if ce.get("sign_agree") else ("✗" if ce.get("sign_agree") is False else "–")
                row(a, v, c, ce["observed_delta"], ce["predicted_delta"], col_val)
    sm_st = s4_st["_summary"]
    L(f"\n*Approx cells: {sm_st['approx_cells_sign_agree']} agree, "
      f"{sm_st['approx_cells_sign_disagree']} disagree.*\n")

    # ---- (c) Precise form motivations ----
    h2("(c) Precise form: Task 3 and Task 2 proportions, Exp. 1 vs. Exp. 2")
    L("*Two-proportion test; n = participants in that form × context cell.*\n")
    row("col", "task", "context", "Exp1 share", "Exp2 share", "diff", "n1", "n2", "p", "")
    sep(10)
    for col in T3_PRECISE_COLS + T2_PRECISE_COLS:
        task = "T3" if col.startswith("t3_") else "T2"
        for ctx_label in ["c_HP", "c_LP", "pooled"]:
            r = out["precise_motivations"][col][ctx_label]
            row(col, task, ctx_label, r["p1"], r["p2"], r["diff"], r["n1"], r["n2"],
                r["p"], pstar(r["p"]))

    # Summary for the question
    L("\n**Answer (c):** For the precise form, context c_LP, pooled:\n")
    easier = out["precise_motivations"]["t3_easierSpeaker"]["c_LP"]
    fussy  = out["precise_motivations"]["t2_fussyPerson"]["c_LP"]
    easier_pool = out["precise_motivations"]["t3_easierSpeaker"]["pooled"]
    fussy_pool  = out["precise_motivations"]["t2_fussyPerson"]["pooled"]
    L(f"- `t3_easierSpeaker` (c_LP): Exp1={easier['p1']}, Exp2={easier['p2']}, "
      f"diff={easier['diff']}, p={easier['p']} {pstar(easier['p'])}")
    L(f"- `t2_fussyPerson` (c_LP):  Exp1={fussy['p1']}, Exp2={fussy['p2']}, "
      f"diff={fussy['diff']}, p={fussy['p']} {pstar(fussy['p'])}")
    L(f"- `t3_easierSpeaker` (pooled): diff={easier_pool['diff']}, p={easier_pool['p']} {pstar(easier_pool['p'])}")
    L(f"- `t2_fussyPerson` (pooled):  diff={fussy_pool['diff']}, p={fussy_pool['p']} {pstar(fussy_pool['p'])}")
    L("")

    # ---- (d) Sweep linearity ----
    h2("(d) Sweep linearity: closed-form P0* and fraction of predicted shift")

    for inference in ["staged", "joint"]:
        r = out["sweep_linearity"][inference]
        L(f"\n### {inference.capitalize()} listener\n")

        L(f"**Linearity holds:** {r['is_linear']} "
          f"(max deviation from (1−P0)·D: {r['max_linear_error']:.2e})\n")

        frac_lo, frac_hi = r["frac_ci_pct"]
        L(f"**Fraction of predicted shift realized in data:** "
          f"{r['frac_pct']} % (95 % boot CI: [{frac_lo}, {frac_hi}] %)\n")
        L(f"**Closed-form P0* = 1 − frac:** {r['p0_closed']:.3f}  "
          f"(grid sweep gave P0* = {r['p0_grid']})\n")
        L(f"**Bootstrap CI for P0*:** [{r['p0_ci_from_frac'][0]}, {r['p0_ci_from_frac'][1]}]\n")
        if r["upper_bound_at_boundary"]:
            L("*Note: upper bound = 1.0 is the boundary of the parameter space "
              "(clipped), not a posterior estimate.*\n")

        row("attr", "context", "D (full predicted shift)", "obs Δ", "sign matches D")
        sep(5)
        for cell in r["per_cell"]:
            agree_str = "✓" if cell["sign_match"] else ("✗" if cell["sign_match"] is False else "–")
            row(cell["attr"], cell["context"], cell["D"], cell["obs_delta"], agree_str)
        L(f"\n*Sign matches: {r['n_sign_match']}/6 approximate cells.*\n")

    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Report written to {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Loading data and JSON results...")
    e1 = pd.read_csv(E1_PATH)
    e2 = pd.read_csv(E2_PATH)
    with open(JSON_PATH) as fh:
        jdata = json.load(fh)

    delta_ci = jdata["q1_delta_ci"]

    print("(a) Pedantry full table...")
    ped_full = pedantry_full(e1, e2)

    print("(b1) Task 2 proxy class shares...")
    t2_shares = t2_class_shares(e1, e2)
    t2_proxies = t2_proxies_approx(e1, e2)

    print("(b2) Joint listener steps 2-4...")
    joint = joint_steps_from_json(jdata, delta_ci)

    print("(b3) Step 4 staged revised...")
    s4_staged_rev = step4_staged_revised(jdata, delta_ci)

    print("(c) Precise form motivations...")
    prec_motiv = precise_motivations(e1, e2)

    print("(d) Sweep linearity...")
    sweep_lin = sweep_linearity(jdata, delta_ci)

    out = {
        "pedantry_full":       ped_full,
        "t2_class_shares":     t2_shares,
        "t2_proxies_approx":   t2_proxies,
        "joint":               joint,
        "step4_staged_revised": s4_staged_rev,
        "precise_motivations": prec_motiv,
        "sweep_linearity":     sweep_lin,
        "json":                jdata,
    }

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    write_report(out, OUT_REPORT)

    # Also save numeric results
    out_json = out.copy()
    del out_json["json"]
    json_path = "data/cs1_experiment/processed/exp2_knowledge_followup_cs1.json"
    with open(json_path, "w") as fh:
        json.dump(out_json, fh, indent=2)
    print(f"JSON written to {json_path}")


if __name__ == "__main__":
    main()
