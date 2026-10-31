"""
scripts/exp2_section44.py
=========================
Reproduce every number in §4.4 of the paper ("A Held-out Test: Experiment 2").

Run:
    PYTHONPATH=. python scripts/exp2_section44.py

Output: prints a table and STOPS with a non-zero exit code if any value
differs from the paper's expected value beyond rounding.

Also implements Step 2b: φ derivation for the source state (t2_infoAvail
and t2_knewExact) using the §3.3 procedure.
"""

import json
import math
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from sem.impact_derivation import _trivalent
from sem.fit import normalize_likert

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

E1_PATH       = "data/cs1_experiment/processed/experiment1.csv"
E2_PATH       = "data/cs1_experiment/processed/experiment2.csv"
D1_JSON       = "data/cs1_experiment/processed/exp2_knowledge_test_cs1.json"
FOLLOWUP_JSON = "data/cs1_experiment/processed/exp2_knowledge_followup_cs1.json"
ACCESS_JSON   = "data/cs1_experiment/processed/exp2_access_split_cs1.json"

# ---------------------------------------------------------------------------
# Tolerance for "stop if differs" check
# ---------------------------------------------------------------------------

RTOL = 0.01   # relative tolerance: ±1 % of the expected value


def _near(got: float, exp: float, rtol: float = RTOL) -> bool:
    if exp == 0.0:
        return abs(got) <= 0.001
    return abs(got - exp) / abs(exp) <= rtol


# ---------------------------------------------------------------------------
# Step 2: §4.4 number table
# ---------------------------------------------------------------------------

def section44_numbers() -> List[Dict]:
    """
    Return a list of dicts, one per §4.4 claim:
      {"claim": str, "expected": float, "got": float, "source": str, "ok": bool}
    """
    rows = []

    def add(claim, expected, got, source):
        ok = _near(float(got), float(expected))
        rows.append({
            "claim":    claim,
            "expected": round(float(expected), 4),
            "got":      round(float(got), 4),
            "source":   source,
            "ok":       ok,
        })

    # ---- Load data ----
    e1 = pd.read_csv(E1_PATH)
    e2 = pd.read_csv(E2_PATH)
    with open(D1_JSON)       as f: d1 = json.load(f)
    with open(FOLLOWUP_JSON) as f: fu = json.load(f)
    with open(ACCESS_JSON)   as f: ac = json.load(f)

    mm   = d1["q1_mixed_model"]
    cs   = fu["t2_class_shares"]
    pm   = fu["precise_motivations"]

    # 1. t3_didntKnow, approx form: E1 70 % → E2 43 %
    e1_apx_dk = float(e1[e1.form == "approx"]["t3_didntKnow"].mean())
    e2_apx_dk = float(e2[e2.form == "approx"]["t3_didntKnow"].mean())
    add("t3_didntKnow approx, Exp 1 (%)",   70.3, e1_apx_dk * 100,
        "experiment1.csv, approx, t3_didntKnow mean")
    add("t3_didntKnow approx, Exp 2 (%)",   42.8, e2_apx_dk * 100,
        "experiment2.csv, approx, t3_didntKnow mean")

    # 2. k_np reasons (t2_notKnow ∪ t2_infoNotAvail), approx, pooled:
    #    Exp1 63 % → Exp2 39 %, p < .001
    knp = cs["k_np"]["pooled"]
    add("k_np reasons approx, Exp 1 (%)",  63.2, knp["p1"] * 100,
        "exp2_knowledge_followup_cs1.json, t2_class_shares k_np pooled p1")
    add("k_np reasons approx, Exp 2 (%)",  39.0, knp["p2"] * 100,
        "exp2_knowledge_followup_cs1.json, t2_class_shares k_np pooled p2")
    add("k_np reasons drop p (< .001)",      0.0, knp["p"],
        "exp2_knowledge_followup_cs1.json, t2_class_shares k_np pooled p")

    # 3. Mixed model: knowledge on competence p = .26; likeability p = .97
    def mm_p(attr_raw, term):
        for entry in mm[attr_raw]:
            if entry["term"] == term:
                return entry["p"]
        raise KeyError(f"{term} not found in {attr_raw}")

    def mm_coef(attr_raw, term):
        for entry in mm[attr_raw]:
            if entry["term"] == term:
                return entry["coef"]
        raise KeyError(f"{term} not found in {attr_raw}")

    add("mixed model: knowledge on competence p", 0.264,
        mm_p("competent", "knowledge1"),
        "exp2_knowledge_test_cs1.json, q1_mixed_model competent knowledge1 p")
    add("mixed model: knowledge on likeability p", 0.970,
        mm_p("likeable", "knowledge1"),
        "exp2_knowledge_test_cs1.json, q1_mixed_model likeable knowledge1 p")

    # 4. No form×knowledge interaction, all p ≥ .17 (min = .1715)
    fi = [mm_p(a, "form1:knowledge1")
          for a in ["competent", "likeable", "pedantic"]]
    add("min p(form × knowledge) across attrs (≥ .17)", 0.1715, min(fi),
        "exp2_knowledge_test_cs1.json, q1_mixed_model, form1:knowledge1 terms")

    # 5. Knowledge on pedantry p = .017 (β = −0.15)
    add("mixed model: knowledge on pedantry p",   0.0169,
        mm_p("pedantic", "knowledge1"),
        "exp2_knowledge_test_cs1.json, q1_mixed_model pedantic knowledge1 p")
    add("mixed model: knowledge on pedantry β",  -0.149,
        mm_coef("pedantic", "knowledge1"),
        "exp2_knowledge_test_cs1.json, q1_mixed_model pedantic knowledge1 coef")

    # 6. Pedantry drop largest: precise / c_LP = −0.211
    dci = d1["q1_delta_ci"]
    add("Δ pedantry, v_prc, c_LP",  -0.211,
        dci["a_ped"]["v_prc"]["c_LP"]["delta"],
        "exp2_knowledge_test_cs1.json, q1_delta_ci a_ped v_prc c_LP delta")

    # 7. Precise form, t2_knewExact: −11 percentage points, p = .003
    kex = pm["t2_knewExact"]["pooled"]
    add("precise t2_knewExact diff (pp)", -11.1, kex["diff"] * 100,
        "exp2_knowledge_followup_cs1.json, precise_motivations t2_knewExact pooled diff")
    add("precise t2_knewExact p",          0.0028, kex["p"],
        "exp2_knowledge_followup_cs1.json, precise_motivations t2_knewExact pooled p")

    # 8. Precise form, t2_infoAvail: +13 percentage points, p < .001
    iav = pm["t2_infoAvail"]["pooled"]
    add("precise t2_infoAvail diff (pp)",  12.6, iav["diff"] * 100,
        "exp2_knowledge_followup_cs1.json, precise_motivations t2_infoAvail pooled diff")
    add("precise t2_infoAvail p",           0.0009, iav["p"],
        "exp2_knowledge_followup_cs1.json, precise_motivations t2_infoAvail pooled p")

    # 9. Strategy reasons unchanged: fussyPerson p = .807; easierSpeaker p = .842
    fp = pm["t2_fussyPerson"]["pooled"]
    es = pm["t3_easierSpeaker"]["pooled"]
    add("precise t2_fussyPerson p (≈ .807)",   0.8067, fp["p"],
        "exp2_knowledge_followup_cs1.json, precise_motivations t2_fussyPerson pooled p")
    add("precise t3_easierSpeaker p (≈ .842)", 0.8424, es["p"],
        "exp2_knowledge_followup_cs1.json, precise_motivations t3_easierSpeaker pooled p")

    # 10. Prior fitted to pedantry ≈ .7 → .698
    pi_ped = ac["step4d"]["model_ii"]["ls_pi"]["a_ped"]["pi_star"]
    add("pedantry-fitted π*",  0.698, pi_ped,
        "exp2_access_split_cs1.json, step4d model_ii ls_pi a_ped pi_star")

    # 11 & 12. Model predictions at ped π* = .698 (4d model ii, α·(1−π))
    pi_val = float(pi_ped)
    alpha_ped_prc = -0.5
    alpha_ped_apx = -1.0 / 6.0
    alpha_comp_apx_HP = 0.5    # model ii, v_apx, c_HP
    alpha_comp_apx_LP = 1.0    # model ii, v_apx, c_LP

    delta_pred_ped_prc = alpha_ped_prc * (1.0 - pi_val)
    delta_pred_ped_apx = alpha_ped_apx * (1.0 - pi_val)
    delta_pred_comp_apx_HP = alpha_comp_apx_HP * (1.0 - pi_val)
    delta_pred_comp_apx_LP = alpha_comp_apx_LP * (1.0 - pi_val)

    add("predicted Δ_ped v_prc (mean contexts)",  -0.151, delta_pred_ped_prc,
        "analytical: α_ped_prc·(1−π) = -0.5·(1-0.698)")
    add("predicted Δ_ped v_apx (mean contexts)",  -0.050, delta_pred_ped_apx,
        "analytical: α_ped_apx·(1−π) = (-1/6)·(1-0.698)")
    add("predicted Δ_comp v_apx c_HP",  0.151, delta_pred_comp_apx_HP,
        "analytical: α_comp_apx_HP·(1−π) = 0.5·(1-0.698)")
    add("predicted Δ_comp v_apx c_LP",  0.302, delta_pred_comp_apx_LP,
        "analytical: α_comp_apx_LP·(1−π) = 1.0·(1-0.698)")

    # 12. Observed pedantry drop: mean of two contexts
    obs_ped_prc = (dci["a_ped"]["v_prc"]["c_HP"]["delta"]
                   + dci["a_ped"]["v_prc"]["c_LP"]["delta"]) / 2
    obs_ped_apx = (dci["a_ped"]["v_apx"]["c_HP"]["delta"]
                   + dci["a_ped"]["v_apx"]["c_LP"]["delta"]) / 2
    add("observed Δ_ped v_prc (mean contexts)", -0.154, obs_ped_prc,
        "exp2_knowledge_test_cs1.json, q1_delta_ci mean a_ped v_prc c_HP + c_LP")
    add("observed Δ_ped v_apx (mean contexts)", -0.041, obs_ped_apx,
        "exp2_knowledge_test_cs1.json, q1_delta_ci mean a_ped v_apx c_HP + c_LP")

    # 13. Observed competence rise (approximate form)
    add("observed Δ_comp v_apx c_HP",  0.072,
        dci["a_comp"]["v_apx"]["c_HP"]["delta"],
        "exp2_knowledge_test_cs1.json, q1_delta_ci a_comp v_apx c_HP delta")
    add("observed Δ_comp v_apx c_LP",  0.037,
        dci["a_comp"]["v_apx"]["c_LP"]["delta"],
        "exp2_knowledge_test_cs1.json, q1_delta_ci a_comp v_apx c_LP delta")

    # 14. Caveat footnote: within Exp.2, precise, infoAvail vs knewExact
    step5_exp2 = ac["step5"]["exp2"]
    add("Step 5 Exp2: Δ_ped (infoAvail − knewExact)",
        -0.232, step5_exp2["a_ped"]["delta_norm"],
        "exp2_access_split_cs1.json, step5 exp2 a_ped delta_norm")
    add("Step 5 Exp2: pedantry p",
        0.127,  step5_exp2["a_ped"]["p"],
        "exp2_access_split_cs1.json, step5 exp2 a_ped p")
    add("Step 5 Exp2: Δ_comp (infoAvail − knewExact)",
        0.197,  step5_exp2["a_comp"]["delta_norm"],
        "exp2_access_split_cs1.json, step5 exp2 a_comp delta_norm")
    add("Step 5 Exp2: competence p",
        0.066,  step5_exp2["a_comp"]["p"],
        "exp2_access_split_cs1.json, step5 exp2 a_comp p")

    return rows


def print_section44_table(rows: List[Dict]) -> int:
    """Print the §4.4 table. Returns number of failures."""
    header = f"{'§4.4 claim':<55} {'expected':>9} {'computed':>9} {'ok':>4}  source"
    print(header)
    print("-" * len(header))
    failures = 0
    for r in rows:
        status = "✓" if r["ok"] else "FAIL"
        if not r["ok"]:
            failures += 1
        print(f"{r['claim']:<55} {r['expected']:>9.4f} {r['got']:>9.4f} {status:>4}  {r['source']}")
    print()
    print(f"{'PASS' if failures == 0 else 'FAIL'}: {len(rows) - failures}/{len(rows)} values reproduced within {RTOL*100:.0f}% tolerance")
    return failures


# ---------------------------------------------------------------------------
# Record: RMSE at pedantry-fitted π vs. no change
# ---------------------------------------------------------------------------

def rmse_record():
    with open(ACCESS_JSON) as f:
        ac = json.load(f)

    pi_ped = ac["step4d"]["model_ii"]["ls_pi"]["a_ped"]["pi_star"]
    pi_overall = ac["step4d"]["model_ii"]["ls_pi"]["overall"]["pi_star"]

    tbl_ped = ac["step4d"]["model_ii"]["tables"]["ped_pi_star"]
    tbl_ovr = ac["step4d"]["model_ii"]["tables"]["overall_pi_star"]
    rmse_no_change = ac["reading2"]["rmse_overall"]

    print("=== For the record (not in paper): RMSE comparison ===")
    print(f"  Model (ii) 4d at ped π* = {pi_ped:.3f}: RMSE = {tbl_ped['rmse_overall']:.4f}")
    print(f"  Model (ii) 4d at overall π* = {pi_overall:.3f}: RMSE = {tbl_ovr['rmse_overall']:.4f}")
    print(f"  Reading (2) — no change (Δ=0): RMSE = {rmse_no_change:.4f}")
    print()


# ---------------------------------------------------------------------------
# Step 2b: φ derivation for t2_infoAvail and t2_knewExact
# ---------------------------------------------------------------------------

def step2b_source_impact() -> Dict:
    """
    Apply the §3.3 φ-derivation procedure to t2_infoAvail (source state)
    and t2_knewExact (knew beforehand) as proxies.

    Run separately on Exp.1, Exp.2 (precise form only) and pooled-over-
    forms (the variant the paper uses for φ).

    Also reports the direct contrast infoAvail-only vs knewExact-only
    (as in Session G Step 5) for comparison.
    """
    e1 = pd.read_csv(E1_PATH)
    e2 = pd.read_csv(E2_PATH)

    raw_attrs = ["competent", "likeable", "pedantic"]
    attr_map  = {"competent": "a_comp", "likeable": "a_like", "pedantic": "a_ped"}

    def phi_procedure(df: pd.DataFrame, col: str, alpha: float) -> Dict:
        checked     = df[df[col] == 1]
        not_checked = df[df[col] == 0]
        n_checked   = int(df[col].sum())
        n_total     = len(df)

        result = {"n_checked": n_checked, "n_total": n_total, "attrs": {}}
        for raw_attr in raw_attrs:
            g1 = checked[raw_attr].dropna()
            g2 = not_checked[raw_attr].dropna()
            if len(g1) < 2 or len(g2) < 2:
                result["attrs"][attr_map[raw_attr]] = {
                    "skipped": True, "n_checked": len(g1)}
                continue
            _, p  = ttest_ind(g1, g2, equal_var=True)
            diff  = float(g1.mean() - g2.mean())
            dnorm = normalize_likert(float(g1.mean())) - normalize_likert(float(g2.mean()))
            result["attrs"][attr_map[raw_attr]] = {
                "n_checked":  len(g1),
                "n_not":      len(g2),
                "diff_raw":   round(diff, 4),
                "diff_norm":  round(dnorm, 4),
                "p":          round(float(p), 4),
                "phi_001":    _trivalent(diff, float(p), 0.01),
                "phi_010":    _trivalent(diff, float(p), 0.10),
            }
        return result

    def direct_contrast(df: pd.DataFrame) -> Dict:
        """infoAvail-only vs knewExact-only (Session G Step 5 logic)."""
        ia_only = df[(df.t2_infoAvail == 1) & (df.t2_knewExact == 0)]
        ke_only = df[(df.t2_infoAvail == 0) & (df.t2_knewExact == 1)]
        both    = df[(df.t2_infoAvail == 1) & (df.t2_knewExact == 1)]
        result  = {
            "n_ia_only": len(ia_only),
            "n_ke_only": len(ke_only),
            "n_both_excl": len(both),
            "attrs": {},
        }
        for raw_attr in raw_attrs:
            if len(ia_only) < 5 or len(ke_only) < 5:
                result["attrs"][attr_map[raw_attr]] = {
                    "skipped": True, "reason": f"n_ia={len(ia_only)}, n_ke={len(ke_only)}"}
                continue
            _, p  = ttest_ind(ia_only[raw_attr], ke_only[raw_attr], equal_var=False)
            diff  = float(ia_only[raw_attr].mean() - ke_only[raw_attr].mean())
            dnorm = (normalize_likert(float(ia_only[raw_attr].mean()))
                     - normalize_likert(float(ke_only[raw_attr].mean())))
            result["attrs"][attr_map[raw_attr]] = {
                "n_ia_only": len(ia_only), "n_ke_only": len(ke_only),
                "diff_raw":  round(diff, 4),
                "diff_norm": round(dnorm, 4),
                "p":         round(float(p), 4),
            }
        return result

    datasets = {
        "exp1_precise": e1[e1.form == "precise"],
        "exp2_precise": e2[e2.form == "precise"],
        "exp1_all":     e1,
        "exp2_all":     e2,
        "pooled_all":   pd.concat([e1, e2], ignore_index=True),
    }

    out = {}
    for dset_name, df in datasets.items():
        out[dset_name] = {
            "phi_infoAvail": phi_procedure(df, "t2_infoAvail", 0.01),
            "phi_knewExact": phi_procedure(df, "t2_knewExact", 0.01),
            "direct_contrast": direct_contrast(df),
        }
    return out


def print_step2b(res: Dict) -> None:
    print("=== Step 2b: φ derivation for source state (t2_infoAvail / t2_knewExact) ===\n")
    print("Procedure: §3.3 (checked vs not-checked, equal-var t-test).")
    print("φ: +1 if p < α and diff > 0; −1 if p < α and diff < 0; 0 otherwise.\n")

    for dset_name, dset in res.items():
        print(f"--- {dset_name} ---")
        for proxy_key, proxy_label in [
            ("phi_infoAvail", "t2_infoAvail (source class)"),
            ("phi_knewExact", "t2_knewExact (knew beforehand)"),
        ]:
            pr = dset[proxy_key]
            print(f"  {proxy_label}: n={pr['n_checked']}/{pr['n_total']} checked")
            for asem, ar in pr["attrs"].items():
                if ar.get("skipped"):
                    print(f"    {asem}: skipped (n_checked={ar.get('n_checked','?')})")
                else:
                    print(f"    {asem}: diff_norm={ar['diff_norm']:+.4f}, p={ar['p']:.4f}, "
                          f"φ(p<.01)={ar['phi_001']:+d}, φ(p<.10)={ar['phi_010']:+d}")

        dc = dset["direct_contrast"]
        print(f"  Direct contrast (infoAvail-only n={dc['n_ia_only']} vs knewExact-only n={dc['n_ke_only']}, "
              f"both excl={dc['n_both_excl']}):")
        for asem, ar in dc["attrs"].items():
            if ar.get("skipped"):
                print(f"    {asem}: skipped ({ar.get('reason','?')})")
            else:
                print(f"    {asem}: Δ_norm={ar['diff_norm']:+.4f}, p={ar['p']:.4f}")
        print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("§4.4 number check")
    print("=" * 80)
    rows = section44_numbers()
    failures = print_section44_table(rows)
    print()

    rmse_record()

    print("=" * 80)
    step2b_res = step2b_source_impact()
    print_step2b(step2b_res)

    if failures > 0:
        print(f"ERROR: {failures} value(s) differ beyond tolerance. Stopping.")
        sys.exit(1)


if __name__ == "__main__":
    main()
