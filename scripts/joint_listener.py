"""
scripts/joint_listener.py
=========================
Staged vs. joint listener for Case Study 1 (numerical (im)precision).

Uses ``SEMScenario(inference="joint")`` (Session F integration); see
``docs/staged_vs_joint.md`` for the model description and
``tests/test_inference.py`` for the numerical cross-checks.

Run:
    PYTHONPATH=. python scripts/joint_listener.py
"""

import json
import os
from typing import Dict, List

import numpy as np

from sem import fit
from sem.comparison import sample_parameters
from case_studies.cs1_imprecision import build_scenario, target_effects

EMPIRICAL_JSON = "data/cs1_experiment/processed/empirical_means_cs1.json"
OUT_PATH = "data/cs1_experiment/processed/joint_listener_cs1.json"

METRICS = ["das", "iss", "rmse", "spearman_r", "ccc",
           "cds_main", "cds_interaction", "cds"]


def posteriors() -> Dict:
    out: Dict = {}
    for inf in ("staged", "joint"):
        sc = build_scenario(inference=inf)
        out[inf] = {
            f"{v}|{c}": sc.prob_k("k_p", v, c)
            for v in ["v_prc", "v_apx"]
            for c in ["c_LP", "c_HP"]
        }
    sc = build_scenario(inference="joint",
                        priors_M={"m_max": 0.8, "m_sit": 0.1, "m_min": 0.1})
    out["joint_pmax_0.8"] = {"v_apx|c_LP": sc.prob_k("k_p", "v_apx", "c_LP")}
    return out


def balanced() -> Dict:
    out: Dict = {}
    for inf in ("staged", "joint"):
        scores = build_scenario(inference=inf).evaluate_all()
        result = fit.evaluate(scores, EMPIRICAL_JSON)
        out[inf] = {
            "scores": scores,
            "effects": target_effects(scores),
            "metrics": {k: result[k] for k in METRICS},
            "esr_main": result["esr_main"],
            "esr_interaction": result["esr_interaction"],
        }
    return out


def sampled(n_samples: int = 100, seed: int = 42) -> Dict:
    params = sample_parameters("full", n_samples=n_samples, seed=seed)
    out: Dict = {}
    for inf in ("staged", "joint"):
        results: List[Dict] = []
        for p in params:
            sc = build_scenario(omega=p["omega"], priors_K=p["priors_K"],
                                priors_M=p["priors_M"], inference=inf)
            scores = sc.evaluate_all()
            r = fit.evaluate(scores, EMPIRICAL_JSON)
            r["effects"] = target_effects(scores)
            results.append(r)
        out[inf] = {
            "metrics": {
                k: {"mean": float(np.mean([r[k] for r in results])),
                    "sd":   float(np.std( [r[k] for r in results]))}
                for k in METRICS
            },
            "all_four":   sum(all(r["effects"]) for r in results),
            "per_effect": [sum(r["effects"][i] for r in results) for i in range(4)],
            "n": n_samples,
        }
    return out


def sweep(step: float = 0.05) -> Dict:
    out: Dict = {}
    for inf in ("staged", "joint"):
        sc = build_scenario(inference=inf)
        rt = sc.robustness_test(target_effects, step=step)
        failures = rt["failures"]
        fail_region: Dict = {}
        if failures:
            fail_region = {
                "omega":    [min(f["omega"]              for f in failures),
                             max(f["omega"]              for f in failures)],
                "P(m_sit)": [min(f["priors_M"]["m_sit"]  for f in failures),
                             max(f["priors_M"]["m_sit"]  for f in failures)],
                "P(m_min)": [min(f["priors_M"]["m_min"]  for f in failures),
                             max(f["priors_M"]["m_min"]  for f in failures)],
            }
        out[inf] = {
            "total":        rt["total"],
            "all_correct":  rt["all_correct"],
            "rate_all":     rt["rate_all"],
            "per_effect":   rt["per_effect"],
            "failure_region": fail_region,
        }
    return out


def _print_scores(scores: Dict) -> None:
    for a in scores:
        cells = "  ".join(f"{v}/{c}={scores[a][v][c]:+.3f}"
                          for v in scores[a] for c in scores[a][v])
        print(f"  {a:<7} {cells}")


def main() -> None:
    results = {"posteriors": posteriors(), "balanced": balanced(),
               "sampled": sampled(), "sweep": sweep()}

    print("=== P(k_p | u, c), flat priors ===")
    for name in ("staged", "joint"):
        for cell, p in results["posteriors"][name].items():
            print(f"  {name:<6} {cell}: {p:.3f}")
    print(f"  joint, P(m_max)=.8, v_apx|c_LP: "
          f"{results['posteriors']['joint_pmax_0.8']['v_apx|c_LP']:.3f}")

    print("\n=== Human means (normalized) ===")
    human = fit.load_empirical_means(EMPIRICAL_JSON)
    _print_scores({a: human[a] for a in ["a_comp", "a_like", "a_ped"]})

    for name in ("staged", "joint"):
        b = results["balanced"][name]
        print(f"\n=== Balanced model (ω = 0.5, flat priors): {name} ===")
        _print_scores(b["scores"])
        print("  effects 1-4:", b["effects"])
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in b["metrics"].items()))

    print("\n=== 100 sampled parameter combinations (seed 42) ===")
    for name in ("staged", "joint"):
        s = results["sampled"][name]
        print(f"  {name:<6} " + "  ".join(
            f"{k}={v['mean']:.2f}±{v['sd']:.2f}" for k, v in s["metrics"].items()))
        print(f"         all four effects: {s['all_four']}/{s['n']}; "
              f"per effect: {s['per_effect']}")

    print("\n=== Robustness sweep (step .05) ===")
    for name in ("staged", "joint"):
        w = results["sweep"][name]
        print(f"  {name:<6} {w['all_correct']}/{w['total']} ({w['rate_all']:.3f}); "
              f"per effect {w['per_effect']}; failures at {w['failure_region']}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
