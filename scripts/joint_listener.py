"""
scripts/joint_listener.py
=========================
Staged vs. joint listener for Case Study 1 (numerical (im)precision).

Background (manuscript §5.5, revision item M15)
-----------------------------------------------
SEM's first stage is *staged*: the knowledge posterior uses truthfulness only
(Eq. 2 in the revised paper, ``SEMScenario.prob_k``),

    P(k|u) ∝ P(k) · χ_t(k,u),

so it ignores which strategies could have produced the utterance. A *joint*
listener computes the knowledge posterior from the full generative model,

    P(k|u,c) ∝ P(k) · Σ_m P(m) · χ_m(c,k,u),

and keeps the strategy posterior P(m|k,u,c) (Eq. 3) unchanged. Under the joint
listener, priors over strategies (and the context) also shift the knowledge
inference ("explaining away" between knowledge and motivation). The two
listeners agree on the precise form (P(k_p|u_prc) = 1 either way) and differ
only on the approximate form.

This script compares the two listeners on the CS1 data with the existing
metric suite:

  1. knowledge posteriors under flat priors (and one non-flat example);
  2. the balanced model (ω = 0.5, flat priors);
  3. 100 sampled parameter combinations (as for the full variant in
     ``sem.comparison``; seed 42);
  4. the robustness sweep over ω, P(k), P(m) (step .05).

Results (Session E1, 2026-10-27): both listeners reproduce all four effects
(balanced: DAS = ISS = 1; sampled: 100/100 staged, 99/100 joint; sweep:
99.9% staged, 98.8% joint). Joint: slightly more main-effect overshoot, slightly
better interaction calibration. The data do not decide between the two.

Starting point for Session F (Claude Code): possible integration as a
``joint`` option of ``SEMScenario`` (e.g. ``SEMScenario(..., inference="joint")``)
plus tests; see TODO_meaning_revision.md, Phase 6.

Run:
    PYTHONPATH=. python scripts/joint_listener.py
"""

import itertools
import json
import os
from typing import Dict, List, Optional

import numpy as np

from sem import fit
from sem.model import SEMScenario
from sem.comparison import sample_parameters
from case_studies.cs1_imprecision import build_scenario, target_effects


EMPIRICAL_JSON = "data/cs1_experiment/processed/empirical_means_cs1.json"
OUT_PATH = "data/cs1_experiment/processed/joint_listener_cs1.json"

METRICS = ["das", "iss", "rmse", "spearman_r", "ccc",
           "cds_main", "cds_interaction", "cds"]


# ---------------------------------------------------------------------------
# Joint listener
# ---------------------------------------------------------------------------

class JointSEMScenario(SEMScenario):
    """SEMScenario whose knowledge posterior marginalizes over strategies.

        P(k|u,c) = P(k) Σ_m P(m) χ_m(c,k,u) / Σ_k' P(k') Σ_m P(m) χ_m(c,k',u)

    The strategy posterior P(m|k,u,c) and the impact function are inherited
    unchanged, so E_L differs from the staged model only through P(k|u,c).
    Since every strategy satisfies m(c,k) ∈ t(k), χ_m implies χ_t.
    """

    def prob_k_joint(self, k: str, v: str, c: str) -> float:
        def weight(kk: str) -> float:
            return self.priors_K[kk] * sum(
                self.priors_M[m] * self.chi_m(m, c, kk, v)
                for m in self.strategies
            )

        z = sum(weight(kk) for kk in self.knowledge_states)
        return weight(k) / z if z > 0.0 else 0.0

    def evaluate(self, a: str, v: str, c: str) -> float:
        return sum(
            self.prob_k_joint(k, v, c)
            * self.prob_m(m, k, c, v)
            * self.impact(a, k, m)
            for k in self.knowledge_states
            for m in self.strategies
        )


def build(joint: bool, omega: float = 0.5,
          priors_K: Optional[Dict[str, float]] = None,
          priors_M: Optional[Dict[str, float]] = None) -> SEMScenario:
    """CS1 scenario, staged (default SEM) or joint."""
    scenario = build_scenario(omega=omega, priors_K=priors_K, priors_M=priors_M)
    if joint:
        scenario.__class__ = JointSEMScenario
    return scenario


def knowledge_posterior(scenario: SEMScenario, joint: bool,
                        k: str, v: str, c: str) -> float:
    if joint:
        return scenario.prob_k_joint(k, v, c)
    return scenario.prob_k(k, v)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def posteriors() -> Dict:
    out: Dict = {}
    for joint in (False, True):
        name = "joint" if joint else "staged"
        s = build(joint)
        out[name] = {
            f"{v}|{c}": knowledge_posterior(s, joint, "k_p", v, c)
            for v, c in itertools.product(["v_prc", "v_apx"], ["c_LP", "c_HP"])
        }
    # Example: a speaker expected to be maximally precise ("nerdy" prior).
    s = build(True, priors_M={"m_max": 0.8, "m_sit": 0.1, "m_min": 0.1})
    out["joint_pmax_0.8"] = {"v_apx|c_LP": s.prob_k_joint("k_p", "v_apx", "c_LP")}
    return out


def balanced() -> Dict:
    out: Dict = {}
    for joint in (False, True):
        name = "joint" if joint else "staged"
        scores = build(joint).evaluate_all()
        result = fit.evaluate(scores, EMPIRICAL_JSON)
        out[name] = {
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
    for joint in (False, True):
        name = "joint" if joint else "staged"
        results: List[Dict] = []
        for p in params:
            scores = build(joint, p["omega"], p["priors_K"], p["priors_M"]).evaluate_all()
            r = fit.evaluate(scores, EMPIRICAL_JSON)
            r["effects"] = target_effects(scores)
            results.append(r)
        out[name] = {
            "metrics": {
                k: {"mean": float(np.mean([r[k] for r in results])),
                    "sd": float(np.std([r[k] for r in results]))}
                for k in METRICS
            },
            "all_four": sum(all(r["effects"]) for r in results),
            "per_effect": [sum(r["effects"][i] for r in results) for i in range(4)],
            "n": n_samples,
        }
    return out


def sweep(step: float = 0.05) -> Dict:
    out: Dict = {}
    for joint in (False, True):
        name = "joint" if joint else "staged"
        rt = build(joint).robustness_test(target_effects, step=step)
        failures = rt["failures"]
        fail_region = {}
        if failures:
            fail_region = {
                "omega": [min(f["omega"] for f in failures),
                          max(f["omega"] for f in failures)],
                "P(m_sit)": [min(f["priors_M"]["m_sit"] for f in failures),
                             max(f["priors_M"]["m_sit"] for f in failures)],
                "P(m_min)": [min(f["priors_M"]["m_min"] for f in failures),
                             max(f["priors_M"]["m_min"] for f in failures)],
            }
        out[name] = {
            "total": rt["total"],
            "all_correct": rt["all_correct"],
            "rate_all": rt["rate_all"],
            "per_effect": rt["per_effect"],
            "failure_region": fail_region,
        }
    return out


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

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
