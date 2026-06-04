"""
sem/comparison.py
=================
Compare four nested SEM variants via parameter sampling to assess which
inferential components are necessary for reproducing the CS1 empirical effects.

All four variants reuse the existing ``SEMScenario`` machinery (via
``case_studies.cs1_imprecision.build_scenario``) with constrained parameters —
this is NOT a separate baseline model:

    full              all parameters free
    context_blind     P(m_sit) fixed at 0   (removes context-sensitive motivation)
    knowledge_only    omega fixed at 1.0    (removes motivation inference)
    motivation_only   omega fixed at 0.0    (removes knowledge-state inference)

Each variant is evaluated across ``n_samples`` sampled parameter combinations,
yielding a distribution over fit metrics per variant.  This makes the comparison
robust to the choice of a single parameter setting.

Self-contained: depends only on numpy, ``sem.fit``, and
``case_studies.cs1_imprecision``.
"""

from typing import Dict, List

import numpy as np

from sem import fit
from case_studies.cs1_imprecision import build_scenario


VARIANTS = ("full", "context_blind", "knowledge_only", "motivation_only")


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def sample_parameters(variant: str, n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """
    Draw ``n_samples`` parameter combinations for ``variant``.

    Returns a list of dicts with keys ``omega``, ``priors_K``, ``priors_M``.
    Uses ``numpy.random.default_rng(seed)`` for reproducibility; all samples are
    drawn vectorized.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")

    rng = np.random.default_rng(seed)

    # --- omega ---
    if variant in ("full", "context_blind"):
        omegas = np.clip(rng.beta(2.0, 2.0, size=n_samples), 0.05, 0.95)
    elif variant == "knowledge_only":
        omegas = np.full(n_samples, 1.0)
    else:  # motivation_only
        omegas = np.full(n_samples, 0.0)

    # --- P(k): uniform over the 2-simplex ---
    p_k = rng.dirichlet([1.0, 1.0], size=n_samples)            # (n, 2)

    # --- P(m) ---
    if variant == "context_blind":
        # m_sit fixed at 0; draw (m_max, m_min) over the 2-simplex.
        p_m2 = rng.dirichlet([1.0, 1.0], size=n_samples)       # (n, 2)
    else:
        p_m = rng.dirichlet([1.0, 1.0, 1.0], size=n_samples)   # (n, 3)

    samples: List[Dict] = []
    for i in range(n_samples):
        priors_K = {"k_p": float(p_k[i, 0]), "k_np": float(p_k[i, 1])}
        if variant == "context_blind":
            priors_M = {"m_max": float(p_m2[i, 0]),
                        "m_sit": 0.0,
                        "m_min": float(p_m2[i, 1])}
        else:
            priors_M = {"m_max": float(p_m[i, 0]),
                        "m_sit": float(p_m[i, 1]),
                        "m_min": float(p_m[i, 2])}
        samples.append({"omega": float(omegas[i]),
                        "priors_K": priors_K,
                        "priors_M": priors_M})

    # Defensive validation of every sample.
    for s in samples:
        assert abs(sum(s["priors_K"].values()) - 1.0) < 1e-9
        assert abs(sum(s["priors_M"].values()) - 1.0) < 1e-9
        assert all(v > 0 for v in s["priors_K"].values())
        if variant == "context_blind":
            assert s["priors_M"]["m_sit"] == 0.0
            assert s["priors_M"]["m_max"] > 0 and s["priors_M"]["m_min"] > 0
            assert 0.05 <= s["omega"] <= 0.95
        else:
            assert all(v > 0 for v in s["priors_M"].values())
        if variant == "knowledge_only":
            assert s["omega"] == 1.0
        elif variant == "motivation_only":
            assert s["omega"] == 0.0

    return samples


# ---------------------------------------------------------------------------
# Per-variant evaluation
# ---------------------------------------------------------------------------

def evaluate_variant(
    variant: str,
    empirical_json_path: str,
    n_samples: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """
    Evaluate ``variant`` across ``n_samples`` sampled parameter combinations.

    Returns a list of per-sample result dicts (the output of ``sem.fit.evaluate``
    augmented with the sampled parameters under ``_params``).
    """
    # Sanity check that the P(m_sit)=0 constraint actually changes the model.
    if variant == "context_blind":
        base_scores  = build_scenario(omega=0.5).evaluate_all()
        blind_scores = build_scenario(
            omega=0.5,
            priors_M={"m_max": 0.5, "m_sit": 0.0, "m_min": 0.5},
        ).evaluate_all()
        assert base_scores != blind_scores, (
            "P(m_sit)=0 produced identical scores to the full model — "
            "the context-blind constraint is not being applied."
        )

    samples = sample_parameters(variant, n_samples=n_samples, seed=seed)

    results: List[Dict] = []
    for s in samples:
        scenario = build_scenario(
            omega=s["omega"], priors_K=s["priors_K"], priors_M=s["priors_M"]
        )
        model_scores = scenario.evaluate_all()
        metrics = fit.evaluate(model_scores, empirical_json_path)
        metrics["_params"] = s
        results.append(metrics)

    return results


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

_SCALAR_METRICS = [
    "das", "iss", "spearman_r", "rmse", "ccc",
    "cds_main", "cds_interaction", "cds",
]


def summarize_results(results: List[Dict]) -> Dict:
    """
    Compute mean and SD across samples for each scalar metric, and per-attribute
    mean/SD for the ESR dicts.
    """
    out: Dict = {}
    for key in _SCALAR_METRICS:
        vals = np.array([r[key] for r in results], dtype=float)
        out[key] = {"mean": float(vals.mean()), "sd": float(vals.std())}

    for esr_key in ("esr_main", "esr_interaction"):
        out[esr_key] = {}
        for attr in results[0][esr_key]:
            vals = np.array([r[esr_key][attr] for r in results], dtype=float)
            out[esr_key][attr] = {"mean": float(vals.mean()), "sd": float(vals.std())}

    return out


def compare_all_variants(
    empirical_json_path: str,
    n_samples: int = 100,
    seed: int = 42,
) -> Dict:
    """Run ``evaluate_variant`` + ``summarize_results`` for all four variants."""
    return {
        variant: summarize_results(
            evaluate_variant(variant, empirical_json_path, n_samples, seed)
        )
        for variant in VARIANTS
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _cell(summary: Dict, key: str) -> str:
    s = summary[key]
    return f"{s['mean']:.2f} ± {s['sd']:.2f}"


def _esr_cell(summary: Dict, esr_key: str, attr: str) -> str:
    s = summary[esr_key].get(attr)
    if s is None:
        return "—"
    return f"{s['mean']:.2f} ± {s['sd']:.2f}"


def print_comparison_table(comparison: Dict) -> None:
    """Print the formatted four-variant comparison table to stdout."""
    cols = ["full", "context_blind", "knowledge_only", "motivation_only"]
    headers = ["SEM-full", "Context-blind", "Know-only", "Motiv-only"]

    print("=== SEM Variant Comparison ===\n")
    head = f"{'Metric':<16}" + "".join(f"{h:<16}" for h in headers)
    print(head)
    print("─" * len(head))

    rows = [
        ("DAS", "das"), ("ISS", "iss"), ("Spearman ρ", "spearman_r"),
        ("RMSE", "rmse"), ("CCC", "ccc"),
        ("CDS (main)", "cds_main"), ("CDS (inter)", "cds_interaction"),
        ("CDS (overall)", "cds"),
    ]
    for label, key in rows:
        line = f"{label:<16}" + "".join(
            f"{_cell(comparison[c], key):<16}" for c in cols
        )
        print(line)

    print("\nESR main effects (significant only):")
    for attr in comparison["full"]["esr_main"]:
        line = f"  {attr:<14}" + "".join(
            f"{_esr_cell(comparison[c], 'esr_main', attr):<16}" for c in cols
        )
        print(line)
    print("ESR interactions (significant only):")
    for attr in comparison["full"]["esr_interaction"]:
        line = f"  {attr:<14}" + "".join(
            f"{_esr_cell(comparison[c], 'esr_interaction', attr):<16}" for c in cols
        )
        print(line)
