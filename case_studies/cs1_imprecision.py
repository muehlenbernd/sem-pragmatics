"""
case_studies/cs1_imprecision.py
================================
Case Study 1: Numerical (Im)precision

Corresponds to Section 3 of the paper ("A Case Study on (Im)Precision")
and the original notebook Imprecision.ipynb.

Scenario description
--------------------
A speaker reports a numerical value (e.g., the time of an event or the price
of an object) to a listener.  The speaker can express the value precisely
("It happened at 8:32" / "The bike cost $509.55") or approximately
("It happened around 8:30" / "The bike cost about $500").

The listener has heard the utterance and now forms a social evaluation of
the speaker on three attributes: competent, likeable, pedantic.

Sets
----
V  = {v_prc, v_apx}          utterances: precise / approximate
C  = {c_HP,  c_LP}           contexts:   high-precision / low-precision needed
A  = {a_comp, a_like, a_ped} attributes: competent, likeable, pedantic
K  = {k_p, k_np}             knowledge:  knows exact value / only approx. value
M  = {m_max, m_sit, m_min}   strategies: maximally precise / situationally
                                          precise / minimally precise

Trivalent impact functions — derived from Experiment 1 data at p < .01
(see data/cs1_experiment/processed/impact_tables_cs1.json for all thresholds)
-----------------------------------------------
φ_K:
    k_p  : a_comp=+1, a_like= 0, a_ped= 0
    k_np : a_comp=-1, a_like= 0, a_ped=-1

φ_M:
    m_max: a_comp= 0, a_like= 0, a_ped=+1
    m_sit: a_comp=+1, a_like=+1, a_ped= 0
    m_min: a_comp=-1, a_like= 0, a_ped= 0

Target empirical effects (all p < .001, from experiment in paper)
-----------------------------------------------------------------
Effect 1: PRECISE rated higher on COMPETENT than APPROX (across contexts)
Effect 2: PRECISE rated higher on PEDANTIC than APPROX (across contexts)
Effect 3: PRECISE–APPROX competence gap is larger in c_HP than c_LP
Effect 4: APPROX–PRECISE likeability gap is larger in c_LP than c_HP
"""

from sem.model import SEMScenario
from typing import Dict, List


# ---------------------------------------------------------------------------
# Utterances, contexts, attributes
# ---------------------------------------------------------------------------

UTTERANCES  = ["v_prc", "v_apx"]
CONTEXTS    = ["c_HP", "c_LP"]
ATTRIBUTES  = ["a_comp", "a_like", "a_ped"]
K_STATES    = ["k_p", "k_np"]
STRATEGIES  = ["m_max", "m_sit", "m_min"]


# ---------------------------------------------------------------------------
# Truthfulness function  t(k) → list of truthful utterances
# Paper: Figure 4(a)
#
#   k_p  (knows exact value):  can say precise OR approximate
#   k_np (only approx. knowledge): can only say approximate
# ---------------------------------------------------------------------------

def truthfulness_fn(k: str) -> List[str]:
    if k == "k_p":
        return ["v_prc", "v_apx"]
    elif k == "k_np":
        return ["v_apx"]
    return []


# ---------------------------------------------------------------------------
# Strategy function  m(strategy, context, knowledge_state) → list of utterances
# Paper: Figure 4(b)
#
#   m_max  always maximally precise (as long as knowledge permits)
#   m_sit  precise only when context demands precision (c_HP)
#   m_min  always uses approximate form
# ---------------------------------------------------------------------------

def strategy_fn(m: str, c: str, k: str) -> List[str]:
    if m == "m_max":
        # Always precise if knowledge allows
        if k == "k_p":
            return ["v_prc"]
        else:
            return ["v_apx"]

    elif m == "m_sit":
        # Precise only in high-precision context AND if knowledge allows
        if k == "k_p" and c == "c_HP":
            return ["v_prc"]
        else:
            return ["v_apx"]

    elif m == "m_min":
        # Always approximate, regardless of knowledge or context
        return ["v_apx"]

    return []


# ---------------------------------------------------------------------------
# Impact functions  φ_K and φ_M  (trivalent: -1, 0, +1)
# Paper: Figure 5(d)
# ---------------------------------------------------------------------------

IMPACT_K: Dict[str, Dict[str, int]] = {
    "k_p":  {"a_comp":  1, "a_like":  0, "a_ped":  0},  # unchanged
    "k_np": {"a_comp": -1, "a_like":  0, "a_ped": -1},  # a_like: -1 → 0
}

IMPACT_M: Dict[str, Dict[str, int]] = {
    "m_max": {"a_comp":  0, "a_like":  0, "a_ped":  1},  # a_like: -1 → 0
    "m_sit": {"a_comp":  1, "a_like":  1, "a_ped":  0},  # a_ped: -1 → 0
    "m_min": {"a_comp": -1, "a_like":  0, "a_ped":  0},  # a_like: -1 → 0
}


# ---------------------------------------------------------------------------
# Factory: build a configured SEMScenario for this case study
# ---------------------------------------------------------------------------

def build_scenario(
    omega: float = 0.5,
    priors_K: Dict[str, float] = None,
    priors_M: Dict[str, float] = None,
) -> SEMScenario:
    """
    Return a SEMScenario configured for Case Study 1 (imprecision).

    Parameters
    ----------
    omega : float
        Weight ω for knowledge vs. motivation (default 0.5 = balanced model).
    priors_K : dict, optional
        Prior probabilities over knowledge states. Uniform if None.
    priors_M : dict, optional
        Prior probabilities over strategies. Uniform if None.
    """
    return SEMScenario(
        utterances       = UTTERANCES,
        contexts         = CONTEXTS,
        attributes       = ATTRIBUTES,
        knowledge_states = K_STATES,
        strategies       = STRATEGIES,
        truthfulness_fn  = truthfulness_fn,
        strategy_fn      = strategy_fn,
        impact_K         = IMPACT_K,
        impact_M         = IMPACT_M,
        priors_K         = priors_K or {},
        priors_M         = priors_M or {},
        omega            = omega,
    )


# ---------------------------------------------------------------------------
# Effect definitions (for robustness testing)
# ---------------------------------------------------------------------------

def target_effects(scores: Dict) -> List[bool]:
    """
    Check the 4 target empirical effects against SEM evaluation scores.

    Parameters
    ----------
    scores : dict
        Output of scenario.evaluate_all(), i.e. scores[a][v][c].

    Returns
    -------
    List of 4 booleans, one per effect:
        [effect_1, effect_2, effect_3, effect_4]
    """
    s = scores  # shorthand

    # Effect 1: PRECISE > APPROX on competence (summed across contexts)
    e1 = (s["a_comp"]["v_prc"]["c_HP"] + s["a_comp"]["v_prc"]["c_LP"] >
          s["a_comp"]["v_apx"]["c_HP"] + s["a_comp"]["v_apx"]["c_LP"])

    # Effect 2: PRECISE > APPROX on pedantry (summed across contexts)
    e2 = (s["a_ped"]["v_prc"]["c_HP"] + s["a_ped"]["v_prc"]["c_LP"] >
          s["a_ped"]["v_apx"]["c_HP"] + s["a_ped"]["v_apx"]["c_LP"])

    # Effect 3: Competence gap (prc - apx) larger in c_HP than c_LP
    e3 = ((s["a_comp"]["v_prc"]["c_HP"] - s["a_comp"]["v_apx"]["c_HP"]) >
          (s["a_comp"]["v_prc"]["c_LP"] - s["a_comp"]["v_apx"]["c_LP"]))

    # Effect 4: Likeability gap (prc - apx) larger in c_HP than c_LP
    #   i.e., approx is relatively MORE likeable in c_LP
    e4 = ((s["a_like"]["v_prc"]["c_HP"] - s["a_like"]["v_apx"]["c_HP"]) >
          (s["a_like"]["v_prc"]["c_LP"] - s["a_like"]["v_apx"]["c_LP"]))

    return [e1, e2, e3, e4]


# ---------------------------------------------------------------------------
# Quantitative fit evaluation (Task 1)
# ---------------------------------------------------------------------------

def run_fit_evaluation(omega=0.5, priors_K=None, priors_M=None,
                       empirical_json_path=None):
    """Run quantitative fit evaluation for the CS1 balanced model.

    Builds the scenario, computes model scores, compares them to the empirical
    human means via ``sem.fit.evaluate``, prints a formatted summary, and returns
    the result dict.
    """
    from sem import fit

    if empirical_json_path is None:
        empirical_json_path = "data/cs1_experiment/processed/empirical_means_cs1.json"

    scenario     = build_scenario(omega=omega, priors_K=priors_K, priors_M=priors_M)
    model_scores = scenario.evaluate_all()
    result       = fit.evaluate(model_scores, empirical_json_path)

    # Counts for the structural-alignment display (matches/considered). DAS and
    # ISS are restricted to the statistically significant effects, so the counts
    # iterate the corresponding significant-attribute lists (see sem/fit.py).
    attrs = ["a_comp", "a_like", "a_ped"]
    sig_main = ["a_comp", "a_ped"]
    sig_interaction = ["a_comp", "a_like"]
    he = fit.extract_effects(fit.load_empirical_means(empirical_json_path), attrs)
    me = fit.extract_effects(model_scores, attrs)
    sgn = lambda x: (x > 0) - (x < 0)

    def counts(key, sig_attrs):
        considered = [a for a in sig_attrs if he[a][key] != 0]
        match = sum(1 for a in considered if sgn(me[a][key]) == sgn(he[a][key]))
        return match, len(considered)

    dm, dt = counts("main_effect", sig_main)
    im, it = counts("interaction", sig_interaction)

    print(f"=== CS1 Fit Evaluation (balanced model: ω={omega}, flat priors) ===\n")
    print("Global metrics:")
    print(f"  Spearman ρ : {result['spearman_r']:.2f}")
    print(f"  RMSE       : {result['rmse']:.2f}")
    print(f"  CCC        : {result['ccc']:.2f}\n")
    print("Structural alignment:")
    print(f"  DAS (main effects)    : {result['das']:.2f}  ({dm}/{dt})")
    print(f"  ISS (interactions)    : {result['iss']:.2f}  ({im}/{it})\n")
    print("Magnitude calibration (significant effects only):")
    print("  ESR main effects:")
    for a, v in result["esr_main"].items():
        print(f"    {a:6s} : {v:.2f}")
    print("  ESR interactions:")
    for a, v in result["esr_interaction"].items():
        print(f"    {a:6s} : {v:.2f}")
    print(f"  CDS (main)        : {result['cds_main']:.2f}")
    print(f"  CDS (interaction) : {result['cds_interaction']:.2f}")
    print(f"  CDS (overall)     : {result['cds']:.2f}")
    return result


# ---------------------------------------------------------------------------
# Quick run: reproduce balanced-model predictions
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenario = build_scenario(omega=0.5)
    scores   = scenario.evaluate_all()

    print("=== Case Study 1: (Im)Precision — Balanced Model Predictions ===\n")
    print(f"{'Attribute':<10} {'Utterance':<8} {'c_HP':>8} {'c_LP':>8}")
    print("-" * 38)
    for a in ATTRIBUTES:
        for v in UTTERANCES:
            hp = scores[a][v]["c_HP"]
            lp = scores[a][v]["c_LP"]
            print(f"{a:<10} {v:<8} {hp:>8.3f} {lp:>8.3f}")
        print()

    effects = target_effects(scores)
    labels  = [
        "Effect 1: PRECISE > APPROX on competence",
        "Effect 2: PRECISE > APPROX on pedantry",
        "Effect 3: Competence gap larger in c_HP",
        "Effect 4: Likeability gap larger in c_HP",
    ]
    print("=== Effect checks ===")
    for label, passed in zip(labels, effects):
        mark = "✓" if passed else "✗"
        print(f"  {mark}  {label}")
    print(f"\nAll effects correct: {all(effects)}")

    import os
    _json = "data/cs1_experiment/processed/empirical_means_cs1.json"
    print()
    if os.path.exists(_json):
        run_fit_evaluation(empirical_json_path=_json)
    else:
        print(f"(skipping fit evaluation: {_json} not found; "
              f"run scripts/compute_empirical_means.py first)")

    print()
    try:
        from sem.comparison import compare_all_variants, print_comparison_table
        comparison = compare_all_variants(empirical_json_path=_json)
        print_comparison_table(comparison)
    except FileNotFoundError:
        print(f"(skipping variant comparison: {_json} not found; "
              f"run scripts/compute_empirical_means.py first)")
