"""
case_studies/cs2_pragmatic_violations.py
=========================================
Case Study 2: Pragmatic Violations (Relevance & Informativeness)

Corresponds to Section 4 of the paper ("A Case Study on Pragmatic Violations")
and the original notebook ImprecisionRelInf.ipynb.

Based on: Beltrama & Papafragou (2023), "Pragmatic violations affect social
inferences about the speaker." Glossa Psycholinguistics, 2(1).

Scenario description
--------------------
Listener (Kim) raises topic A and asks for information.
Speaker (John) responds with one of four utterance types:

    v_A : highly informative about topic A (relevant + high info)
    v_a : lowly informative about topic A  (relevant + low info)
    v_B : highly informative about topic B (irrelevant + high info)
    v_b : lowly informative about topic B  (irrelevant + low info)

The listener evaluates the speaker on two attributes: competent, likeable.

Sets
----
V  = {v_A, v_a, v_B, v_b}   utterances (topic × informativeness)
C  = {c_topic}               one context (listener has introduced topic A)
A  = {a_comp, a_like}        attributes: competent, likeable
K  = {k_AB, k_Ab, k_aB, k_ab}  knowledge states (can/cannot give high-info
                                 response about each of topics A and B)
M  = {m_RQ, m_Rq, m_rQ, m_rq}  strategies: follow/violate Relevance (R/r)
                                              × follow/violate Quantity (Q/q)

Knowledge state notation
------------------------
Capital letter = speaker has highly informative knowledge of that topic.
    k_AB : high-info knowledge of both A and B
    k_Ab : high-info knowledge of A, low-info knowledge of B
    k_aB : low-info knowledge of A, high-info knowledge of B
    k_ab : low-info knowledge of both A and B

Strategy notation
-----------------
    m_RQ  : follow Relevance AND Quantity  (cooperative)
    m_Rq  : follow Relevance, violate Quantity  (under-informative but on-topic)
    m_rQ  : violate Relevance, follow Quantity  (off-topic but informative)
    m_rq  : violate both  (off-topic and under-informative)

Trivalent impact functions (Figure 7d and 7e in paper)
------------------------------------------------------
φ_K:
    k_AB : a_comp=+1, a_like= 0
    k_Ab : a_comp=+1, a_like= 0
    k_aB : a_comp=-1, a_like=-1
    k_ab : a_comp=-1, a_like=-1

φ_M:
    m_RQ : a_comp=+1, a_like=+1
    m_Rq : a_comp=+1, a_like= 0
    m_rQ : a_comp=-1, a_like=-1
    m_rq : a_comp=-1, a_like=-1

Target empirical effects (all p < .001, from Beltrama & Papafragou 2023)
------------------------------------------------------------------------
Effect 1: Relevant utterances (v_A, v_a) rated higher on competence than
          irrelevant ones (v_B, v_b)
Effect 2: Relevant utterances rated higher on likeability than irrelevant
Effect 3: High-info utterances (v_A, v_B) rated higher on competence than
          low-info (v_a, v_b)
Effect 4: High-info utterances rated higher on likeability than low-info
Effect 5: Competence gap between v_A and v_a > gap between v_B and v_b
          (informativeness matters more when relevance is obeyed)
Effect 6: Likeability gap between v_A and v_a > gap between v_B and v_b
"""

from sem.model import SEMScenario
from typing import Dict, List


# ---------------------------------------------------------------------------
# Utterances, contexts, attributes
# ---------------------------------------------------------------------------

UTTERANCES  = ["v_A", "v_a", "v_B", "v_b"]
CONTEXTS    = ["c_topic"]
ATTRIBUTES  = ["a_comp", "a_like"]
K_STATES    = ["k_AB", "k_Ab", "k_aB", "k_ab"]
STRATEGIES  = ["m_RQ", "m_Rq", "m_rQ", "m_rq"]


# ---------------------------------------------------------------------------
# Truthfulness function  t(k) → list of truthful utterances
# Paper: Figure 7(a)
#
#   k_AB : knows a lot about both A and B → can use any of the 4 utterances
#   k_Ab : knows a lot about A, little about B → can use v_A, v_a, v_b
#   k_aB : knows little about A, a lot about B → can use v_B, v_a, v_b
#   k_ab : knows little about both → can only use v_a, v_b
# ---------------------------------------------------------------------------

def truthfulness_fn(k: str) -> List[str]:
    mapping = {
        "k_AB": ["v_A", "v_a", "v_B", "v_b"],
        "k_Ab": ["v_A", "v_a", "v_b"],
        "k_aB": ["v_a", "v_B", "v_b"],
        "k_ab": ["v_a", "v_b"],
    }
    return mapping.get(k, [])


# ---------------------------------------------------------------------------
# Strategy function  m(strategy, context, knowledge_state) → list of utterances
# Paper: Figure 7(b)
#
#   m_RQ : relevant (topic A) AND informative → give best available info on A
#   m_Rq : relevant but under-informative → always give v_a (low-info on A)
#   m_rQ : irrelevant (topic B) AND informative → give best available info on B
#   m_rq : irrelevant AND under-informative → always give v_b
#
# Note: each strategy is constrained by the Gricean quality maxim —
# it can only return utterances that are truthful given the knowledge state.
# This is enforced by the truthfulness indicator χ_t in prob_k().
# ---------------------------------------------------------------------------

def strategy_fn(m: str, c: str, k: str) -> List[str]:
    if m == "m_RQ":
        # Maximally informative about topic A
        if k in ("k_AB", "k_Ab"):
            return ["v_A"]   # has high-info about A
        else:
            return ["v_a"]   # only low-info about A available

    elif m == "m_Rq":
        # On-topic but under-informative; always gives v_a
        return ["v_a"]

    elif m == "m_rQ":
        # Off-topic but informative about B
        if k in ("k_AB", "k_aB"):
            return ["v_B"]   # has high-info about B
        else:
            return ["v_b"]   # only low-info about B available

    elif m == "m_rq":
        # Off-topic and under-informative; always gives v_b
        return ["v_b"]

    return []


# ---------------------------------------------------------------------------
# Impact functions  φ_K and φ_M  (trivalent: -1, 0, +1)
# Paper: Figures 7(d) and 7(e)
# ---------------------------------------------------------------------------

IMPACT_K: Dict[str, Dict[str, int]] = {
    "k_AB": {"a_comp":  1, "a_like":  0},
    "k_Ab": {"a_comp":  1, "a_like":  0},
    "k_aB": {"a_comp": -1, "a_like": -1},
    "k_ab": {"a_comp": -1, "a_like": -1},
}

IMPACT_M: Dict[str, Dict[str, int]] = {
    "m_RQ": {"a_comp":  1, "a_like":  1},
    "m_Rq": {"a_comp":  1, "a_like":  0},
    "m_rQ": {"a_comp": -1, "a_like": -1},
    "m_rq": {"a_comp": -1, "a_like": -1},
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
    Return a SEMScenario configured for Case Study 2 (pragmatic violations).

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
    Check the 6 target empirical effects against SEM evaluation scores.

    Parameters
    ----------
    scores : dict
        Output of scenario.evaluate_all(), i.e. scores[a][v][c].

    Returns
    -------
    List of 6 booleans, one per effect.
    """
    s = scores
    c = "c_topic"

    # Shorthand helpers
    def sc(a, v):
        return s[a][v][c]

    # Effect 1: Relevant (v_A, v_a) > Irrelevant (v_B, v_b) on competence
    e1 = (sc("a_comp", "v_A") + sc("a_comp", "v_a") >
          sc("a_comp", "v_B") + sc("a_comp", "v_b"))

    # Effect 2: Relevant > Irrelevant on likeability
    e2 = (sc("a_like", "v_A") + sc("a_like", "v_a") >
          sc("a_like", "v_B") + sc("a_like", "v_b"))

    # Effect 3: High-info (v_A, v_B) > Low-info (v_a, v_b) on competence
    e3 = (sc("a_comp", "v_A") + sc("a_comp", "v_B") >
          sc("a_comp", "v_a") + sc("a_comp", "v_b"))

    # Effect 4: High-info > Low-info on likeability
    e4 = (sc("a_like", "v_A") + sc("a_like", "v_B") >
          sc("a_like", "v_a") + sc("a_like", "v_b"))

    # Effect 5: Competence gap (v_A - v_a) > (v_B - v_b)
    e5 = (sc("a_comp", "v_A") - sc("a_comp", "v_a") >
          sc("a_comp", "v_B") - sc("a_comp", "v_b"))

    # Effect 6: Likeability gap (v_A - v_a) > (v_B - v_b)
    e6 = (sc("a_like", "v_A") - sc("a_like", "v_a") >
          sc("a_like", "v_B") - sc("a_like", "v_b"))

    return [e1, e2, e3, e4, e5, e6]


# ---------------------------------------------------------------------------
# Quick run: reproduce balanced-model predictions
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenario = build_scenario(omega=0.5)
    scores   = scenario.evaluate_all()
    c        = "c_topic"

    print("=== Case Study 2: Pragmatic Violations — Balanced Model Predictions ===\n")
    print(f"{'Attribute':<10} {'Utterance':<8} {'Score':>8}")
    print("-" * 30)
    for a in ATTRIBUTES:
        for v in UTTERANCES:
            print(f"{a:<10} {v:<8} {scores[a][v][c]:>8.3f}")
        print()

    effects = target_effects(scores)
    labels  = [
        "Effect 1: Relevant > Irrelevant on competence",
        "Effect 2: Relevant > Irrelevant on likeability",
        "Effect 3: High-info > Low-info on competence",
        "Effect 4: High-info > Low-info on likeability",
        "Effect 5: Competence gap larger when relevant",
        "Effect 6: Likeability gap larger when relevant",
    ]
    print("=== Effect checks ===")
    for label, result in zip(labels, effects):
        mark = "✓" if result else "✗"
        print(f"  {mark}  {label}")
    print(f"\nAll effects correct: {all(effects)}")
