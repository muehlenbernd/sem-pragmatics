"""
sem/model.py
============
Core implementation of the Social Evaluation Model (SEM).

Paper reference:
    "Modeling Pragmatic Reasoning behind Social Meaning"
    Journal: Open Mind (submitted); target resubmission: Meaning

Mathematical summary
--------------------
The listener's evaluation function (Eq. 1 in the paper) is:

    E_L(a | v, c) = Σ_k Σ_m  P(k|v) · P(m|k,v,c) · I(a|k,m; ω)

where:
    v  ∈ V   — utterance chosen by the speaker
    c  ∈ C   — situational context
    k  ∈ K   — knowledge state of the speaker
    m  ∈ M   — motivation-based strategy of the speaker
    a  ∈ A   — social attribute being evaluated

The three components are:

(1) P(k|v)  — knowledge state posterior (Eq. 2):
        P(k|v) = P(k) · χ_t(k,v) / Σ_{k'} P(k') · χ_t(k',v)
    χ_t(k,v) = 1 if v is truthful under k, else 0

(2) P(m|k,v,c)  — strategy posterior (Eq. 3):
        P(m|k,v,c) = P(m) · χ_m(c,k,v) / Σ_{m'} P(m') · χ_m(c,k',v)
    χ_m(c,k,v) = 1 if strategy m selects v in context c with knowledge k

(3) I(a|k,m; ω)  — impact function (Eq. 4):
        I(a|k,m; ω) = ω · φ_K(k,a) + (1-ω) · φ_M(m,a)
    φ_K, φ_M ∈ {-1, 0, 1}  (trivalent impact functions)
    ω ∈ (0,1)               (weight: knowledge vs. motivation)

The evaluation score E_L ∈ [-1, 1].

Usage
-----
    from sem.model import SEMScenario

    scenario = SEMScenario(
        utterances=["v_prc", "v_apx"],
        contexts=["c_HP", "c_LP"],
        attributes=["a_comp", "a_like", "a_ped"],
        knowledge_states=["k_p", "k_np"],
        strategies=["m_max", "m_sit", "m_min"],
        truthfulness_fn=my_truthfulness_fn,
        strategy_fn=my_strategy_fn,
        impact_K=my_impact_K,
        impact_M=my_impact_M,
    )
    score = scenario.evaluate("a_comp", "v_prc", "c_HP")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List


# ---------------------------------------------------------------------------
# Type aliases for readability
# ---------------------------------------------------------------------------
Utterance  = str
Context    = str
Attribute  = str
KnowState  = str
Strategy   = str

TruthFn    = Callable[[KnowState], List[Utterance]]
StrategyFn = Callable[[Strategy, Context, KnowState], List[Utterance]]

ImpactDict = Dict[str, Dict[str, int]]   # e.g. impact_K[k][a] ∈ {-1, 0, 1}
PriorDict  = Dict[str, float]


# ---------------------------------------------------------------------------
# SEMScenario — the main class
# ---------------------------------------------------------------------------

@dataclass
class SEMScenario:
    """
    A fully configured SEM scenario.

    Parameters
    ----------
    utterances : list[str]
        The set V of possible utterances.
    contexts : list[str]
        The set C of situational contexts.
    attributes : list[str]
        The set A of social attributes to be evaluated.
    knowledge_states : list[str]
        The set K of possible speaker knowledge states.
    strategies : list[str]
        The set M of motivation-based strategies.
    truthfulness_fn : callable
        t(k) → list of utterances that are truthful under knowledge state k.
    strategy_fn : callable
        m(strategy, context, knowledge_state) → list of utterances the
        strategy would produce in that (c, k) combination.
    impact_K : dict
        φ_K: impact_K[k][a] ∈ {-1, 0, 1}.
        Effect of knowledge state k on attribute a.
    impact_M : dict
        φ_M: impact_M[m][a] ∈ {-1, 0, 1}.
        Effect of strategy m on attribute a.
    priors_K : dict, optional
        Prior probabilities P(k). Uniform if not provided.
    priors_M : dict, optional
        Prior probabilities P(m). Uniform if not provided.
    omega : float, optional
        Weight ω ∈ (0,1) for knowledge vs. motivation in the impact function.
        Default 0.5 (balanced model).
    """

    utterances      : List[Utterance]
    contexts        : List[Context]
    attributes      : List[Attribute]
    knowledge_states: List[KnowState]
    strategies      : List[Strategy]
    truthfulness_fn : TruthFn
    strategy_fn     : StrategyFn
    impact_K        : ImpactDict
    impact_M        : ImpactDict
    priors_K        : PriorDict = field(default_factory=dict)
    priors_M        : PriorDict = field(default_factory=dict)
    omega           : float = 0.5

    def __post_init__(self):
        # Set uniform priors if not provided
        if not self.priors_K:
            n = len(self.knowledge_states)
            self.priors_K = {k: 1.0 / n for k in self.knowledge_states}
        if not self.priors_M:
            n = len(self.strategies)
            self.priors_M = {m: 1.0 / n for m in self.strategies}

    # ------------------------------------------------------------------
    # Indicator functions
    # ------------------------------------------------------------------

    def chi_t(self, k: KnowState, v: Utterance) -> int:
        """
        χ_t(k, v) — truthfulness indicator.
        Returns 1 if v is a truthful utterance under knowledge state k.
        """
        return 1 if v in self.truthfulness_fn(k) else 0

    def chi_m(self, m: Strategy, c: Context, k: KnowState, v: Utterance) -> int:
        """
        χ_m(c, k, v) — strategy indicator.
        Returns 1 if strategy m produces utterance v in context c
        with knowledge state k.
        """
        return 1 if v in self.strategy_fn(m, c, k) else 0

    # ------------------------------------------------------------------
    # Probability functions (Equations 2 and 3)
    # ------------------------------------------------------------------

    def prob_k(self, k: KnowState, v: Utterance) -> float:
        """
        P(k|v) — posterior probability over knowledge states given utterance v.
        Equation 2 in the paper.
        """
        numerator   = self.priors_K[k] * self.chi_t(k, v)
        denominator = sum(
            self.priors_K[k2] * self.chi_t(k2, v)
            for k2 in self.knowledge_states
        )
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    def prob_m(self, m: Strategy, k: KnowState, c: Context, v: Utterance) -> float:
        """
        P(m|k,v,c) — posterior probability over strategies given k, v, c.
        Equation 3 in the paper.
        """
        numerator   = self.priors_M[m] * self.chi_m(m, c, k, v)
        denominator = sum(
            self.priors_M[m2] * self.chi_m(m2, c, k, v)
            for m2 in self.strategies
        )
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    # ------------------------------------------------------------------
    # Impact function (Equation 4)
    # ------------------------------------------------------------------

    def impact(self, a: Attribute, k: KnowState, m: Strategy) -> float:
        """
        I(a|k,m; ω) — impact of knowledge state k and strategy m on attribute a.
        Equation 4 in the paper.
        """
        phi_K = self.impact_K[k][a]
        phi_M = self.impact_M[m][a]
        return self.omega * phi_K + (1.0 - self.omega) * phi_M

    # ------------------------------------------------------------------
    # Evaluation function (Equation 1)  — the main SEM output
    # ------------------------------------------------------------------

    def evaluate(self, a: Attribute, v: Utterance, c: Context) -> float:
        """
        E_L(a | v, c) — listener's social evaluation of attribute a,
        given that the speaker used utterance v in context c.

        Returns a score in [-1, 1], where:
            -1 = lowest possible rating
             0 = neutral
            +1 = highest possible rating

        Equation 1 in the paper.
        """
        score = 0.0
        for k in self.knowledge_states:
            for m in self.strategies:
                score += (
                    self.prob_k(k, v)
                    * self.prob_m(m, k, c, v)
                    * self.impact(a, k, m)
                )
        return score

    # ------------------------------------------------------------------
    # Convenience: evaluate all (a, v, c) combinations
    # ------------------------------------------------------------------

    def evaluate_all(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Returns a nested dict of all evaluation scores:
            result[a][v][c] = E_L(a|v,c)
        """
        result = {}
        for a in self.attributes:
            result[a] = {}
            for v in self.utterances:
                result[a][v] = {}
                for c in self.contexts:
                    result[a][v][c] = self.evaluate(a, v, c)
        return result

    # ------------------------------------------------------------------
    # Robustness test: sweep the full free-parameter space
    # ------------------------------------------------------------------

    def robustness_test(
        self,
        effects_fn: Callable[[Dict], List[bool]],
        step: float = 0.05,
        omega_range: tuple = (0.05, 0.95),
    ) -> dict:
        """
        Sweep all free parameters (ω, P(k), P(m)) and check whether the
        model correctly predicts the target empirical effects at each
        parameter combination.

        Parameters
        ----------
        effects_fn : callable
            Takes the output of evaluate_all() and returns a list of bool,
            one per target effect.  True = effect correctly predicted.
        step : float
            Granularity of the parameter sweep (default 0.05 → 20 steps).
        omega_range : tuple
            (min, max) values for ω sweep (inclusive).

        Returns
        -------
        dict with keys:
            'total'         : int   — number of parameter combinations tested
            'all_correct'   : int   — combinations where ALL effects hold
            'per_effect'    : list  — count per effect
            'rate_all'      : float — fraction of combos with all effects correct
            'failures'      : list  — param combos where at least one effect fails
                                      (each entry is a dict with params + which
                                       effects failed)
        """
        import itertools

        steps_list = [round(i * step, 6) for i in range(1, round(1.0 / step))]

        # ω sweep
        omega_values = [
            round(o, 6) for o in steps_list
            if omega_range[0] <= round(o, 6) <= omega_range[1]
        ]

        # Generate all valid prior distributions over n states with given step
        def _prior_combinations(states: List[str]) -> List[PriorDict]:
            """All distributions over `states` whose values sum to 1.0
            and are multiples of `step`."""
            n = len(states)
            combos = []
            int_steps = round(1.0 / step)

            def recurse(remaining_states, remaining_mass, current):
                if len(remaining_states) == 1:
                    val = round(remaining_mass, 6)
                    if val > 0:
                        current[remaining_states[0]] = val
                        combos.append(dict(current))
                    return
                s = remaining_states[0]
                rest = remaining_states[1:]
                for i in range(1, round(remaining_mass / step)):
                    val = round(i * step, 6)
                    current[s] = val
                    recurse(rest, round(remaining_mass - val, 6), current)
                current.pop(s, None)

            recurse(states, 1.0, {})
            return combos

        k_combos = _prior_combinations(self.knowledge_states)
        m_combos = _prior_combinations(self.strategies)

        total       = 0
        all_correct = 0
        n_effects   = None
        per_effect  = None
        failures    = []

        # Save original priors and omega so we can restore them
        orig_priors_K = dict(self.priors_K)
        orig_priors_M = dict(self.priors_M)
        orig_omega    = self.omega

        for omega in omega_values:
            for pk in k_combos:
                for pm in m_combos:
                    self.omega    = omega
                    self.priors_K = pk
                    self.priors_M = pm

                    scores  = self.evaluate_all()
                    effects = effects_fn(scores)

                    if n_effects is None:
                        n_effects  = len(effects)
                        per_effect = [0] * n_effects

                    total += 1
                    if all(effects):
                        all_correct += 1
                    else:
                        failures.append({
                            "omega": omega,
                            "priors_K": dict(pk),
                            "priors_M": dict(pm),
                            "effects":  effects,
                        })

                    for i, e in enumerate(effects):
                        if e:
                            per_effect[i] += 1

        # Restore
        self.priors_K = orig_priors_K
        self.priors_M = orig_priors_M
        self.omega    = orig_omega

        return {
            "total":       total,
            "all_correct": all_correct,
            "per_effect":  per_effect,
            "rate_all":    all_correct / total if total > 0 else 0.0,
            "failures":    failures,
        }
