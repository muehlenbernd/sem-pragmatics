"""
sem/access_split.py
===================
Access-split listener for SEM.

See docs/access_split.md for the model description.

The split model distinguishes two knowledge variables:

- **access** s: what the speaker has at hand when speaking.
  Drives the truthfulness function and the strategy indicators.
- **prior knowledge** h: whether the speaker knew the value beforehand.
  Drives the impact function φ_K.

Constraint: h_p implies s_p.

Usage
-----
    from case_studies.cs1_imprecision import build_scenario
    from sem.access_split import AccessSplitScenario

    base  = build_scenario()   # balanced SEM, inference="staged"
    split = AccessSplitScenario(base, pi=0.5)

    # Exp. 1 (s = h): identical to vanilla SEM with P(k_p) = 0.5
    e1 = split.evaluate_all_exp1()

    # Exp. 2 (s = s_p fixed, P(h_p) = pi)
    e2 = split.evaluate_all_exp2()

    # Sweep over pi
    for pi in [0.25, 0.5, 0.75, 1.0]:
        e2_pi = split.evaluate_all_exp2(pi=pi)
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Dict, Optional

from sem.model import SEMScenario


@dataclass
class AccessSplitScenario:
    """
    Access-split listener: access (s) drives truthfulness/strategies;
    prior knowledge (h) drives the impact function.

    Parameters
    ----------
    base : SEMScenario
        A fully configured SEMScenario.  Its truthfulness_fn, strategy_fn,
        impact_K, and impact_M are reused without modification.
    pi : float
        P(h_p) — prior probability that the speaker knew the value
        beforehand ("by heart").  Default 0.5 (balanced).
    access_state : str, optional
        The knowledge-state label corresponding to full access (s_p).
        Defaults to the first element of base.knowledge_states ("k_p" in CS1).
    """

    base: SEMScenario
    pi: float = 0.5
    access_state: Optional[str] = None

    def __post_init__(self):
        if self.access_state is None:
            self.access_state = self.base.knowledge_states[0]
        if not 0.0 <= self.pi <= 1.0:
            raise ValueError(f"pi must be in [0, 1], got {self.pi}")

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _h_prior(self, pi: float) -> Dict[str, float]:
        """Prior P(h) = {access_state: pi, others: (1-pi)/n}."""
        others = [s for s in self.base.knowledge_states if s != self.access_state]
        n = len(others)
        return {
            self.access_state: pi,
            **{s: (1.0 - pi) / n for s in others},
        }

    def _base_with_priors_k(self, pi: float) -> SEMScenario:
        """Shallow copy of base with priors_K set to pi."""
        sc = copy(self.base)
        sc.priors_K = self._h_prior(pi)
        return sc

    # ------------------------------------------------------------------
    # Experiment 1: s = h  →  vanilla SEM with P(k_p) = pi
    # ------------------------------------------------------------------

    def evaluate_exp1(self, a: str, v: str, c: str,
                      pi: float = None) -> float:
        """
        Exp. 1 (source not shown, s = h).
        Reduces exactly to vanilla SEM with P(k_p) = pi.
        """
        pi = self.pi if pi is None else pi
        return self._base_with_priors_k(pi).evaluate(a, v, c)

    def evaluate_all_exp1(self, pi: float = None) -> Dict:
        """evaluate_all() under Exp. 1 semantics (s = h, P(k_p) = pi)."""
        pi = self.pi if pi is None else pi
        return self._base_with_priors_k(pi).evaluate_all()

    # ------------------------------------------------------------------
    # Experiment 2: s = s_p fixed  →  P(h) unchanged by utterance
    # ------------------------------------------------------------------

    def evaluate_exp2(self, a: str, v: str, c: str,
                      pi: float = None) -> float:
        """
        Exp. 2 (source consulted, s = s_p for every speaker).

        P(h) = pi (flat prior, not updated by utterance because
        truthfulness depends on s, not h).  Strategy inference uses
        access state s_p.

            E_L(a|u,c) = Σ_h P(h) · Σ_m P(m|s_p,u,c) · I(a|h,m; ω)

        Note: the staged and joint listeners are equivalent here because
        the strategies depend on s (fixed to s_p), not on h, so the
        joint update P(h|u,c) reduces to the prior P(h).
        """
        pi = self.pi if pi is None else pi
        h_prior = self._h_prior(pi)
        score = 0.0
        for h, p_h in h_prior.items():
            for m in self.base.strategies:
                p_m = self.base.prob_m(m, self.access_state, c, v)
                imp = self.base.impact(a, h, m)
                score += p_h * p_m * imp
        return score

    def evaluate_all_exp2(self, pi: float = None) -> Dict:
        """evaluate_all() under Exp. 2 semantics."""
        pi = self.pi if pi is None else pi
        result = {}
        for a in self.base.attributes:
            result[a] = {}
            for v in self.base.utterances:
                result[a][v] = {}
                for c in self.base.contexts:
                    result[a][v][c] = self.evaluate_exp2(a, v, c, pi=pi)
        return result
