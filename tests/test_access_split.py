"""
tests/test_access_split.py
==========================
Tests for the access-split listener (sem/access_split.py).

Key guarantees tested
---------------------
1. Exp. 1 reduction: split.evaluate_exp1() == vanilla SEM with P(k_p) = pi,
   for all 12 (a, v, c) cells and pi in {.25, .5, .9}.
2. Exp. 2 with pi = 1: split.evaluate_exp2() == vanilla SEM with P(k_p) = 1
   (Session F "knowledge" model), for all 12 cells.
3. All scores in [-1, 1].
4. Staged and joint base scenarios give identical evaluate_exp2() results
   (because h is not updated by u in Exp. 2).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from case_studies.cs1_imprecision import (
    build_scenario,
    UTTERANCES, CONTEXTS, ATTRIBUTES,
)
from sem.access_split import AccessSplitScenario


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_staged():
    return build_scenario(inference="staged")


@pytest.fixture
def base_joint():
    return build_scenario(inference="joint")


# ---------------------------------------------------------------------------
# 1. Exp. 1 reduction: must equal vanilla SEM with P(k_p) = pi
# ---------------------------------------------------------------------------

class TestExp1Reduction:
    """split.evaluate_exp1(pi) == vanilla SEM evaluate(pi) for all 12 cells."""

    @pytest.mark.parametrize("pi", [0.25, 0.5, 0.9])
    def test_all_cells(self, base_staged, pi):
        split   = AccessSplitScenario(base_staged, pi=pi)
        vanilla = build_scenario(
            priors_K={"k_p": pi, "k_np": 1.0 - pi},
            inference="staged",
        )
        for a in ATTRIBUTES:
            for v in UTTERANCES:
                for c in CONTEXTS:
                    assert split.evaluate_exp1(a, v, c) == pytest.approx(
                        vanilla.evaluate(a, v, c), abs=1e-10
                    ), f"Mismatch at a={a}, v={v}, c={c}, pi={pi}"

    @pytest.mark.parametrize("pi", [0.25, 0.5, 0.9])
    def test_evaluate_all_matches_evaluate(self, base_staged, pi):
        split = AccessSplitScenario(base_staged, pi=pi)
        all_scores = split.evaluate_all_exp1(pi=pi)
        for a in ATTRIBUTES:
            for v in UTTERANCES:
                for c in CONTEXTS:
                    assert split.evaluate_exp1(a, v, c, pi=pi) == pytest.approx(
                        all_scores[a][v][c], abs=1e-10
                    )


# ---------------------------------------------------------------------------
# 2. Exp. 2 with pi = 1: must equal vanilla SEM with P(k_p) = 1
# ---------------------------------------------------------------------------

class TestExp2PiEqualsOne:
    """split.evaluate_exp2(pi=1) == vanilla SEM(P(k_p)=1) for all 12 cells."""

    def test_all_cells(self, base_staged):
        split   = AccessSplitScenario(base_staged, pi=1.0)
        vanilla = build_scenario(
            priors_K={"k_p": 1.0, "k_np": 0.0},
            inference="staged",
        )
        for a in ATTRIBUTES:
            for v in UTTERANCES:
                for c in CONTEXTS:
                    assert split.evaluate_exp2(a, v, c) == pytest.approx(
                        vanilla.evaluate(a, v, c), abs=1e-10
                    ), f"Mismatch at a={a}, v={v}, c={c}"


# ---------------------------------------------------------------------------
# 3. All scores in [-1, 1]
# ---------------------------------------------------------------------------

class TestScoreRange:
    @pytest.mark.parametrize("pi", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_exp1_scores_in_range(self, base_staged, pi):
        split = AccessSplitScenario(base_staged, pi=pi)
        scores = split.evaluate_all_exp1()
        for a in ATTRIBUTES:
            for v in UTTERANCES:
                for c in CONTEXTS:
                    s = scores[a][v][c]
                    assert -1.0 <= s <= 1.0, f"Out of range: {a},{v},{c},{pi} = {s}"

    @pytest.mark.parametrize("pi", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_exp2_scores_in_range(self, base_staged, pi):
        split = AccessSplitScenario(base_staged, pi=pi)
        scores = split.evaluate_all_exp2()
        for a in ATTRIBUTES:
            for v in UTTERANCES:
                for c in CONTEXTS:
                    s = scores[a][v][c]
                    assert -1.0 <= s <= 1.0, f"Out of range: {a},{v},{c},{pi} = {s}"


# ---------------------------------------------------------------------------
# 4. Staged == joint for Exp. 2 (h not updated by u)
# ---------------------------------------------------------------------------

class TestStagedJointEquivalenceExp2:
    """
    The split listener's evaluate_exp2 uses prob_m(m, access_state, c, v)
    which does not depend on the inference mode of the base scenario.
    Both staged and joint base scenarios must give identical Exp. 2 scores.
    """

    @pytest.mark.parametrize("pi", [0.25, 0.5, 0.9])
    def test_all_cells(self, base_staged, base_joint, pi):
        split_st = AccessSplitScenario(base_staged, pi=pi)
        split_jt = AccessSplitScenario(base_joint, pi=pi)
        for a in ATTRIBUTES:
            for v in UTTERANCES:
                for c in CONTEXTS:
                    assert split_st.evaluate_exp2(a, v, c) == pytest.approx(
                        split_jt.evaluate_exp2(a, v, c), abs=1e-10
                    ), f"Staged≠Joint at a={a}, v={v}, c={c}, pi={pi}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_pi_raises():
    base = build_scenario()
    with pytest.raises(ValueError):
        AccessSplitScenario(base, pi=1.5)


def test_pi_zero_exp2_equals_k_np(base_staged):
    """pi=0: all h=k_np, Exp.2 eval should equal strategy-only score with k_np impact."""
    split   = AccessSplitScenario(base_staged, pi=0.0)
    vanilla = build_scenario(
        priors_K={"k_p": 0.0, "k_np": 1.0},
        inference="staged",
    )
    # Note: evaluate_exp2 with pi=0 is NOT the same as vanilla SEM with P(k_p)=0,
    # because strategies still use s=s_p (access_state).
    # Verify: evaluate_exp2(pi=0) = Σ_m P(m|s_p,u,c) * impact(a, k_np, m)
    for a in ATTRIBUTES:
        for v in UTTERANCES:
            for c in CONTEXTS:
                # Manual computation
                expected = sum(
                    base_staged.prob_m(m, "k_p", c, v) * base_staged.impact(a, "k_np", m)
                    for m in base_staged.strategies
                )
                assert split.evaluate_exp2(a, v, c) == pytest.approx(
                    expected, abs=1e-10
                ), f"pi=0 mismatch at {a},{v},{c}"
