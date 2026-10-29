"""
tests/test_inference.py
=======================
Tests for the staged vs. joint knowledge inference option (SEMScenario.inference).

Cross-checks:
  - Joint flat-prior posteriors from the manuscript (§5.5 footnote).
  - Precise-utterance posterior = 1 under both listeners.
  - 1/6 example (P(m_max) = .8, P(m_sit) = P(m_min) = .1, flat P(k)).
  - Staged balanced-model scores are unchanged after the refactor.
  - Sampled RMSE / CDS match the E-session reference values (seed 42).
  - Slow: .02 grid — staged 99.87 %, joint 98.86 % (2,593,080 settings each).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from case_studies.cs1_imprecision import build_scenario, target_effects
from sem.comparison import evaluate_variant, summarize_results

_MEANS = os.path.join(
    os.path.dirname(__file__), "..",
    "data", "cs1_experiment", "processed", "empirical_means_cs1.json",
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_inference_raises():
    with pytest.raises(ValueError):
        build_scenario(inference="flat")


# ---------------------------------------------------------------------------
# Knowledge posteriors — flat priors, v_apx
# ---------------------------------------------------------------------------

def test_staged_flat_prior_approx():
    sc = build_scenario(inference="staged")
    assert sc.prob_k("k_p", "v_apx", "c_LP") == pytest.approx(0.5)
    assert sc.prob_k("k_p", "v_apx", "c_HP") == pytest.approx(0.5)


def test_joint_flat_prior_approx():
    sc = build_scenario(inference="joint")
    assert sc.prob_k("k_p", "v_apx", "c_LP") == pytest.approx(0.40)
    assert sc.prob_k("k_p", "v_apx", "c_HP") == pytest.approx(0.25)


def test_joint_flat_prior_precise():
    """P(k_p | v_prc, c) = 1 under joint (only k_p can utter v_prc)."""
    sc = build_scenario(inference="joint")
    assert sc.prob_k("k_p", "v_prc", "c_LP") == pytest.approx(1.0)
    assert sc.prob_k("k_p", "v_prc", "c_HP") == pytest.approx(1.0)


def test_staged_flat_prior_precise():
    """P(k_p | v_prc) = 1 under staged."""
    sc = build_scenario(inference="staged")
    assert sc.prob_k("k_p", "v_prc") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 1/6 example (§5.5 footnote cross-check)
# ---------------------------------------------------------------------------

def test_joint_biased_mmax_approx_clp():
    """
    Joint listener, P(m_max) = .8, P(m_sit) = P(m_min) = .1, flat P(k).
    m_max always prefers v_prc when k=k_p, so v_apx strongly implies k_np.
    P(k_p | v_apx, c_LP) = 1/6 ≈ .167.
    """
    sc = build_scenario(
        inference="joint",
        priors_M={"m_max": 0.8, "m_sit": 0.1, "m_min": 0.1},
    )
    assert sc.prob_k("k_p", "v_apx", "c_LP") == pytest.approx(1 / 6)


# ---------------------------------------------------------------------------
# Staged balanced model — regression against pre-refactor values
# ---------------------------------------------------------------------------

class TestStagedBalancedModelUnchanged:
    """
    All 12 balanced-model cells must be numerically identical to the values
    computed before the inference= field was added (verified against the
    committed test suite in test_case_studies.py and the paper's Table X).
    """

    def setup_method(self):
        self.sc = build_scenario(inference="staged", omega=0.5)
        self.scores = self.sc.evaluate_all()

    def test_acomp_vprc(self):
        s = self.scores["a_comp"]
        assert s["v_prc"]["c_HP"] == pytest.approx(0.750)
        assert s["v_prc"]["c_LP"] == pytest.approx(0.500)

    def test_acomp_vapx(self):
        s = self.scores["a_comp"]
        assert s["v_apx"]["c_HP"] == pytest.approx(-0.250)
        assert s["v_apx"]["c_LP"] == pytest.approx(0.000, abs=1e-12)

    def test_alike_vprc(self):
        s = self.scores["a_like"]
        assert s["v_prc"]["c_HP"] == pytest.approx(0.250)
        assert s["v_prc"]["c_LP"] == pytest.approx(0.000, abs=1e-12)

    def test_alike_vapx(self):
        s = self.scores["a_like"]
        assert s["v_apx"]["c_HP"] == pytest.approx(1 / 12)   # 0.0833…
        assert s["v_apx"]["c_LP"] == pytest.approx(5 / 24)   # 0.2083…

    def test_aped_vprc(self):
        s = self.scores["a_ped"]
        assert s["v_prc"]["c_HP"] == pytest.approx(0.250)
        assert s["v_prc"]["c_LP"] == pytest.approx(0.500)

    def test_aped_vapx(self):
        s = self.scores["a_ped"]
        assert s["v_apx"]["c_HP"] == pytest.approx(-1 / 6)
        assert s["v_apx"]["c_LP"] == pytest.approx(-1 / 6)

    def test_all_four_effects(self):
        assert all(target_effects(self.scores))

    def test_staged_default_same_as_explicit(self):
        """build_scenario() (no inference=) and inference='staged' give identical scores."""
        default_scores = build_scenario(omega=0.5).evaluate_all()
        assert default_scores == self.scores


# ---------------------------------------------------------------------------
# Sampled RMSE / CDS — E-session reference values (seed 42)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(_MEANS), reason="empirical_means_cs1.json not present")
def test_staged_sampled_rmse_cds():
    """Staged sampled RMSE 0.30 ± 0.10, CDS 0.94 ± 0.43 (E-session reference)."""
    results = evaluate_variant("full", _MEANS, n_samples=100, seed=42,
                               inference="staged")
    s = summarize_results(results)
    assert s["rmse"]["mean"] == pytest.approx(0.30, abs=0.01)
    assert s["rmse"]["sd"]   == pytest.approx(0.10, abs=0.01)
    assert s["cds"]["mean"]  == pytest.approx(0.94, abs=0.01)
    assert s["cds"]["sd"]    == pytest.approx(0.43, abs=0.01)


@pytest.mark.skipif(not os.path.exists(_MEANS), reason="empirical_means_cs1.json not present")
def test_joint_sampled_all_four_effects():
    """Joint listener: 99/100 sampled combinations reproduce all four effects."""
    from sem.comparison import sample_parameters
    from sem import fit

    params = sample_parameters("full", n_samples=100, seed=42)
    count = 0
    for p in params:
        sc = build_scenario(omega=p["omega"], priors_K=p["priors_K"],
                            priors_M=p["priors_M"], inference="joint")
        if all(target_effects(sc.evaluate_all())):
            count += 1
    assert count == 99


# ---------------------------------------------------------------------------
# Slow: .02 grid — 2,593,080 settings per listener
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(_MEANS), reason="empirical_means_cs1.json not present")
def test_grid_02_staged():
    """Staged listener: 99.87 % of .02-step grid combinations predict all four effects."""
    sc = build_scenario(inference="staged")
    rt = sc.robustness_test(target_effects, step=0.02)
    assert rt["total"] == 2_593_080
    assert rt["rate_all"] == pytest.approx(0.9987, abs=0.0001)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(_MEANS), reason="empirical_means_cs1.json not present")
def test_grid_02_joint():
    """Joint listener: 98.86 % of .02-step grid combinations predict all four effects."""
    sc = build_scenario(inference="joint")
    rt = sc.robustness_test(target_effects, step=0.02)
    assert rt["total"] == 2_593_080
    assert rt["rate_all"] == pytest.approx(0.9886, abs=0.0001)
