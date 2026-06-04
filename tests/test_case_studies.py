"""
tests/test_case_studies.py
==========================
Tests that verify the SEM correctly reproduces the empirical effects
reported in the paper for both case studies.

Run with:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from case_studies.cs1_imprecision      import build_scenario as build_cs1, target_effects as effects_cs1
from case_studies.cs2_pragmatic_violations import build_scenario as build_cs2, target_effects as effects_cs2

# Paths used by TestImpactDerivation
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DATA_CSV  = os.path.join(_REPO_ROOT, "data", "cs1_experiment", "processed", "experiment1.csv")
_JSON_PATH = os.path.join(_REPO_ROOT, "data", "cs1_experiment", "processed", "impact_tables_cs1.json")


# ===========================================================================
# Case Study 1 — (Im)Precision
# ===========================================================================

class TestCS1BalancedModel:
    """
    Verify the balanced model (ω=0.5, uniform priors) correctly predicts
    all four empirical effects from the imprecision experiment.
    """

    def setup_method(self):
        self.scenario = build_cs1(omega=0.5)
        self.scores   = self.scenario.evaluate_all()

    def test_all_four_effects(self):
        effects = effects_cs1(self.scores)
        assert all(effects), (
            f"Balanced model failed effects: "
            f"{[i+1 for i, e in enumerate(effects) if not e]}"
        )

    def test_effect1_precise_more_competent(self):
        """PRECISE rated higher than APPROX on competence (across contexts)."""
        s = self.scores
        assert (s["a_comp"]["v_prc"]["c_HP"] + s["a_comp"]["v_prc"]["c_LP"] >
                s["a_comp"]["v_apx"]["c_HP"] + s["a_comp"]["v_apx"]["c_LP"])

    def test_effect2_precise_more_pedantic(self):
        """PRECISE rated higher than APPROX on pedantry (across contexts)."""
        s = self.scores
        assert (s["a_ped"]["v_prc"]["c_HP"] + s["a_ped"]["v_prc"]["c_LP"] >
                s["a_ped"]["v_apx"]["c_HP"] + s["a_ped"]["v_apx"]["c_LP"])

    def test_effect3_competence_interaction(self):
        """Competence gap (prc - apx) is larger in c_HP than c_LP."""
        s = self.scores
        gap_HP = s["a_comp"]["v_prc"]["c_HP"] - s["a_comp"]["v_apx"]["c_HP"]
        gap_LP = s["a_comp"]["v_prc"]["c_LP"] - s["a_comp"]["v_apx"]["c_LP"]
        assert gap_HP > gap_LP

    def test_effect4_likeability_interaction(self):
        """Likeability gap (prc - apx) is larger in c_HP than c_LP."""
        s = self.scores
        gap_HP = s["a_like"]["v_prc"]["c_HP"] - s["a_like"]["v_apx"]["c_HP"]
        gap_LP = s["a_like"]["v_prc"]["c_LP"] - s["a_like"]["v_apx"]["c_LP"]
        assert gap_HP > gap_LP

    def test_scores_in_valid_range(self):
        """All evaluation scores must be in [-1, 1]."""
        for a, by_v in self.scores.items():
            for v, by_c in by_v.items():
                for c, score in by_c.items():
                    assert -1.0 <= score <= 1.0, (
                        f"Score out of range: {a},{v},{c} = {score}"
                    )

    def test_knowledge_inference_precise_utterance(self):
        """
        After hearing v_prc, the listener should infer k_p with probability 1
        (only a speaker with precise knowledge can truthfully say v_prc).
        """
        assert self.scenario.prob_k("k_p",  "v_prc") == pytest.approx(1.0)
        assert self.scenario.prob_k("k_np", "v_prc") == pytest.approx(0.0)

    def test_knowledge_inference_approx_utterance(self):
        """
        After hearing v_apx, both knowledge states remain possible
        (under uniform priors, they should be equal).
        """
        p_kp  = self.scenario.prob_k("k_p",  "v_apx")
        p_knp = self.scenario.prob_k("k_np", "v_apx")
        assert p_kp  == pytest.approx(0.5)
        assert p_knp == pytest.approx(0.5)

    def test_strategy_inference_precise_low_precision_context(self):
        """
        In c_LP, a precise utterance uniquely implies m_max
        (m_sit would use v_apx in c_LP, m_min always uses v_apx).
        Paper: Figure 4(c), left panel.
        """
        p_max = self.scenario.prob_m("m_max", "k_p", "c_LP", "v_prc")
        p_sit = self.scenario.prob_m("m_sit", "k_p", "c_LP", "v_prc")
        p_min = self.scenario.prob_m("m_min", "k_p", "c_LP", "v_prc")
        assert p_max == pytest.approx(1.0)
        assert p_sit == pytest.approx(0.0)
        assert p_min == pytest.approx(0.0)

    def test_strategy_inference_precise_high_precision_context(self):
        """
        In c_HP, a precise utterance is consistent with both m_max and m_sit,
        so under uniform priors both get probability 0.5.
        Paper: Figure 4(c), right panel.
        """
        p_max = self.scenario.prob_m("m_max", "k_p", "c_HP", "v_prc")
        p_sit = self.scenario.prob_m("m_sit", "k_p", "c_HP", "v_prc")
        assert p_max == pytest.approx(0.5)
        assert p_sit == pytest.approx(0.5)


# ===========================================================================
# Case Study 2 — Pragmatic Violations
# ===========================================================================

class TestCS2BalancedModel:
    """
    Verify the balanced model (ω=0.5, uniform priors) correctly predicts
    all six empirical effects from Beltrama & Papafragou (2023).
    """

    def setup_method(self):
        self.scenario = build_cs2(omega=0.5)
        self.scores   = self.scenario.evaluate_all()
        self.c        = "c_topic"

    def _sc(self, a, v):
        return self.scores[a][v][self.c]

    def test_all_six_effects(self):
        effects = effects_cs2(self.scores)
        assert all(effects), (
            f"Balanced model failed effects: "
            f"{[i+1 for i, e in enumerate(effects) if not e]}"
        )

    def test_effect1_relevance_competence(self):
        """Relevant utterances rated higher on competence."""
        assert (self._sc("a_comp","v_A") + self._sc("a_comp","v_a") >
                self._sc("a_comp","v_B") + self._sc("a_comp","v_b"))

    def test_effect2_relevance_likeability(self):
        """Relevant utterances rated higher on likeability."""
        assert (self._sc("a_like","v_A") + self._sc("a_like","v_a") >
                self._sc("a_like","v_B") + self._sc("a_like","v_b"))

    def test_effect3_informativeness_competence(self):
        """High-info utterances rated higher on competence."""
        assert (self._sc("a_comp","v_A") + self._sc("a_comp","v_B") >
                self._sc("a_comp","v_a") + self._sc("a_comp","v_b"))

    def test_effect4_informativeness_likeability(self):
        """High-info utterances rated higher on likeability."""
        assert (self._sc("a_like","v_A") + self._sc("a_like","v_B") >
                self._sc("a_like","v_a") + self._sc("a_like","v_b"))

    def test_effect5_interaction_competence(self):
        """Informativeness gap on competence larger when relevant."""
        assert (self._sc("a_comp","v_A") - self._sc("a_comp","v_a") >
                self._sc("a_comp","v_B") - self._sc("a_comp","v_b"))

    def test_effect6_interaction_likeability(self):
        """Informativeness gap on likeability larger when relevant."""
        assert (self._sc("a_like","v_A") - self._sc("a_like","v_a") >
                self._sc("a_like","v_B") - self._sc("a_like","v_b"))

    def test_scores_in_valid_range(self):
        for a, by_v in self.scores.items():
            for v, by_c in by_v.items():
                for c, score in by_c.items():
                    assert -1.0 <= score <= 1.0

    def test_relevant_utterances_always_more_competent_than_irrelevant(self):
        """
        Every relevant utterance (v_A, v_a) should score higher on competence
        than every irrelevant utterance (v_B, v_b).

        NOTE: Under the balanced model with uniform priors, v_B and v_b score
        identically on competence (-0.5 each).  This is because the only
        differentiator between them is knowledge state inference, but the
        φ_K values for k_aB and k_ab are both -1 for a_comp — so informativeness
        within the irrelevant set has no competence effect at the balanced model.
        This is an interesting model property worth discussing in the paper
        (it contrasts with effect 3, which holds only in aggregate).
        """
        assert self._sc("a_comp", "v_A") > self._sc("a_comp", "v_B")
        assert self._sc("a_comp", "v_A") > self._sc("a_comp", "v_b")
        assert self._sc("a_comp", "v_a") > self._sc("a_comp", "v_B")
        assert self._sc("a_comp", "v_a") > self._sc("a_comp", "v_b")


# ===========================================================================
# Task 0 — Impact derivation
# ===========================================================================

class TestImpactDerivation:
    """Verify that derive_trivalent_matrix reproduces the expected
    p<.01 impact matrices from the CS1 data.

    Balanced-model prediction scores after p<.01 matrix update (ω=0.5):
        a_comp: v_prc/c_HP=+0.750, v_prc/c_LP=+0.500,
                v_apx/c_HP=-0.250, v_apx/c_LP= 0.000
        a_like: v_prc/c_HP=+0.250, v_prc/c_LP= 0.000,
                v_apx/c_HP=+0.083, v_apx/c_LP=+0.208
        a_ped:  v_prc/c_HP=+0.250, v_prc/c_LP=+0.500,
                v_apx/c_HP=-0.167, v_apx/c_LP=-0.167
    All four target effects hold: ✓✓✓✓
    """

    def test_p001_derivation_runs(self):
        """derive_trivalent_matrix runs without error at p=0.001."""
        if not os.path.exists(_DATA_CSV):
            pytest.skip("Data CSV not present (CI environment)")
        from sem.impact_derivation import (
            derive_trivalent_matrix,
            CS1_SEM_MAPPING,
            CS1_SEM_ATTR_MAPPING,
        )
        result = derive_trivalent_matrix(
            _DATA_CSV, CS1_SEM_MAPPING, CS1_SEM_ATTR_MAPPING, alpha=0.001
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"k_p", "k_np", "m_max", "m_sit", "m_min"}
        for sem_var in result:
            assert set(result[sem_var].keys()) == {"a_comp", "a_like", "a_ped"}
            for val in result[sem_var].values():
                assert val in (-1, 0, 1)

    def test_p01_matches_cs1_impact_matrices(self):
        """Derivation at p<.01 matches IMPACT_K and IMPACT_M in cs1_imprecision."""
        if not os.path.exists(_DATA_CSV):
            pytest.skip("Data CSV not present (CI environment)")
        from sem.impact_derivation import (
            derive_trivalent_matrix,
            CS1_SEM_MAPPING,
            CS1_SEM_ATTR_MAPPING,
        )
        from case_studies.cs1_imprecision import IMPACT_K, IMPACT_M

        derived = derive_trivalent_matrix(
            _DATA_CSV, CS1_SEM_MAPPING, CS1_SEM_ATTR_MAPPING, alpha=0.01
        )
        expected = {**IMPACT_K, **IMPACT_M}
        for sem_var, attrs in derived.items():
            for sem_attr, val in attrs.items():
                assert val == expected[sem_var][sem_attr], (
                    f"Mismatch at {sem_var}[{sem_attr}]: "
                    f"derived={val}, model={expected[sem_var][sem_attr]}"
                )

    def test_all_thresholds_saved(self):
        """impact_tables_cs1.json exists and contains all four threshold keys."""
        if not os.path.exists(_JSON_PATH):
            pytest.skip("impact_tables_cs1.json not present (CI environment)")
        import json
        with open(_JSON_PATH) as fh:
            tables = json.load(fh)
        assert set(tables.keys()) == {"0.001", "0.01", "0.05", "0.1"}
        for alpha_key, matrix in tables.items():
            assert set(matrix.keys()) == {"k_p", "k_np", "m_max", "m_sit", "m_min"}
            for sem_var, attrs in matrix.items():
                assert set(attrs.keys()) == {"a_comp", "a_like", "a_ped"}


# ===========================================================================
# Task 1 — Quantitative fit metrics (sem/fit.py)
# ===========================================================================

_MEANS_JSON = os.path.join(
    _REPO_ROOT, "data", "cs1_experiment", "processed", "empirical_means_cs1.json"
)


def _effects(main_signs, inter_signs):
    """Build an effects dict with given main/interaction magnitudes."""
    attrs = ["a_comp", "a_like", "a_ped"]
    return {
        a: {"main_effect": m, "interaction": i, "delta_HP": 0.0, "delta_LP": 0.0}
        for a, m, i in zip(attrs, main_signs, inter_signs)
    }


class TestFitMetrics:
    """Verify sem/fit.py metric functions on known inputs."""

    def test_normalize_likert_midpoint(self):
        from sem.fit import normalize_likert
        assert normalize_likert(4.0) == 0.0

    def test_normalize_likert_endpoints(self):
        from sem.fit import normalize_likert
        assert normalize_likert(1.0) == -1.0
        assert normalize_likert(7.0) == 1.0

    def test_das_perfect(self):
        from sem.fit import das
        he = _effects([0.5, 0.2, 0.4], [0.3, -0.1, 0.2])
        me = _effects([0.8, 0.1, 0.6], [0.9, -0.5, 0.7])
        assert das(he, me) == 1.0

    def test_das_all_wrong(self):
        from sem.fit import das
        he = _effects([0.5, 0.2, 0.4], [0.3, -0.1, 0.2])
        me = _effects([-0.5, -0.2, -0.4], [0.3, -0.1, 0.2])
        assert das(he, me) == 0.0

    def test_iss_partial(self):
        from sem.fit import iss
        he = _effects([0.5, 0.2, 0.4], [0.3, 0.1, 0.2])
        me = _effects([0.5, 0.2, 0.4], [0.3, 0.1, -0.2])   # 2 of 3 match
        assert iss(he, me) == pytest.approx(2 / 3)

    def test_esr_perfect_calibration(self):
        from sem.fit import esr
        he = _effects([0.5, 0.2, 0.4], [0.3, 0.1, 0.2])
        me = _effects([0.5, 0.2, 0.4], [0.3, 0.1, 0.2])
        res = esr(he, me, sig_main=["a_comp", "a_ped"],
                  sig_interaction=["a_comp", "a_like"])
        assert all(v == pytest.approx(1.0) for v in res["main"].values())
        assert all(v == pytest.approx(1.0) for v in res["interaction"].values())

    def test_esr_double_magnitude(self):
        from sem.fit import esr
        he = _effects([0.5, 0.2, 0.4], [0.3, 0.1, 0.2])
        me = _effects([1.0, 0.4, 0.8], [0.6, 0.2, 0.4])
        res = esr(he, me, sig_main=["a_comp", "a_ped"],
                  sig_interaction=["a_comp", "a_like"])
        assert res["main"]["a_comp"] == pytest.approx(2.0)
        assert res["interaction"]["a_like"] == pytest.approx(2.0)

    def test_cds_zero(self):
        from sem.fit import cds
        esr_vals = {"main": {"a_comp": 1.0, "a_ped": 1.0},
                    "interaction": {"a_comp": 1.0, "a_like": 1.0}}
        out = cds(esr_vals)
        assert out["cds"] == 0.0

    def test_global_metrics_identical(self):
        from sem.fit import global_metrics
        means = {
            "a_comp": {"v_prc": {"c_HP": 0.7, "c_LP": 0.4},
                       "v_apx": {"c_HP": -0.2, "c_LP": 0.0}},
            "a_like": {"v_prc": {"c_HP": 0.2, "c_LP": -0.1},
                       "v_apx": {"c_HP": 0.1, "c_LP": 0.3}},
            "a_ped":  {"v_prc": {"c_HP": 0.3, "c_LP": 0.5},
                       "v_apx": {"c_HP": -0.1, "c_LP": -0.2}},
        }
        g = global_metrics(means, means)
        assert g["spearman_r"] == pytest.approx(1.0)
        assert g["rmse"] == pytest.approx(0.0)
        assert g["ccc"] == pytest.approx(1.0)

    def test_evaluate_runs_on_cs1(self):
        if not os.path.exists(_MEANS_JSON):
            pytest.skip("empirical_means_cs1.json not present")
        from sem import fit
        scores = build_cs1(omega=0.5).evaluate_all()
        res = fit.evaluate(scores, _MEANS_JSON)
        assert 0.0 <= res["das"] <= 1.0
        assert 0.0 <= res["iss"] <= 1.0
        assert res["cds"] >= 0.0
        assert res["rmse"] >= 0.0
        assert -1.0 <= res["ccc"] <= 1.0
        assert -1.0 <= res["spearman_r"] <= 1.0
        assert set(res["esr_main"]) == {"a_comp", "a_ped"}
        assert set(res["esr_interaction"]) == {"a_comp", "a_like"}


# ===========================================================================
# Task 2 — SEM variant comparison (sem/comparison.py)
# ===========================================================================

_FAKE_RESULT = {
    "das": 1.0, "iss": 1.0, "spearman_r": 0.5, "rmse": 0.2, "ccc": 0.6,
    "cds_main": 1.0, "cds_interaction": 0.3, "cds": 0.65,
    "esr_main": {"a_comp": 2.0, "a_ped": 1.5},
    "esr_interaction": {"a_comp": 1.2, "a_like": 1.1},
}


class TestVariantComparison:
    """Verify sem/comparison.py sampling and summarization logic."""

    def test_sample_parameters_count(self):
        from sem.comparison import sample_parameters
        assert len(sample_parameters("full", n_samples=10)) == 10

    def test_sample_parameters_omega_range(self):
        from sem.comparison import sample_parameters
        for s in sample_parameters("full", n_samples=20):
            assert 0.0 < s["omega"] < 1.0

    def test_sample_parameters_priors_sum_to_one(self):
        from sem.comparison import sample_parameters
        for s in sample_parameters("full", n_samples=20):
            assert abs(sum(s["priors_K"].values()) - 1.0) < 1e-9
            assert abs(sum(s["priors_M"].values()) - 1.0) < 1e-9

    def test_context_blind_msit_zero(self):
        from sem.comparison import sample_parameters
        for s in sample_parameters("context_blind", n_samples=20):
            assert s["priors_M"]["m_sit"] == 0.0

    def test_knowledge_only_omega_one(self):
        from sem.comparison import sample_parameters
        for s in sample_parameters("knowledge_only", n_samples=20):
            assert s["omega"] == 1.0

    def test_motivation_only_omega_zero(self):
        from sem.comparison import sample_parameters
        for s in sample_parameters("motivation_only", n_samples=20):
            assert s["omega"] == 0.0

    def test_summarize_results_keys(self):
        from sem.comparison import summarize_results
        out = summarize_results([dict(_FAKE_RESULT) for _ in range(5)])
        for key in ["das", "iss", "spearman_r", "rmse", "ccc",
                    "cds_main", "cds_interaction", "cds",
                    "esr_main", "esr_interaction"]:
            assert key in out

    def test_summarize_results_sd_nonnegative(self):
        from sem.comparison import summarize_results
        r2 = dict(_FAKE_RESULT)
        r2["das"], r2["iss"], r2["cds"] = 0.0, 0.5, 0.9
        out = summarize_results([dict(_FAKE_RESULT), r2])
        for key in ["das", "iss", "spearman_r", "rmse", "ccc",
                    "cds_main", "cds_interaction", "cds"]:
            assert out[key]["sd"] >= 0.0

    def test_compare_all_variants_runs(self):
        if not os.path.exists(_MEANS_JSON):
            pytest.skip("empirical_means_cs1.json not present")
        from sem.comparison import compare_all_variants
        comp = compare_all_variants(_MEANS_JSON, n_samples=5, seed=42)
        assert set(comp.keys()) == {
            "full", "context_blind", "knowledge_only", "motivation_only"
        }
