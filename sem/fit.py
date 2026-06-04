"""
sem/fit.py
==========
Quantitative fit-evaluation engine for the Social Evaluation Model (SEM).

This module compares SEM predictions (already in [-1, 1]) to empirical human
means (raw 1-7 Likert, loaded from JSON and normalized at call time) using a
metric suite from Mühlenbernd (2026, CMCL):

  - DAS  (Directional Agreement Score)      — sign match on main effects
  - ISS  (Interaction Sensitivity Score)    — sign match on form x context
  - ESR  (Effect Size Ratio)                — |model| / |human| per sig. effect
  - CDS  (Calibration Deviation Score)      — mean |ESR - 1|
  - Spearman rho, RMSE, CCC                 — global calibration over all cells

Directional/magnitude metrics (DAS, ISS, ESR, CDS) are computed over the
statistically significant effects only; Spearman, RMSE, and CCC are global,
computed cell-by-cell over all 12 condition means.

The module is self-contained and importable independently of the rest of the
SEM package (it only needs ``scipy`` and the JSON produced by
``scripts/compute_empirical_means.py``).
"""

import json
import math
from typing import Dict, List, Optional

from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Normalization & loading
# ---------------------------------------------------------------------------

def normalize_likert(value: float, scale_min: int = 1, scale_max: int = 7) -> float:
    """
    Map a raw Likert mean onto [-1, 1].

        normalized = 2 * (value - scale_min) / (scale_max - scale_min) - 1

    For a 1-7 scale this is ``(value - 4) / 3`` — the same range as the SEM's
    evaluation output, making model and human values directly comparable.
    """
    return 2.0 * (value - scale_min) / (scale_max - scale_min) - 1.0


def load_empirical_means(
    json_path: str,
    attributes: Optional[List[str]] = None,
) -> Dict:
    """
    Load the JSON produced by ``scripts/compute_empirical_means.py`` and return
    the same nested structure with every mean normalized to [-1, 1].

    The ``_meta`` block is preserved unchanged under the same key.  If
    ``attributes`` is given, only those attribute keys are kept.
    """
    with open(json_path) as fh:
        raw = json.load(fh)

    out: Dict = {}
    for attr, by_utt in raw.items():
        if attr.startswith("_"):
            out[attr] = by_utt  # preserve _meta verbatim
            continue
        if attributes is not None and attr not in attributes:
            continue
        out[attr] = {
            utt: {ctx: normalize_likert(val) for ctx, val in by_ctx.items()}
            for utt, by_ctx in by_utt.items()
        }
    return out


# ---------------------------------------------------------------------------
# Effect extraction
# ---------------------------------------------------------------------------

def _attr_keys(d: Dict, attributes: Optional[List[str]]) -> List[str]:
    """Attribute keys to iterate: the given list, or all non-``_meta`` keys."""
    if attributes is not None:
        return list(attributes)
    return [k for k in d if not k.startswith("_")]


def extract_effects(
    means: Dict,
    attributes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    From a means dict ``{attr: {utterance: {context: score}}}`` compute, per
    attribute, the derived quantities used by every metric:

        delta_HP     = v_prc@c_HP - v_apx@c_HP
        delta_LP     = v_prc@c_LP - v_apx@c_LP
        main_effect  = (delta_HP + delta_LP) / 2          (main effect of form)
        interaction  = delta_HP - delta_LP                (form x context)

    Works identically on human means and model scores (both in [-1, 1]).
    """
    result: Dict[str, Dict[str, float]] = {}
    for a in _attr_keys(means, attributes):
        hp = means[a]["v_prc"]["c_HP"] - means[a]["v_apx"]["c_HP"]
        lp = means[a]["v_prc"]["c_LP"] - means[a]["v_apx"]["c_LP"]
        result[a] = {
            "delta_HP": hp,
            "delta_LP": lp,
            "main_effect": (hp + lp) / 2.0,
            "interaction": hp - lp,
        }
    return result


# ---------------------------------------------------------------------------
# Structural agreement: DAS / ISS
# ---------------------------------------------------------------------------

def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def _agreement(human_effects, model_effects, key, attributes) -> float:
    """Fraction of attributes whose model/human ``key`` effects share a sign.

    Attributes with a zero human effect (undefined sign) are skipped from both
    numerator and denominator.
    """
    attrs = _attr_keys(human_effects, attributes)
    matches, considered = 0, 0
    for a in attrs:
        h = human_effects[a][key]
        if h == 0:
            continue
        considered += 1
        if _sign(model_effects[a][key]) == _sign(h):
            matches += 1
    return matches / considered if considered else 0.0


def das(human_effects, model_effects, attributes=None) -> float:
    """Directional Agreement Score — sign agreement on the main effect of form.

    ``attributes`` should list the attributes with a *statistically significant*
    main effect (for CS1: ``a_comp``, ``a_ped``); the score is the proportion of
    those on which model and human agree in sign. Passing ``None`` falls back to
    every attribute, which is only appropriate when all main effects are
    significant.
    """
    return _agreement(human_effects, model_effects, "main_effect", attributes)


def iss(human_effects, model_effects, attributes=None) -> float:
    """Interaction Sensitivity Score — sign agreement on the form x context term.

    ``attributes`` should list the attributes with a *statistically significant*
    form x context interaction (for CS1: ``a_comp``, ``a_like``); the score is
    the proportion of those on which model and human agree in sign. Passing
    ``None`` falls back to every attribute.
    """
    return _agreement(human_effects, model_effects, "interaction", attributes)


# ---------------------------------------------------------------------------
# Magnitude calibration: ESR / CDS
# ---------------------------------------------------------------------------

def esr(
    human_effects: Dict,
    model_effects: Dict,
    sig_main: List[str],
    sig_interaction: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Effect Size Ratio ``|model| / |human|`` per *significant* effect.

    ``sig_main`` / ``sig_interaction`` list the attributes whose human effect is
    statistically significant (and therefore has non-negligible magnitude).

    Raises ``ValueError`` if a listed significant effect has |human| == 0 (that
    indicates a data error, not a near-zero non-significant effect).
    """
    def ratio(a, key):
        h = abs(human_effects[a][key])
        if h == 0:
            raise ValueError(
                f"|human {key}| == 0 for significant attribute '{a}'"
            )
        return abs(model_effects[a][key]) / h

    return {
        "main": {a: ratio(a, "main_effect") for a in sig_main},
        "interaction": {a: ratio(a, "interaction") for a in sig_interaction},
    }


def cds(esr_values: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calibration Deviation Score — mean absolute deviation of ESR from 1.

        CDS_main        = mean(|ESR_main[a] - 1|)
        CDS_interaction = mean(|ESR_interaction[a] - 1|)
        CDS             = mean(CDS_main, CDS_interaction)

    Lower is better; 0 = perfect magnitude calibration.
    """
    def mad(d):
        vals = [abs(v - 1.0) for v in d.values()]
        return sum(vals) / len(vals) if vals else 0.0

    cds_main = mad(esr_values["main"])
    cds_int = mad(esr_values["interaction"])
    return {
        "cds_main": cds_main,
        "cds_interaction": cds_int,
        "cds": (cds_main + cds_int) / 2.0,
    }


# ---------------------------------------------------------------------------
# Global metrics: Spearman / RMSE / CCC
# ---------------------------------------------------------------------------

def _flatten(means: Dict, attributes: List[str]) -> List[float]:
    """Flatten to a deterministic vector over attr x utterance x context."""
    vec = []
    for a in attributes:
        for utt in sorted(means[a]):
            for ctx in sorted(means[a][utt]):
                vec.append(means[a][utt][ctx])
    return vec


def global_metrics(
    human_means: Dict,
    model_scores: Dict,
    attributes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Spearman rho, RMSE, and CCC across the flat vector of all (attr, utterance,
    context) cells (12 cells for 3 attrs x 2 utterances x 2 contexts).
    """
    attrs = _attr_keys(human_means, attributes)
    H = _flatten(human_means, attrs)
    M = _flatten(model_scores, attrs)
    n = len(H)

    rho, _ = spearmanr(H, M)
    rmse = math.sqrt(sum((h - m) ** 2 for h, m in zip(H, M)) / n)

    mean_h = sum(H) / n
    mean_m = sum(M) / n
    var_h = sum((h - mean_h) ** 2 for h in H) / n
    var_m = sum((m - mean_m) ** 2 for m in M) / n
    cov = sum((h - mean_h) * (m - mean_m) for h, m in zip(H, M)) / n
    denom = var_h + var_m + (mean_h - mean_m) ** 2
    ccc = (2 * cov / denom) if denom else 1.0

    return {"spearman_r": float(rho), "rmse": rmse, "ccc": ccc}


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def evaluate(
    model_scores: Dict,
    empirical_json_path: str,
    sig_main: Optional[List[str]] = None,
    sig_interaction: Optional[List[str]] = None,
    attributes: Optional[List[str]] = None,
) -> Dict:
    """
    Compute the full metric suite comparing ``model_scores`` (SEM output, already
    in [-1, 1]) to the empirical means stored at ``empirical_json_path``.

    Defaults are the CS1 case: 3 attributes, with significant main effects on
    {a_comp, a_ped} and significant interactions on {a_comp, a_like}.
    """
    if attributes is None:
        attributes = ["a_comp", "a_like", "a_ped"]
    if sig_main is None:
        sig_main = ["a_comp", "a_ped"]
    if sig_interaction is None:
        sig_interaction = ["a_comp", "a_like"]

    human_means = load_empirical_means(empirical_json_path, attributes=attributes)

    human_effects = extract_effects(human_means, attributes)
    model_effects = extract_effects(model_scores, attributes)

    esr_values = esr(human_effects, model_effects, sig_main, sig_interaction)
    cds_values = cds(esr_values)
    glob = global_metrics(human_means, model_scores, attributes)

    return {
        "das": das(human_effects, model_effects, sig_main),
        "iss": iss(human_effects, model_effects, sig_interaction),
        "esr_main": esr_values["main"],
        "esr_interaction": esr_values["interaction"],
        "cds_main": cds_values["cds_main"],
        "cds_interaction": cds_values["cds_interaction"],
        "cds": cds_values["cds"],
        "spearman_r": glob["spearman_r"],
        "rmse": glob["rmse"],
        "ccc": glob["ccc"],
    }
