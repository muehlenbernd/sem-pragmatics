"""
sem/plotting.py
===============
Reusable plotting functions for SEM evaluation results.

All functions return matplotlib Figure objects so they can be
saved, shown, or embedded in notebooks.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared style defaults
# ---------------------------------------------------------------------------

LATEX_STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif":  ["Computer Modern Roman"],
}

# Fall back gracefully if LaTeX is not available
def _apply_style(use_latex: bool = True):
    if use_latex:
        try:
            plt.rcParams.update(LATEX_STYLE)
        except Exception:
            pass  # silently fall back to default fonts


# ---------------------------------------------------------------------------
# Case Study 1 — 3-panel bar chart (competent / likeable / pedantic)
# ---------------------------------------------------------------------------

def plot_cs1(
    scores: Dict,
    attributes: List[str]    = ("a_comp", "a_like", "a_ped"),
    utterances: List[str]    = ("v_apx", "v_prc"),
    contexts:   List[str]    = ("c_HP", "c_LP"),
    attr_labels: List[str]   = (r"$a_{comp}$", r"$a_{like}$", r"$a_{ped}$"),
    utt_labels:  List[str]   = (r"$v_{apx}$",  r"$v_{prc}$"),
    ctx_labels:  List[str]   = (r"$c_{HP}$",   r"$c_{LP}$"),
    colors:     Tuple[str, str] = ("darkblue", "lightcoral"),
    use_latex:  bool = True,
    title:      Optional[str] = None,
    figsize:    Tuple[float, float] = (7.5, 3.0),
) -> plt.Figure:
    """
    Three-panel bar chart for Case Study 1 (imprecision).

    For each attribute, shows evaluation scores for both utterance forms
    (v_apx in dark blue, v_prc in light coral) side-by-side within each
    context group (c_HP and c_LP).

    Parameters
    ----------
    scores : dict
        Output of scenario.evaluate_all() — scores[a][v][c].
    """
    _apply_style(use_latex)

    fig, axes = plt.subplots(1, len(attributes), figsize=figsize)
    plt.subplots_adjust(wspace=0.35)

    for i, (ax, attr, attr_lbl) in enumerate(zip(axes, attributes, attr_labels)):
        # Bar positions: two groups (contexts), two bars each (utterances)
        group_positions = [0.0, 1.5]   # centre of each context group
        bar_width = 0.5
        offsets   = [-bar_width / 2, bar_width / 2]  # apx left, prc right

        for j, (utt, color, utt_lbl) in enumerate(zip(utterances, colors, utt_labels)):
            bar_x  = [gp + offsets[j] for gp in group_positions]
            values = [scores[attr][utt][c] for c in contexts]
            ax.bar(bar_x, values, width=bar_width, color=color, label=utt_lbl)

        ax.set_title(attr_lbl, fontsize=14)
        ax.set_xticks(group_positions)
        ax.set_xticklabels(ctx_labels, fontsize=13)
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if i == 0:
            ax.set_ylabel(r"$E_L$ score", fontsize=13)
        if i == len(attributes) - 1:
            ax.legend(fontsize=11, loc="upper right", frameon=True)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Case Study 2 — 2-panel bar chart (competent / likeable)
# ---------------------------------------------------------------------------

def plot_cs2(
    scores: Dict,
    attributes: List[str]    = ("a_comp", "a_like"),
    utterances: List[str]    = ("v_A", "v_B", "v_a", "v_b"),
    context:    str          = "c_topic",
    attr_labels: List[str]   = (r"$f_{comp}$", r"$f_{like}$"),
    utt_labels:  List[str]   = (r"$v_A$", r"$v_B$", r"$v_a$", r"$v_b$"),
    colors:     Tuple[str, ...] = ("lightcoral", "darkblue", "lightcoral", "darkblue"),
    use_latex:  bool = True,
    title:      Optional[str] = None,
    figsize:    Tuple[float, float] = (6.0, 3.0),
) -> plt.Figure:
    """
    Two-panel bar chart for Case Study 2 (pragmatic violations).

    Shows evaluation scores for the four utterance types, colour-coded
    by topic (topic A = light coral, topic B = dark blue).

    Parameters
    ----------
    scores : dict
        Output of scenario.evaluate_all() — scores[a][v][c].
    """
    _apply_style(use_latex)

    fig, axes = plt.subplots(1, len(attributes), figsize=figsize)
    plt.subplots_adjust(wspace=0.35)

    positions = [0.0, 0.55, 1.35, 1.9]  # matches original notebook spacing
    bar_width = 0.5

    for i, (ax, attr, attr_lbl) in enumerate(zip(axes, attributes, attr_labels)):
        values = [scores[attr][v][context] for v in utterances]
        ax.bar(positions, values, width=bar_width, color=colors)

        ax.set_title(attr_lbl, fontsize=14)
        ax.set_xticks(positions)
        ax.set_xticklabels(utt_labels, fontsize=13)
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if i == 0:
            ax.set_ylabel(r"$E_L$ score", fontsize=13)
        if i == len(attributes) - 1:
            legend_elements = [
                Patch(facecolor="lightcoral", label="topic A"),
                Patch(facecolor="darkblue",   label="topic B"),
            ]
            ax.legend(handles=legend_elements, fontsize=11,
                      loc="upper right", frameon=True)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Robustness summary chart — bar chart of per-effect success rates
# ---------------------------------------------------------------------------

def plot_robustness(
    robustness_result: Dict,
    effect_labels: List[str],
    title: str = "Robustness: fraction of parameter combinations\ncorrectly predicting each effect",
    use_latex: bool = True,
    figsize: Tuple[float, float] = (7.0, 4.0),
) -> plt.Figure:
    """
    Bar chart showing what fraction of parameter combinations correctly
    predict each target effect.

    Parameters
    ----------
    robustness_result : dict
        Output of SEMScenario.robustness_test().
    effect_labels : list[str]
        Human-readable labels for each effect.
    """
    _apply_style(use_latex)

    total      = robustness_result["total"]
    per_effect = robustness_result["per_effect"]
    rates      = [n / total for n in per_effect]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(rates)), rates, color="steelblue", edgecolor="white")

    # Annotate with percentage
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rate * 100:.1f}%",
            ha="center", va="bottom", fontsize=10
        )

    ax.set_xticks(range(len(effect_labels)))
    ax.set_xticklabels(effect_labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Fraction correct", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=11)

    # Annotate overall rate
    overall = robustness_result["rate_all"]
    ax.text(
        0.99, 0.03,
        f"All effects: {overall * 100:.1f}%",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray")
    )

    plt.tight_layout()
    return fig
