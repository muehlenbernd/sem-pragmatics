"""
sem/impact_derivation.py
========================
Derives empirical trivalent impact matrices (φ_K, φ_M) from experimental data.

Procedure
---------
1. Treat each t2_ motivation checkbox as a proxy for a SEM variable (K or M).
2. For each proxy × raw attribute pair, run an independent-samples t-test
   (checked vs. not-checked participants).
3. Apply a significance threshold: if p < α and diff > 0 → +1;
   if p < α and diff < 0 → −1; otherwise → 0.
4. Map raw attributes directly to SEM attributes (one-to-one, no aggregation):
     a_comp → competent
     a_like → likeable
     a_ped  → pedantic
5. Where multiple proxies map to one SEM variable, take majority vote across
   proxies.

Usage
-----
    from sem.impact_derivation import (
        derive_trivalent_matrix,
        derive_all_thresholds,
        save_impact_tables,
        print_impact_table,
        CS1_SEM_MAPPING,
        CS1_SEM_ATTR_MAPPING,
    )
"""

import json
from typing import Dict, List, Optional

import pandas as pd
from scipy.stats import ttest_ind


# ---------------------------------------------------------------------------
# Default mappings for Case Study 1 (Imprecision)
# ---------------------------------------------------------------------------

CS1_SEM_MAPPING: Dict[str, List[str]] = {
    "k_p":   ["knewExact", "infoAvail"],
    "k_np":  ["notKnow", "infoNotAvail"],
    "m_max": ["fussyPerson"],
    "m_sit": ["needed", "purpose", "realWorld"],
    "m_min": ["easySpeaker", "usualApx"],
}

# Direct one-to-one mapping: one raw attribute per SEM attribute.
# knowledgeable, well_prepared, and helpful are intentionally excluded
# to avoid the theoretical assumption that they proxy the same construct.
CS1_SEM_ATTR_MAPPING: Dict[str, List[str]] = {
    "a_comp": ["competent"],
    "a_like": ["likeable"],
    "a_ped":  ["pedantic"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _majority_vote(values: List[int]) -> int:
    """
    Return the majority value from a list of trivalent integers {-1, 0, +1}.

    Rule: sum > 0 → +1, sum < 0 → −1, sum == 0 → 0.
    A single non-zero vote therefore wins when all others are 0.
    """
    if not values:
        return 0
    s = sum(values)
    if s > 0:
        return 1
    if s < 0:
        return -1
    return 0


def _trivalent(diff: float, p_value: float, alpha: float) -> int:
    """Map a (mean diff, p-value) pair to {-1, 0, +1}."""
    if p_value < alpha:
        return 1 if diff > 0 else -1
    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def derive_trivalent_matrix(
    data_path: str,
    sem_mapping: Dict[str, List[str]],
    sem_attr_mapping: Dict[str, List[str]],
    alpha: float,
) -> Dict[str, Dict[str, int]]:
    """
    Derive a trivalent impact matrix from experimental data.

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the experimental data.
    sem_mapping : dict
        Maps each SEM variable name to a list of proxy column suffixes
        (without the 't2_' prefix).  E.g. {"k_p": ["knewExact"], ...}
    sem_attr_mapping : dict
        Maps each SEM attribute name to a list of raw rating columns.
        E.g. {"a_comp": ["competent"], ...}
    alpha : float
        Significance threshold for t-tests.

    Returns
    -------
    dict
        result[sem_var][sem_attr] → int in {-1, 0, +1}
    """
    df = pd.read_csv(data_path)

    # Identify all t2 binary columns (exclude the free-text t2_response column)
    t2_binary_cols = [
        c for c in df.columns
        if c.startswith("t2_") and c != "t2_response"
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Skip rows where t2 column sum < 5 (insufficient data)
    # This filters by per-proxy column sums: proxies with fewer than 5
    # checked participants are excluded from the analysis.
    valid_proxy_cols = {
        c for c in t2_binary_cols if int(df[c].sum()) >= 5
    }

    result: Dict[str, Dict[str, int]] = {}

    for sem_var, proxies in sem_mapping.items():
        result[sem_var] = {}

        for sem_attr, raw_attrs in sem_attr_mapping.items():
            # Collect one vote per proxy for this (sem_var, sem_attr) pair
            proxy_votes: List[int] = []

            for proxy in proxies:
                col = f"t2_{proxy}"
                if col not in df.columns or col not in valid_proxy_cols:
                    continue

                checked     = df[df[col] == 1]
                not_checked = df[df[col] == 0]

                if len(checked) < 2 or len(not_checked) < 2:
                    proxy_votes.append(0)
                    continue

                # One vote per raw attribute → SEM-attr value
                raw_votes: List[int] = []
                for raw_attr in raw_attrs:
                    if raw_attr not in df.columns:
                        continue
                    g1 = checked[raw_attr].dropna()
                    g2 = not_checked[raw_attr].dropna()
                    if len(g1) < 2 or len(g2) < 2:
                        raw_votes.append(0)
                        continue
                    _, p = ttest_ind(g1, g2, equal_var=True)
                    diff = float(g1.mean() - g2.mean())
                    raw_votes.append(_trivalent(diff, float(p), alpha))

                proxy_votes.append(_majority_vote(raw_votes))

            result[sem_var][sem_attr] = _majority_vote(proxy_votes)

    return result


def derive_all_thresholds(
    data_path: str,
    sem_mapping: Dict[str, List[str]],
    sem_attr_mapping: Dict[str, List[str]],
    alphas: List[float] = None,
) -> Dict[float, Dict[str, Dict[str, int]]]:
    """
    Derive trivalent matrices for multiple significance thresholds.

    Parameters
    ----------
    data_path : str
        Path to the CSV file.
    sem_mapping : dict
        Proxy-to-SEM-variable mapping (see derive_trivalent_matrix).
    sem_attr_mapping : dict
        Raw-attr-to-SEM-attr mapping (see derive_trivalent_matrix).
    alphas : list of float, optional
        Significance thresholds to evaluate.
        Defaults to [0.001, 0.01, 0.05, 0.10].

    Returns
    -------
    dict
        result[alpha][sem_var][sem_attr] → int in {-1, 0, +1}
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.05, 0.10]

    return {
        alpha: derive_trivalent_matrix(
            data_path, sem_mapping, sem_attr_mapping, alpha
        )
        for alpha in alphas
    }


def save_impact_tables(tables: Dict, output_path: str) -> None:
    """
    Save the output of derive_all_thresholds to a JSON file.

    Parameters
    ----------
    tables : dict
        Output of derive_all_thresholds: tables[alpha][sem_var][sem_attr].
    output_path : str
        Destination file path for the JSON output.

    JSON structure
    --------------
    {"0.001": {sem_var: {sem_attr: value}}, "0.01": {...}, ...}
    """
    serialisable = {str(alpha): matrix for alpha, matrix in tables.items()}
    with open(output_path, "w") as fh:
        json.dump(serialisable, fh, indent=2)


def print_impact_table(
    matrix: Dict[str, Dict[str, int]],
    current_model: Optional[Dict[str, Dict[str, int]]] = None,
    title: str = "",
) -> None:
    """
    Pretty-print a single threshold's impact matrix.

    Parameters
    ----------
    matrix : dict
        matrix[sem_var][sem_attr] → int in {-1, 0, +1}.
    current_model : dict, optional
        If provided, cells that differ from current_model are marked with ■.
    title : str, optional
        Header line printed above the table.
    """
    row_order = ["k_p", "k_np", "m_max", "m_sit", "m_min"]
    col_order = ["a_comp", "a_like", "a_ped"]

    if title:
        print(title)

    header = f"{'':8s}" + "".join(f"  {c:>8s}" for c in col_order)
    print(header)
    print("-" * len(header))

    for row in row_order:
        if row not in matrix:
            continue
        parts = []
        for col in col_order:
            val = matrix[row].get(col, 0)
            sign = f"{val:+d}"
            if current_model and current_model.get(row, {}).get(col, 0) != val:
                sign += "■"
            parts.append(f"  {sign:>9s}")
        print(f"{row:<8s}" + "".join(parts))
    print()
