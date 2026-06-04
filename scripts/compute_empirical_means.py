"""
scripts/compute_empirical_means.py
==================================
Pre-compute the empirical Task-1 attribute means for Case Study 1 and store them
as a JSON lookup file consumed by ``sem/fit.py``.

This is a one-time preprocessing step: ``sem/fit.py`` then works only with these
pre-computed means (not the raw CSV), keeping the fit module self-contained and
fast.  Means are stored RAW (1-7 Likert); normalization to [-1, 1] happens inside
``sem/fit.py`` at call time so the JSON stays human-readable.

Run:
    PYTHONPATH=. python scripts/compute_empirical_means.py
"""

import json
import os

import pandas as pd

DATA_PATH = "data/cs1_experiment/processed/experiment1.csv"
OUT_PATH = "data/cs1_experiment/processed/empirical_means_cs1.json"

# Mapping from CSV values to SEM symbols (verified against the data).
FORM_MAP = {"precise": "v_prc", "approx": "v_apx"}
CONTEXT_MAP = {"highPr": "c_HP", "lowPr": "c_LP"}
ATTR_MAP = {"a_comp": "competent", "a_like": "likeable", "a_ped": "pedantic"}


def compute_means(data_path: str = DATA_PATH) -> dict:
    """Return the nested means dict (+ _meta) for the CS1 data."""
    df = pd.read_csv(data_path)

    result: dict = {a: {} for a in ATTR_MAP}
    n_per_condition: dict = {}

    for form_val, v in FORM_MAP.items():
        for ctx_val, c in CONTEXT_MAP.items():
            cell = df[(df["form"] == form_val) & (df["context"] == ctx_val)]
            n_per_condition[f"{v}_{c}"] = int(len(cell))
            for sem_attr, raw_col in ATTR_MAP.items():
                result[sem_attr].setdefault(v, {})[c] = round(
                    float(cell[raw_col].mean()), 4
                )

    result["_meta"] = {
        "scale": "7-point Likert (1-7, raw, not normalized)",
        "source": data_path,
        "n_total": int(len(df)),
        "n_per_condition": n_per_condition,
    }
    return result


def main() -> None:
    means = compute_means(DATA_PATH)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(means, fh, indent=2)

    # Summary table for manual sanity-checking (12 means).
    print(f"Wrote {OUT_PATH}")
    print(f"N = {means['_meta']['n_total']}  "
          f"per-condition: {means['_meta']['n_per_condition']}\n")
    print(f"{'attribute':<10}{'form':<8}{'c_HP':>8}{'c_LP':>8}")
    print("-" * 34)
    for a in ATTR_MAP:
        for v in ("v_prc", "v_apx"):
            print(f"{a:<10}{v:<8}{means[a][v]['c_HP']:>8.2f}{means[a][v]['c_LP']:>8.2f}")
        print()


if __name__ == "__main__":
    main()
