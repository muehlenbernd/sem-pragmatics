"""
scripts/run_comparison.py
=========================
Run the four-variant SEM comparison (Task 2) and save the summarized results.

Runs ``compare_all_variants`` with the default settings (n=100, seed=42), prints
the comparison table, and writes the summary dict to
``data/cs1_experiment/processed/variant_comparison_cs1.json``.

Run:
    PYTHONPATH=. python scripts/run_comparison.py
"""

import json
import os

from sem.comparison import compare_all_variants, print_comparison_table

EMPIRICAL_JSON = "data/cs1_experiment/processed/empirical_means_cs1.json"
OUT_PATH = "data/cs1_experiment/processed/variant_comparison_cs1.json"


def _to_builtin(obj):
    """Recursively cast numpy scalars to Python floats for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    try:
        return float(obj)
    except (TypeError, ValueError):
        return obj


def main() -> None:
    comparison = compare_all_variants(EMPIRICAL_JSON, n_samples=100, seed=42)

    print_comparison_table(comparison)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(_to_builtin(comparison), fh, indent=2)
    print(f"\nSaved summary to {OUT_PATH}")


if __name__ == "__main__":
    main()
