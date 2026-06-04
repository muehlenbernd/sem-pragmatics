"""
scripts/derive_impact_tables.py
================================
Derive empirical trivalent impact matrices from the CS1 experiment data
and save lookup tables for all significance thresholds.

Usage
-----
    PYTHONPATH=. python scripts/derive_impact_tables.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sem.impact_derivation import (
    CS1_SEM_MAPPING,
    CS1_SEM_ATTR_MAPPING,
    derive_all_thresholds,
    save_impact_tables,
    print_impact_table,
)
from case_studies.cs1_imprecision import IMPACT_K, IMPACT_M

DATA_PATH   = "data/cs1_experiment/processed/experiment1.csv"
OUTPUT_PATH = "data/cs1_experiment/processed/impact_tables_cs1.json"

CURRENT_MODEL = {**IMPACT_K, **IMPACT_M}


def main():
    print(f"Loading data from: {DATA_PATH}")
    tables = derive_all_thresholds(
        data_path=DATA_PATH,
        sem_mapping=CS1_SEM_MAPPING,
        sem_attr_mapping=CS1_SEM_ATTR_MAPPING,
    )

    save_impact_tables(tables, OUTPUT_PATH)
    print(f"Saved impact tables to: {OUTPUT_PATH}\n")

    for alpha in [0.001, 0.01, 0.05, 0.10]:
        print_impact_table(
            tables[alpha],
            current_model=CURRENT_MODEL,
            title=f"=== p < {alpha} (■ = differs from current model) ===",
        )

    # Compact summary for verification
    print("=== Summary: p < 0.01 ===")
    m = tables[0.01]
    for row in ["k_p", "k_np", "m_max", "m_sit", "m_min"]:
        c = m[row]["a_comp"]
        l = m[row]["a_like"]
        p = m[row]["a_ped"]
        print(f"{row:<5}: a_comp={c:+d}, a_like={l:+d}, a_ped={p:+d}")


if __name__ == "__main__":
    main()
