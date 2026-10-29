"""
EXPLORATORY (not part of the committed pipeline).

Re-run the four-variant SEM comparison using phi matrices derived at p<.05
instead of the committed p<.01, to see whether a looser threshold improves the
full model's fit. Monkeypatches the IMPACT matrices at runtime only — the
committed values in cs1_imprecision.py (p<.01) are NOT changed.

Run: PYTHONPATH=. python scripts/_explore_threshold.py
"""

from sem.impact_derivation import (
    derive_trivalent_matrix, CS1_SEM_MAPPING, CS1_SEM_ATTR_MAPPING,
)
import case_studies.cs1_imprecision as cs1
from sem.comparison import compare_all_variants, print_comparison_table

DATA  = "data/cs1_experiment/processed/experiment1.csv"
MEANS = "data/cs1_experiment/processed/empirical_means_cs1.json"
ROWS  = ["k_p", "k_np", "m_max", "m_sit", "m_min"]
COLS  = ["a_comp", "a_like", "a_ped"]


def split(matrix):
    K = {k: matrix[k] for k in ("k_p", "k_np")}
    M = {m: matrix[m] for m in ("m_max", "m_sit", "m_min")}
    return K, M


def show_diff(m01, m05):
    print("phi matrices: p<.01  ->  p<.05   (■ = changed)\n")
    print(f"{'':7s}" + "".join(f"{c:>10s}" for c in COLS))
    for r in ROWS:
        cells = []
        for c in COLS:
            a, b = m01[r][c], m05[r][c]
            mark = "" if a == b else " ■"
            cells.append(f"{a:+d}->{b:+d}{mark}")
        print(f"{r:7s}" + "".join(f"{x:>10s}" for x in cells))
    print()


def main():
    m01 = derive_trivalent_matrix(DATA, CS1_SEM_MAPPING, CS1_SEM_ATTR_MAPPING, 0.01)
    m05 = derive_trivalent_matrix(DATA, CS1_SEM_MAPPING, CS1_SEM_ATTR_MAPPING, 0.05)
    show_diff(m01, m05)

    # Monkeypatch to the p<.05 matrices (runtime only).
    K05, M05 = split(m05)
    cs1.IMPACT_K = K05
    cs1.IMPACT_M = M05

    print("Four-variant comparison with phi derived at p<.05:\n")
    comparison = compare_all_variants(MEANS, n_samples=100, seed=42)
    print_comparison_table(comparison)


if __name__ == "__main__":
    main()
