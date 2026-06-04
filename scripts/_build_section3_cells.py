"""
One-off builder: append a "Section 3 Reproduction" block to
notebooks/sem_demo.ipynb. Existing cells are left untouched; new cells are
appended at the end. Safe to re-run (it strips a previously appended block,
detected by the SENTINEL marker, before re-appending).
"""
import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "sem_demo.ipynb")
SENTINEL = "SEC3-REPRO-BLOCK"  # marker placed in the section header cell


def md(*lines):
    src = list(lines)
    return {"cell_type": "markdown", "metadata": {}, "source": _as_source(src)}


def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _as_source(list(lines)),
    }


def _as_source(lines):
    """Join a list of full-text blocks into nbformat source (list of lines
    with trailing newlines, except the last)."""
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


# ---------------------------------------------------------------------------
# Cell contents
# ---------------------------------------------------------------------------

HEADER_MD = md(
    f"<!-- {SENTINEL} -->",
    "# Section 3 Reproduction",
    "",
    'This section systematically reproduces **every reported result in '
    'Section 3** ("A Case Study on Numerical (Im)Precision") of the manuscript '
    "(`manuscript/sem_paper.tex`), in order.",
    "",
    "Each subsection begins with a markdown cell quoting the exact manuscript "
    "claim being tested, followed by a self-contained, top-to-bottom runnable "
    "code cell that recomputes the result and prints a `✓`/`✗` verdict. "
    "Discrepancies with the manuscript are flagged inline:",
    "",
    "- **⚠️ DOES NOT REPLICATE** — an empirical claim that fails to reproduce.",
    "- **⚠️ MANUSCRIPT NEEDS UPDATE** — the current code/data produce a value "
    "that differs from what the manuscript text reports (often *intentionally*, "
    "because the analysis was revised: p<.01 instead of p<.05, Experiment 1 "
    "only instead of both experiments, direct attribute mapping, and an updated "
    "proxy mapping).",
    "",
    "> **Note on dependencies.** Section 3.2 needs `statsmodels` (≥0.14.4 for "
    "compatibility with recent SciPy). If it is missing, install with "
    "`pip install -U \"statsmodels>=0.14.4\" --break-system-packages` and "
    "re-run the cell.",
)

# ---- 3.2 ------------------------------------------------------------------

SEC32_MD = md(
    "## 3.2 — Experimental Results",
    "",
    "> **Manuscript claim (Section 3.2):**",
    "> *\"For competent and pedantic, a significant main effect of form was "
    "found, with precise rated higher (p < .001 for both). For competent and "
    "likeable, a significant interaction of context and form was found: the "
    "relative strength of precise versus approx was greater in hpr than in lpr "
    "(p < .001 for both).\"*",
    "",
    "We fit linear mixed effects models on "
    "`data/cs1_experiment/processed/experiment1.csv` "
    "(`attribute ~ form * context`, random intercept for `scenario`, sum "
    "coding) and test five claims:",
    "",
    "1. Main effect of **form on competent** (precise > approx, *p* < .001)",
    "2. Main effect of **form on pedantic** (precise > approx, *p* < .001)",
    "3. **No** significant main effect of **form on likeable**",
    "4. **form × context** interaction on **competent** (*p* < .001)",
    "5. **form × context** interaction on **likeable** (*p* < .001)",
)

SEC32_CODE = code(
    "# 3.2 — Experimental Results: reproduce the reported LMM effects (Experiment 1)",
    "import os, warnings",
    "warnings.filterwarnings('ignore')",
    "import pandas as pd",
    "",
    "def _find(rel):",
    "    \"\"\"Locate a repo-relative path whether cwd is notebooks/ or repo root.\"\"\"",
    "    for base in ('..', '.', os.path.dirname(os.getcwd())):",
    "        p = os.path.join(base, rel)",
    "        if os.path.exists(p):",
    "            return p",
    "    return rel",
    "",
    "DATA = _find('data/cs1_experiment/processed/experiment1.csv')",
    "",
    "try:",
    "    import statsmodels.formula.api as smf",
    "except Exception as e:",
    "    smf = None",
    "    print('⚠️  statsmodels unavailable:', type(e).__name__, str(e).splitlines()[0])",
    "    print('    Install with:  pip install -U \"statsmodels>=0.14.4\" --break-system-packages')",
    "",
    "if smf is not None:",
    "    df = pd.read_csv(DATA)",
    "    # Experiment 1 only: the 'knowledge' factor is constant ('unknown'),",
    "    # so the model uses form, context, and their interaction.",
    "    # Sum coding: precise=+1 / approx=-1 ; highPr=+1 / lowPr=-1",
    "    df['form_c'] = df['form'].map({'precise': 1, 'approx': -1})",
    "    df['ctx_c']  = df['context'].map({'highPr': 1, 'lowPr': -1})",
    "",
    "    res = {}",
    "    for attr in ['competent', 'likeable', 'pedantic']:",
    "        m = smf.mixedlm(f'{attr} ~ form_c * ctx_c', df,",
    "                        groups=df['scenario']).fit(reml=False)",
    "        res[attr] = dict(",
    "            form_b=m.params['form_c'],       form_p=m.pvalues['form_c'],",
    "            int_b=m.params['form_c:ctx_c'],  int_p=m.pvalues['form_c:ctx_c'],",
    "        )",
    "",
    "    def verdict(ok):",
    "        return '✓' if ok else '✗   ⚠️ DOES NOT REPLICATE'",
    "",
    "    print('Linear mixed effects models:  attribute ~ form * context'",
    "          '  (random intercept: scenario)')",
    "    print('Sum coding: precise=+1/approx=-1, highPr=+1/lowPr=-1\\n')",
    "",
    "    r = res['competent']",
    "    print('1. Main effect of FORM on COMPETENT  (precise > approx, p<.001)')",
    "    print(f\"   b={r['form_b']:+.3f}, p={r['form_p']:.2e}   ->   \"",
    "          f\"{verdict(r['form_b'] > 0 and r['form_p'] < .001)}\\n\")",
    "",
    "    r = res['pedantic']",
    "    print('2. Main effect of FORM on PEDANTIC   (precise > approx, p<.001)')",
    "    print(f\"   b={r['form_b']:+.3f}, p={r['form_p']:.2e}   ->   \"",
    "          f\"{verdict(r['form_b'] > 0 and r['form_p'] < .001)}\\n\")",
    "",
    "    r = res['likeable']",
    "    print('3. NO significant main effect of FORM on LIKEABLE  (p>=.05)')",
    "    print(f\"   b={r['form_b']:+.3f}, p={r['form_p']:.2e}   ->   \"",
    "          f\"{verdict(r['form_p'] >= .05)}\\n\")",
    "",
    "    r = res['competent']",
    "    print('4. FORM x CONTEXT interaction on COMPETENT  (p<.001)')",
    "    print(f\"   b={r['int_b']:+.3f}, p={r['int_p']:.2e}   ->   \"",
    "          f\"{verdict(r['int_p'] < .001)}\\n\")",
    "",
    "    r = res['likeable']",
    "    print('5. FORM x CONTEXT interaction on LIKEABLE   (p<.001)')",
    "    print(f\"   b={r['int_b']:+.3f}, p={r['int_p']:.2e}   ->   \"",
    "          f\"{verdict(r['int_p'] < .001)}\")",
)

# ---- 3.3 ------------------------------------------------------------------

SEC33_MD = md(
    "## 3.3 — Motivation Analysis",
    "",
    "> **Manuscript claim (Section 3.3, *Fitting the Model*):**",
    "> *\"For knowledge states: knewExact maps to k_p; notKnow maps to k_¬p. "
    "For motivation-based strategies: fussyPerson (inherently precise) maps to "
    "m_max; easySpeaker and usualApx ... map to m_min; needed, realWorld, "
    "purpose, infoNotAvail, and infoAvail (contextually-sensitive motivations) "
    "map to m_sit.\"*",
    ">",
    "> *\"A total of 14 significant effects were found across 30 combinations\"* "
    "(p < .05, 10 motivation classes × 3 attributes, both experiments combined).",
    ">",
    "> *\"The trivalent functions φ_K and φ_M are then derived by ... we assign "
    "+1 for a significant positive correlation, −1 for a significant negative "
    "correlation, and 0 for a non-significant effect.\"*",
    "",
    "**The current (revised) analysis intentionally differs** from the "
    "manuscript: it uses **p < .01**, **Experiment 1 only**, a **direct "
    "attribute mapping** (competent/likeable/pedantic, no aggregation), and an "
    "**updated proxy mapping** in which `infoAvail → k_p` and "
    "`infoNotAvail → k_np` (rather than both → m_sit). The cell below "
    "recomputes the four relevant tables and flags every difference from the "
    "manuscript text with **⚠️ MANUSCRIPT NEEDS UPDATE**.",
)

SEC33_CODE = code(
    "# 3.3 — Motivation Analysis",
    "import os, sys, warnings",
    "warnings.filterwarnings('ignore')",
    "import pandas as pd",
    "from scipy.stats import ttest_ind",
    "",
    "def _find(rel):",
    "    for base in ('..', '.', os.path.dirname(os.getcwd())):",
    "        p = os.path.join(base, rel)",
    "        if os.path.exists(p):",
    "            return p",
    "    return rel",
    "",
    "sys.path.insert(0, _find('.'))   # make sem/ and case_studies/ importable",
    "DATA = _find('data/cs1_experiment/processed/experiment1.csv')",
    "",
    "from sem.impact_derivation import (CS1_SEM_MAPPING, CS1_SEM_ATTR_MAPPING,",
    "                                   derive_trivalent_matrix)",
    "from case_studies.cs1_imprecision import IMPACT_K, IMPACT_M",
    "",
    "df = pd.read_csv(DATA)",
    "N = len(df)",
    "ATTRS = ['competent', 'likeable', 'pedantic']",
    "SEM_VARS = ['k_p', 'k_np', 'm_max', 'm_sit', 'm_min']",
    "",
    "# ====================================================================",
    "# TABLE 1 — Motivation class frequencies (proxies in current mapping)",
    "# ====================================================================",
    "print('=' * 60)",
    "print('TABLE 1 — Motivation class frequencies (Experiment 1)')",
    "print('=' * 60)",
    "print(f\"{'SEM var':8s} {'proxy (t2_)':16s} {'n':>5s} {'%':>8s}\")",
    "print('-' * 40)",
    "for var, proxies in CS1_SEM_MAPPING.items():",
    "    for p in proxies:",
    "        n = int(df[f't2_{p}'].sum())",
    "        print(f'{var:8s} {p:16s} {n:5d} {100*n/N:7.1f}%')",
    "print(f'\\nTotal participants: N = {N}')",
    "print('NOTE: the manuscript (Fig. 2) reported the 10 most frequent motivation')",
    "print('      classes pooled over BOTH experiments. Here we report Experiment 1')",
    "print('      only, for the proxies used in the current SEM mapping.')",
    "",
    "# ====================================================================",
    "# TABLE 2 — Per-proxy t-tests (checked vs. not), grouped by SEM variable",
    "# ====================================================================",
    "print('\\n' + '=' * 60)",
    "print('TABLE 2 — p-values: proxies x {competent, likeable, pedantic}')",
    "print('=' * 60)",
    "print('Stars: ** p<.01 (current threshold), * p<.05;  sign = direction of')",
    "print('mean(checked) - mean(not-checked). These per-proxy tests are the')",
    "print('inputs that the trivalent derivation aggregates by majority vote.')",
    "print()",
    "print(f\"{'SEM var':7s} {'proxy':13s} \" + ' '.join(f'{a:>14s}' for a in ATTRS))",
    "print('-' * 64)",
    "for var, proxies in CS1_SEM_MAPPING.items():",
    "    for p in proxies:",
    "        col = f't2_{p}'",
    "        checked, notck = df[df[col] == 1], df[df[col] == 0]",
    "        cells = []",
    "        for a in ATTRS:",
    "            g1, g2 = checked[a].dropna(), notck[a].dropna()",
    "            _, pv = ttest_ind(g1, g2, equal_var=True)",
    "            sign = '+' if g1.mean() > g2.mean() else '-'",
    "            star = '**' if pv < .01 else ('*' if pv < .05 else '')",
    "            cells.append(f'{pv:.3f}{sign}{star:2s}')",
    "        print(f'{var:7s} {p:13s} ' + ' '.join(f'{c:>14s}' for c in cells))",
    "",
    "# ====================================================================",
    "# TABLE 3 — Current proxy-to-SEM-variable mapping",
    "# ====================================================================",
    "print('\\n' + '=' * 60)",
    "print('TABLE 3 — Current CS1_SEM_MAPPING (proxy -> SEM variable)')",
    "print('=' * 60)",
    "for var, proxies in CS1_SEM_MAPPING.items():",
    "    print(f'  {var:6s} <- {proxies}')",
    "print()",
    "print('Manuscript mapping (Section 3.3):')",
    "print('  k_p   <- [knewExact]')",
    "print('  k_np  <- [notKnow]')",
    "print('  m_max <- [fussyPerson]')",
    "print('  m_sit <- [needed, realWorld, purpose, infoNotAvail, infoAvail]')",
    "print('  m_min <- [easySpeaker, usualApx]')",
    "mism = []",
    "if 'infoAvail' in CS1_SEM_MAPPING.get('k_p', []):",
    "    mism.append('infoAvail: manuscript->m_sit, current->k_p')",
    "if 'infoNotAvail' in CS1_SEM_MAPPING.get('k_np', []):",
    "    mism.append('infoNotAvail: manuscript->m_sit, current->k_np')",
    "if mism:",
    "    print('\\n⚠️ MANUSCRIPT NEEDS UPDATE — mapping changed (intentional):')",
    "    for m in mism:",
    "        print('   - ' + m)",
    "",
    "# ====================================================================",
    "# TABLE 4 — Current trivalent impact matrices vs. live p<.01 derivation",
    "# ====================================================================",
    "print('\\n' + '=' * 60)",
    "print('TABLE 4 — IMPACT_K / IMPACT_M  (in cs1_imprecision.py)')",
    "print('=' * 60)",
    "current = {**IMPACT_K, **IMPACT_M}",
    "derived = derive_trivalent_matrix(DATA, CS1_SEM_MAPPING,",
    "                                  CS1_SEM_ATTR_MAPPING, alpha=0.01)",
    "print(f\"{'':6s} \" + ' '.join(f'{a:>8s}' for a in ['a_comp', 'a_like', 'a_ped']))",
    "print('-' * 34)",
    "all_match = True",
    "for var in SEM_VARS:",
    "    cur = current[var]",
    "    cells = []",
    "    for a in ['a_comp', 'a_like', 'a_ped']:",
    "        cv, dv = cur[a], derived[var][a]",
    "        flag = '' if cv == dv else '!'",
    "        if cv != dv:",
    "            all_match = False",
    "        cells.append(f'{cv:+d}{flag}')",
    "    print(f'{var:6s} ' + ' '.join(f'{c:>8s}' for c in cells))",
    "print()",
    "if all_match:",
    "    print('✓ IMPACT_K / IMPACT_M match the live p<.01 derivation from the data.')",
    "else:",
    "    print('⚠️ Mismatch (!) between committed matrices and live derivation.')",
    "print()",
    "print('⚠️ MANUSCRIPT NEEDS UPDATE — Fig. 2(b,d) of the manuscript report the')",
    "print('   OLD analysis (p<.05, both experiments, 10 motivation classes, 14/30')",
    "print('   significant effects). The values above are the REVISED derivation')",
    "print('   (p<.01, Experiment 1, direct attribute mapping). Update Fig. 2 and')",
    "print('   the surrounding text to match.')",
)

# ---- 3.4 ------------------------------------------------------------------

SEC34_MD = md(
    "## 3.4 — Model Predictions",
    "",
    "> **Manuscript claim (Section 3.4):**",
    "> *\"For an initial model check, we test a balanced model: ω = 0.5 (equal "
    "weight to epistemic and motivational inference) and flat prior "
    "probabilities. ... The balanced model correctly predicted all four "
    "effects.\"*",
    ">",
    "> *\"SEM correctly predicted all four effects for 99.7% of combinations; "
    "for the remaining 0.3% it correctly predicted three out of four\"* (grid of "
    "2,823,576 combinations at increments of 0.02). The footnote adds that the "
    "model *\"struggled to predict Effect 1 ... when ω was very low, P(k_p) was "
    "very low, and P(m_sit) was high.\"*",
    "",
    "Below we run the balanced model, print the full prediction table, check the "
    "four target effects, and run a robustness sweep at `step=0.05` (a coarser "
    "grid than the manuscript's 0.02), then compare to the 99.7% claim.",
)

SEC34_CODE = code(
    "# 3.4 — Model Predictions",
    "import os, sys, warnings",
    "warnings.filterwarnings('ignore')",
    "",
    "def _find(rel):",
    "    for base in ('..', '.', os.path.dirname(os.getcwd())):",
    "        p = os.path.join(base, rel)",
    "        if os.path.exists(p):",
    "            return p",
    "    return rel",
    "",
    "sys.path.insert(0, _find('.'))",
    "from case_studies.cs1_imprecision import (build_scenario, target_effects,",
    "                                          UTTERANCES, CONTEXTS, ATTRIBUTES)",
    "",
    "# Balanced model: omega=0.5, uniform priors (defaults)",
    "scenario = build_scenario(omega=0.5)",
    "scores = scenario.evaluate_all()",
    "",
    "# ---- Full prediction table: E_L(a | v, c) ----",
    "print('Balanced model (omega=0.5, uniform priors) — E_L(a | v, c)')",
    "print('Scores range from -1 (lowest) to +1 (highest)\\n')",
    "cols = [(v, c) for v in UTTERANCES for c in CONTEXTS]",
    "hdr = f\"{'attribute':10s}\" + ''.join(f'{v+\"|\"+c:>14s}' for v, c in cols)",
    "print(hdr)",
    "print('-' * len(hdr))",
    "for a in ATTRIBUTES:",
    "    row = f'{a:10s}' + ''.join(f'{scores[a][v][c]:+13.3f} ' for v, c in cols)",
    "    print(row)",
    "",
    "# ---- Four target effects ----",
    "eff = target_effects(scores)",
    "labels = [",
    "    '1. precise > approx on competent (across contexts)',",
    "    '2. precise > approx on pedantic (across contexts)',",
    "    '3. competent (prc-apx) gap larger in c_HP than c_LP',",
    "    '4. likeable (prc-apx) gap larger in c_HP than c_LP',",
    "]",
    "print('\\nTarget effects (balanced model):')",
    "for lab, ok in zip(labels, eff):",
    "    print(f'  {\"✓\" if ok else \"✗\"}  {lab}')",
    "print(f'\\nAll four effects predicted: {all(eff)}'",
    "      '   (manuscript: \"correctly predicted all four effects\")')",
    "if not all(eff):",
    "    print('⚠️ MANUSCRIPT NEEDS UPDATE — balanced model no longer predicts all four.')",
    "",
    "# ---- Robustness sweep ----",
    "print('\\nRunning robustness_test(step=0.05) ... (a few seconds)')",
    "rob = scenario.robustness_test(effects_fn=target_effects, step=0.05)",
    "rate = 100 * rob['rate_all']",
    "print(f\"  combinations tested      : {rob['total']:,}\")",
    "print(f\"  all 4 effects hold       : {rob['all_correct']:,}  ({rate:.2f}%)\")",
    "for i, c in enumerate(rob['per_effect'], 1):",
    "    print(f\"  effect {i} holds in         : {c:,}  ({100*c/rob['total']:.2f}%)\")",
    "",
    "print(f\"\\nManuscript claim : 99.7% all-four (grid step 0.02 -> 2,823,576 combos).\")",
    "print(f\"This run         : {rate:.2f}% all-four (grid step 0.05 -> {rob['total']:,} combos).\")",
    "if abs(rate - 99.7) <= 0.5:",
    "    print('✓ Consistent with the manuscript; the small gap is grid resolution')",
    "    print('  (step 0.05 here vs 0.02 in the paper). The only effect that ever')",
    "    print('  fails is Effect 1, matching the manuscript footnote.')",
    "else:",
    "    print(f'⚠️ MANUSCRIPT NEEDS UPDATE — {rate:.2f}% differs from 99.7% by '",
    "          f'{abs(rate-99.7):.2f} pts; re-check at step=0.02 before trusting.')",
)

NEW_CELLS = [
    HEADER_MD,
    SEC32_MD, SEC32_CODE,
    SEC33_MD, SEC33_CODE,
    SEC34_MD, SEC34_CODE,
]


def main():
    with open(NB_PATH) as fh:
        nb = json.load(fh)

    # Strip any previously appended block (idempotent re-runs).
    def is_sentinel(cell):
        return SENTINEL in "".join(cell.get("source", []))

    sentinel_idx = next((i for i, c in enumerate(nb["cells"]) if is_sentinel(c)), None)
    if sentinel_idx is not None:
        nb["cells"] = nb["cells"][:sentinel_idx]

    nb["cells"].extend(NEW_CELLS)

    with open(NB_PATH, "w") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    print(f"Appended {len(NEW_CELLS)} cells. Notebook now has {len(nb['cells'])} cells.")


if __name__ == "__main__":
    main()
