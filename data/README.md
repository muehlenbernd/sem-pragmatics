# Data

This folder contains experimental data from the matched-guise study
reported in Case Study 1 of the paper (numerical im/precision).

---

## Study Overview

Two experiments, both using a 2 × 2 between-subjects design
(context: high / low precision required × utterance form: precise / approximate).
Six pre-tested scenarios, one item per participant.

| Experiment | N | Knowledge condition | knowledge (CSV) | Recruitment note |
|---|---|---|---|---|
| Exp. 1 | 362 | Base (knowledge not established) | `unknown` | |
| Exp. 2 | 390 | Established (speaker told exact value) | `knowl` | 391 recruited; one repeat participant removed before analysis |

### Tasks

**Task 1 — Speaker ratings (7-point Likert)**  
Participants rated the speaker on six attributes:
`COMPETENT`, `KNOWLEDGEABLE`, `WELL-PREPARED`, `LIKEABLE`, `HELPFUL`, `PEDANTIC`

**Task 2 — Motivation inference (checkboxes)**  
Participants indicated why they think the speaker used the precise/approximate
expression. Each item is a binary column `t2_<proxy>` (1 = checked, 0 = not).

**Task 3 — Speaker inference (checkboxes)**  
Participants indicated what they inferred about the speaker's knowledge and
situation. Each item is a binary column `t3_<proxy>`.

See `codebook.csv` for the full variable list with descriptions.

---

## Folder Structure

```
data/
├── README.md                        # This file
└── cs1_experiment/
    ├── codebook.csv                 # Variable descriptions for both experiments
    ├── processed/
    │   ├── experiment1.csv          # Exp. 1 cleaned data (N = 362)
    │   ├── experiment2.csv          # Exp. 2 cleaned data (N = 390)
    │   ├── empirical_means_cs1.json     # Cell means, Exp. 1
    │   ├── empirical_means_cs1_exp2.json # Cell means, Exp. 2
    │   ├── impact_tables_cs1.json       # φ_K, φ_M from Task 2 (Exp. 1)
    │   ├── variant_comparison_cs1.json  # Variant-comparison metrics
    │   ├── joint_listener_cs1.json      # Staged vs. joint comparison
    │   ├── within_condition_check_cs1.json
    │   ├── exp2_knowledge_test_cs1.json     # D1 analysis results
    │   ├── exp2_knowledge_followup_cs1.json # Session F follow-up (a)–(d)
    │   └── exp2_access_split_cs1.json       # Session G access-split analysis
    └── analysis/
        └── README.md                # Note: R scripts not included in this repo
```

---

## Data Availability

The processed data are included here. The full materials, raw data and
preregistration are archived publicly; the link will be added after peer review.

---

## Variable Codebook (Task 1 ratings)

| Variable | Type | Values | Description |
|---|---|---|---|
| `participant_id` | int | — | Anonymized participant identifier |
| `scenario` | string | `bike`, `cinema`, ... | Which of the 6 scenarios |
| `context` | string | `highPr`, `lowPr` | High / low precision required |
| `form` | string | `precise`, `approx` | Utterance form shown |
| `knowledge` | string | `unknown`, `knowl` | Exp. 1 / Exp. 2 condition |
| `competent` | int | 1–7 | Rating on COMPETENT |
| `knowledgeable` | int | 1–7 | Rating on KNOWLEDGEABLE |
| `well_prepared` | int | 1–7 | Rating on WELL-PREPARED |
| `likeable` | int | 1–7 | Rating on LIKEABLE |
| `helpful` | int | 1–7 | Rating on HELPFUL |
| `pedantic` | int | 1–7 | Rating on PEDANTIC |

See `codebook.csv` for the full variable list including `t2_` and `t3_` columns.
