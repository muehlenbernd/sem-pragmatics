# Data

This folder contains experimental data from the matched-guise study
reported in **Case Study 1** of the paper (numerical im/precision).

---

## Study Overview

- **Participants:** 762 native English speakers (US-based, via Prolific)
- **Design:** 2 × 2 between-subjects (Precision level: high/low × Form: precise/approximate)
- **Scenarios:** 6 pre-tested scenarios (1 item per participant)
- **Compensation:** £1.20

### Tasks

**Task 1 — Speaker ratings (7-point Likert)**  
Participants rated the speaker on 6 attributes:
`COMPETENT`, `KNOWLEDGEABLE`, `WELL-PREPARED`, `LIKEABLE`, `HELPFUL`, `PEDANTIC`

**Task 2 — Motivation inference (free text)**  
Participants described why they think the speaker used the precise/approximate
expression rather than the alternative.

---

## Folder Structure

```
data/
├── README.md                        # This file
└── cs1_experiment/
    ├── README.md                    # Detailed variable codebook
    ├── raw/
    │   └── [data files — see access note below]
    ├── processed/
    │   └── [cleaned data — see access note below]
    └── analysis/
        ├── mixed_effects_models.R   # Linear mixed effects models (Task 1)
        ├── motivation_coding.R      # Motivation class analysis (Task 2)
        └── README.md                # How to run the R scripts
```

---

## Access and Ethics

The raw data contains participant responses collected under IRB approval.
Before sharing, responses are anonymized (participant IDs replaced with
random tokens, free-text responses checked for identifying information).

**Data availability:**  
The processed, anonymized dataset will be made available upon paper
acceptance. If you need access for review purposes, please contact the
corresponding author.

**Ethics approval:** [Institution IRB reference — to be added]

---

## Variable Codebook (Task 1)

| Variable | Type | Values | Description |
|---|---|---|---|
| `participant_id` | string | anon token | Anonymized participant identifier |
| `scenario` | string | `bike`, `train`, ... | Which of the 6 scenarios |
| `context` | string | `HPR`, `LPR` | High / low precision required |
| `form` | string | `precise`, `approx` | Utterance form shown |
| `competent` | int | 1–7 | Rating on COMPETENT |
| `knowledgeable` | int | 1–7 | Rating on KNOWLEDGEABLE |
| `well_prepared` | int | 1–7 | Rating on WELL-PREPARED |
| `likeable` | int | 1–7 | Rating on LIKEABLE |
| `helpful` | int | 1–7 | Rating on HELPFUL |
| `pedantic` | int | 1–7 | Rating on PEDANTIC |

## Variable Codebook (Task 2)

| Variable | Type | Values | Description |
|---|---|---|---|
| `participant_id` | string | anon token | Anonymized participant identifier |
| `scenario` | string | — | Scenario label |
| `form` | string | `precise`, `approx` | Which form the participant saw |
| `motivation_text` | string | free text | Raw motivation response |
| `motivation_class` | string | see below | Coded motivation category |

### Motivation classes (10 most frequent)

| Class | Description |
|---|---|
| `KNEWEXACT` | Speaker knew / was confident about exact information |
| `NOTKNOW` | Speaker did not know / was uncertain |
| `FUSSYPERSON` | Speaker is detail-oriented / pedantic by nature |
| `EASYSPEAKER` | Using approximate form was easier / faster |
| `USUALAPX` | Speaker usually uses approximate forms |
| `NEEDED` | The level of precision was required / called for |
| `REALWORLD` | Real-world consequences of the communicated information |
| `PURPOSE` | Speaker was purposefully precise / approximate |
| `INFONOTAVAIL` | Precise information was not available |
| `INFOAVAIL` | Precise information was available |

---

## R Analysis Scripts

The statistical analyses use R (v4.3.2) with the `lmerTest` package.
To reproduce:

```r
# Install dependencies
install.packages(c("lmerTest", "ggplot2", "dplyr"))

# Run main analysis
source("cs1_experiment/analysis/mixed_effects_models.R")
```

See `cs1_experiment/analysis/README.md` for full instructions.
