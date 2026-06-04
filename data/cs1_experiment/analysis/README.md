# R Analysis Scripts

## Requirements

- R >= 4.3.0
- Packages: `lmerTest`, `ggplot2`, `dplyr`, `tidyr`

Install with:
```r
install.packages(c("lmerTest", "ggplot2", "dplyr", "tidyr"))
```

## Scripts

| Script | Purpose |
|---|---|
| `mixed_effects_models.R` | Linear mixed effects models for Task 1 ratings |
| `motivation_coding.R` | Analysis of Task 2 motivation classes and correlations |

## Running

```bash
Rscript mixed_effects_models.R
Rscript motivation_coding.R
```

Both scripts expect processed data in `../processed/`. See `../README.md`
for data access information.

## Output

Results are printed to console and saved as `.csv` files in `../processed/`.
