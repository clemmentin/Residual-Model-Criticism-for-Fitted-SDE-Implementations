# Neural training variance comparison

These files reproduce the paper's neural conditional-score comparison and
supplementary training-variance figure. The plug-in and tangent objectives
use the same 12 training groups, normalization, initialization seeds and
conditioning prefixes. Both are evaluated with tangent-standardized residuals
and the six-component centring--energy score.

| Training paths | Training variance | Median CDF gap | Mean conditional rejection |
| ---: | --- | ---: | ---: |
| Oracle | Known parameters | 0.011 | 5.04% |
| 48 | Plug-in | 0.433 | 44.54% |
| 48 | Tangent | 0.104 | 13.30% |
| 192 | Plug-in | 0.232 | 23.73% |
| 192 | Tangent | 0.025 | 6.07% |

The nominal level is 5%; with 2,500 references the exact continuous-score
rank target is 0.04998001. Means summarize these 12 fixed training groups.
The simulation intervals measure error in their conditional score distributions,
not uncertainty across a population of future model fits. An interval containing
the nominal level does not establish calibration.

## Included results

- [combined_results.csv](combined_results.csv): all 60 unique conditional comparisons.
- [paired_differences.csv](paired_differences.csv): tangent-minus-plug-in differences.
- [summary.csv](summary.csv): aggregated results.
- [comparison_checks.json](comparison_checks.json): agreement of paired inputs and oracle arrays.
- [training_calculation_checks.json](training_calculation_checks.json): loss and gradient equivalence checks.

Each fitted condition uses 48 or 192 length-112 training paths, 3,000 Adam
updates, ten Euler substeps per observation, a 60-step continuation, 2,500 pilot
paths, and 8,192 paths under each of the true and fitted models. The known
diffusion is 0.42. The simultaneous 95% simulation intervals cover all 60
unique comparisons; the identical oracle comparisons are counted once.

This is a controlled synthetic experiment. It measures the effect of changing
the training variance on the fitted-model diagnostic, and does not estimate
power against alternatives or refit the SIR model.

## Reproduction

From the repository root, rebuild the main table using only the included CSVs:

```powershell
.\.venv\Scripts\python.exe reproduce.py table2
```

For score-level reanalysis, supply `output/ce_conditional_fit_pilot/` and
`output/ce_conditional_tangent_fit/` from the matching bundle, or generate them
using [section 6 of REPRODUCE.md](../../REPRODUCE.md#6-current-neural-training-comparison).
Then run:

```powershell
.\.venv\Scripts\python.exe experiments/compare_ce_training_variances.py
.\.venv\Scripts\python.exe experiments/check_ce_training_calculation.py
```

The comparison script supports both archived shared oracle files and the two
complete output directories produced by fresh runs. It also generates
`training_variance_comparison.png`; that generated figure is not included here.
