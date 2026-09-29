# Current figure and table reproduction map

This map follows the figures and tables in the main paper and supplement.
Run commands from the release root. `.\.venv\Scripts\python.exe reproduce.py tables` rebuilds
the nonlinear and neural comparison tables from included per-fit results.
`.\.venv\Scripts\python.exe reproduce.py figures`
generates eleven numerical figures and copies the supplied conceptual diagram,
covering all twelve cited figures, using the matching `cache/` and `models/`
bundle where needed. It also retains three historical comparisons.

## Figures

The current figure builder is `experiments/make_manuscript_figures.py`.
It reuses the core and linear builders and the SIR data readers. The core
builder is `experiments/make_paper1_core_figures.py`. The single entry above
calls both and writes PDFs and PNGs to `output/figures/`.

| Manuscript label | PDF file | Generator |
| --- | --- | --- |
| Main `audit-motivation` | `01_main_motivation.pdf` | current figure builder |
| Main `score-comparisons` | `score_comparisons.pdf` | supplied vector diagram in `assets/`; copied by current figure builder |
| Main `finite-grid-motivation` | `02_main_geometry.pdf` | current figure builder |
| Supplement `linear-coordinate-sensitivity` | `03_main_sensitivity.pdf` | current figure builder |
| Supplement `nonlinear-weights` | `10_main_weight_selection.pdf` | current figure builder |
| Main `sir-fixed-calendar-paths` | `05_fixed_calendar.pdf` | current figure builder |
| Main `sir-mda-dependence` | `09_main_mda_dependence.pdf` | current figure builder |
| Supplement `feedback-weight-selection` | `11_supp_feedback_selection.pdf` | current figure builder |
| Supplement `ce-training-variance` | `08_main_neural_matching.pdf` | current figure builder |
| Supplement `sir-country-patterns` | `06_country_energy.pdf` | current figure builder |
| Supplement `sir-window-patterns` | `07_window_energy.pdf`, panels (a)--(c) | current figure builder; TeX crops the lower panels |
| Supplement `nordic-window-sensitivity` | `paper1_nordic_window_sensitivity.pdf` | core builder, styled by current figure builder |

Submission filenames differ from the descriptive plotting filenames:

| Plot output | Submission file | Location |
| --- | --- | --- |
| `01_main_motivation.pdf` | `main_figure_1.pdf` | Main Figure 1 |
| `score_comparisons.pdf` | `score_comparisons.pdf` | Main Figure 2 |
| `02_main_geometry.pdf` | `main_figure_2.pdf` | Main Figure 3 |
| `05_fixed_calendar.pdf` | `main_figure_5.pdf` | Main Figure 4 |
| `09_main_mda_dependence.pdf` | `main_figure_6.pdf` | Main Figure 5 |
| `11_supp_feedback_selection.pdf` | `supp_figure_S1.pdf` | Supplement |
| `03_main_sensitivity.pdf` | `main_figure_3.pdf` | Supplement |
| `08_main_neural_matching.pdf` | `supp_figure_S2.pdf` | Supplement |
| `10_main_weight_selection.pdf` | `main_figure_4.pdf` | Supplement |
| `06_country_energy.pdf` | `supp_figure_S3.pdf` | Supplement |
| `07_window_energy.pdf` | `supp_figure_S4.pdf` | Supplement; crop as specified above |
| `paper1_nordic_window_sensitivity.pdf` | `supp_figure_S5.pdf` | Supplement |

The neural figure reads the included
`output/neural_training_comparison/combined_results.csv`.
The nonlinear figure reads `coupling/results.csv`, `separation/point_bounds.csv`
and `separation/bound_refinement.csv` in the manuscript-cited nonlinear study.
The feedback figure reads `fit00/bounds.csv` through `fit15/bounds.csv`
in `experiments/feedback_weight_selection/results/`.
CZE/GRC residual and log-I paths are replayed from retained fits and seeds
when absent; all other application figures use archived component summaries
and the six-country residual cache. Window ratios are recomputed from those
summaries, so no separate intermediate CSV is required.

## Tables

| Manuscript label | Numerical source / command |
| --- | --- |
| Supplement `nonlinear-weights` | `.\.venv\Scripts\python.exe reproduce.py table1`; aggregates `experiments/nonlinear_weight_selection/coupling/results.csv` at kappa=0.1. |
| Supplement `ce-conditional` | `.\.venv\Scripts\python.exe reproduce.py table2`; aggregates `output/neural_training_comparison/combined_results.csv`. |
| Supplement `internal-p-full` | `.\.venv\Scripts\python.exe reproduce.py full-refit`; summarizes 5,000 retained p-values from the matching artifact bundle. |
| Supplement `coordinate-ablation-strengths` and `coordinate-ablation` | `.\.venv\Scripts\python.exe experiments/summarize_coordinate_ablation.py`; uses the six trial CSVs and metadata in `cache/summaries/coordinate_ablation/`. |
| Supplement `nonlinear-strong-power` | `experiments/nonlinear_weight_selection/coupling/results.csv`; strong-direction power, averaged over 64 fits. |
| Supplementary Section S3.7 feedback extension | `.\.venv\Scripts\python.exe experiments/feedback_weight_selection/summarize.py`; recomputes bounds and final ranks from the 16 included fits with overall level below 5%. Full simulation commands are in that directory's README. |
| Supplement `joint-feedback-exclusion` | `.\.venv\Scripts\python.exe experiments/feedback_weight_selection/joint_point_check.py`; recomputes five exclusions from the retained training ranks and two independent probability checks. |
| Supplement `fixed-calendar-holdout-protocol` | Settings and results in `cache/summaries/fixed_calendar_holdout/`; `run_fixed_calendar_holdout.py` and `run_second_fixed_calendar_holdout.py`, each `--phase freeze`, then `--phase evaluate`. |
| Supplement `sir-analysis-status` | Protocol descriptions and retained outputs; `experiments/build_sir_bi_audit_scorecard.py`. |
| Supplement `sir-country-summary` and window summaries | `global_rank_pvalue` from the saved country/window score CSVs listed below; see the country runners and `run_random_country_selection_sensitivity.py` in `REPRODUCE.md`. |

For `sir-country-summary`, read each country's `layer_and_global_scores.csv`:
BGR/HUN/SVK are under `cache/summaries/random_country_5000_reference_draws/`,
and LTU/MDA/SVN are under
`cache/summaries/sde_native_random_country_replication/random_country_batch2_frozen/evaluation/`.
The additional-window summaries read `window_scores.csv` under
`cache/summaries/random_country_selection_sensitivity/fixed_calendar_and_neighbor_windows/evaluation/`;
the figure builder also writes the energy ratios to `output/sir_figure_data/window_energy_ratios.csv`.

The tables are typeset directly in TeX. Aggregated CSV probabilities must be
converted to percentages where the caption says percent. Full commands for
rerunning the nonlinear simulations and neural fits are in sections 5–6 of
`REPRODUCE.md`. Smaller smoke runs verify execution only.

## Archived analyses

The supplement retains the supporting experiments above, the country energy
paths, the window energy comparison and the Nordic window-length comparison.
The extra geometry figure,
reconstruction-memory table, detailed Nordic result histories, separate
generator diagnostic and remaining country/window figures are outside the
current manuscript. Their existing scripts,
saved outputs and commands in REPRODUCE.md remain available as research records.
