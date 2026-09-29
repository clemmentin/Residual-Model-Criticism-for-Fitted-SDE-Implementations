# Current paper reproduction protocol

Run all commands from the root of this directory. Requires Python 3.12.

Commands below use the virtual environment's Python explicitly; activation is
not required. On Linux/macOS use `.venv/bin/python` and forward slashes in paths.

For the nonlinear and neural comparison tables, install the dependencies and run
`.\.venv\Scripts\python.exe reproduce.py tables`. Their small per-fit numerical sources are
included in this release. The command aliases are retained: `table1` is the
nonlinear weight comparison, now in the supplement, and `table2` is the
neural conditional-score comparison in Supplementary Section S5.2. The supplementary
5,000-refit benchmark uses `.\.venv\Scripts\python.exe reproduce.py full-refit`
and requires the artifact bundle.

`.\.venv\Scripts\python.exe reproduce.py figures` produces the twelve current figures and three
retained historical comparisons. See step 4 for its inputs and the first-run CZE/GRC path replay.

For the feedback extension in Supplementary Section S3.7, run
`.\.venv\Scripts\python.exe experiments/feedback_weight_selection/summarize.py`.
It rebuilds the bounds from saved paired counts and replays the 16 final tests
at internal rank level 0.024; the overall null upper bound is 0.048453125.
The [experiment README](experiments/feedback_weight_selection/README.md)
gives the model, seeds, dependencies and commands for full simulation.

The two-parameter comparison in Supplementary Table `joint-feedback-exclusion`
uses `.\.venv\Scripts\python.exe experiments/feedback_weight_selection/joint_point_check.py`.
It rebuilds the bounds from included results for one fixed training dataset;
`--replay` regenerates the training ranks, weights and probability checks.

For a new SIR fit, `.\.venv\Scripts\python.exe reproduce.py sir-settings` prints the reported
nine-country configuration and `.\.venv\Scripts\python.exe reproduce.py sir-fit` fits it into
`output/sir_fit/`. Existing valid outputs there are loaded; a new `--out-dir`
requests a separate fit. This does not run the country audits below or replace
their retained checkpoints. `train_sde.py` remains the development-template
entry point.

**Windows:** unpack near the drive root (e.g. `C:\repro\`). Some artifact
paths in the bundle are long (`cache/summaries/random_country_selection_sensitivity/.../evaluation/<ISO>/<window>/null_metric_draws.csv`),
and under a deep working directory the 260-character `MAX_PATH` limit can
affect them.
Either keep the path short or enable long paths (`git config --system
core.longpaths true` and the Win32 `LongPathsEnabled` policy). Not an issue
on Linux/macOS.

## Key settings

`config.py` holds the development *baseline* template. The reported models
override it. The table below records the settings used by the reported runs;
the source code and this document are the authoritative description.

| Setting | Reported value | Where set | Cross-reference |
| --- | --- | --- | --- |
| Training cohort | DEU, FRA, ITA, ESP, NLD, BEL, AUT, CHE, GBR (nine) | `DEVELOPMENT_COUNTRIES`, `_config()` in `experiments/run_sir_sde_native_country_audit.py` | Supplement, "Fitted implementation" (12,661 length-60 sequences) |
| Held-out descriptive analyses | FIN, NOR, SWE (never trained on; windows selected using outcomes) | `CONFIRMATORY_COUNTRIES` (historical variable name) in the same runner | Supplement, "Score choice and analysis history"; detailed results archived |
| `VAL_COUNTRY` (`config.py`) | template only; GBR is a *training* country in the reported fit | n/a | Validation pass is diagnostic; no early stopping / best-checkpoint selection |
| Registered random-draw countries | LTU, MDA, SVN | `experiments/run_registered_random_country_replication.py` | NumPy `Generator(PCG64)`, draw seed 20260813, no substitution |
| First fixed-calendar holdout | CZE | `experiments/run_fixed_calendar_holdout.py` | NumPy `Generator(PCG64)`, draw seed 20260903; 2021-01-01--2021-03-02; 2,500 pilot and 2,500 reference paths |
| Second fixed-calendar holdout | GRC | `experiments/run_second_fixed_calendar_holdout.py` | Same window and score; CZE excluded; draw seed 2026090302; reference-bank base seed 2026090302001 |
| Reconstruction horizon | 28 days (`recov28_frzgamma`); `config.py` default is 14 | `CASES` in `experiments/run_sir_bi_bootstrap.py` | Supplement, "Reconstruction on the recorded scale" |
| gamma | frozen (`FREEZE_GAMMA = True`) | `CASES["recov28_frzgamma"]` | Supplement, "Fitted implementation" |
| Solver substeps | 10 projected Euler substeps/day, `SDE_SUBSTEP_DT = 0.1`, `FAST_EULER_LOSS = False` | `config.py` and `run_sir_sde_native_country_audit.py` | Supplement, "Fitted implementation" |
| Optimiser | 60 fixed epochs, batch 512, AdamW wd 1e-4, grad-clip 1; cosine LR starts at 5e-3, last applied LR is about 5.1834e-4 | `config.py`; `sir_training.py`; details below | Supplement, "Training uses 60 fixed epochs..." |
| Training-data seed | `SEED = 42` | `config.py` | Supplement, training protocol |
| Full-refit seed | `20260710`; 5,000 outer replicates, 500 pilot paths, 999 evaluation paths | `experiments/bi_synthetic_full_refit_internal_pvalue_calibration.py` | Supplement, `internal-p-full` |
| Incremental benchmark seed | `20260716`; 200 null and 200 alternative refits per scenario | `experiments/bi_synthetic_incremental_power_benchmark.py` | Archived benchmark |
| Coordinate-ablation seeds | `20260828`--`20260833`, one per condition/strength row | the six commands below | Supplement, coordinate/calibration tables |
| Evaluation-bank seeds | draw 20260813, Nordic bank 20260814, geometry-precision FIN/NOR/SWE 20260720/21/22, synthetic geometry master 2026082807 | the corresponding runner defaults and commands below | see `docs/REPRODUCTION_MAP.md` |

For the reported 12,661 training sequences, each epoch uses 24 full batches
of 512 and drops the remaining 373 sequences after shuffling. This gives
1,440 optimizer updates. The cosine schedule uses the ceiling batch count
(25 per epoch), so its `5e-4` endpoint is at update 1,500; the last applied
learning rate is approximately `5.1834e-4`. These are the retained training
settings.

## 1. Install and check the source tree

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m compileall -q experiments tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test*.py" -q
```

These are ordinary installation, syntax, and unit-test checks for a clean run.

## 2. Supply the matching artifacts

The [data and model deposit](https://doi.org/10.5281/zenodo.22212995)
currently has restricted file access. Once you have the matching artifact
bundle, unpack these directories at the
repository root:

```text
cache/
models/
```

The small neural comparison tables under `output/` are included in this source
release. CZE/GRC path files under `output/sir_figure_data/residuals/`
are optional: step 4 reconstructs them if missing. Fresh neural score-level
reanalysis requires both full neural output directories (or the refits below).

If the retrospective 2026-08-31 window-sensitivity result is cited, the bundle
must also include
`cache/summaries/random_country_selection_sensitivity/fixed_calendar_and_neighbor_windows/`.

The fixed-calendar CZE result is retained under
`cache/summaries/fixed_calendar_holdout/causal_window/`.
The second fixed-calendar GRC result is retained under
`cache/summaries/fixed_calendar_holdout/second_holdout/`.

The source repository excludes these large or separately licensed artifacts.
Table aggregation and the feedback examples can run without them; SIR analyses
and the complete figure build require the bundle.

## 3. Reproduce the numerical results

The current tables, feedback example and diagnostics are listed in
docs/REPRODUCTION_MAP.md. The supplement retains the full-refit benchmark,
coordinate/calibration study and short country/window comparisons.
The detailed reconstruction-memory audit and historical figure galleries
remain available below as research records.

Use the retained CSV/NPZ results for ordinary figure and table reproduction.
The runners below regenerate the same analyses when a fresh simulation is
needed; they perform only the input, parameter, and output checks required for
the calculation itself.

The retained draw-level exports can be regenerated with the following runners
when the matching bundle is available. The country runner writes
the paper-facing `country_cv_bi_bootstrap_full_*.csv` aliases:

```powershell
.\.venv\Scripts\python.exe experiments\run_country_cv_obsscale_audit.py bootstrap-smoke --n-bootstrap 500 --write-full-aliases
.\.venv\Scripts\python.exe experiments\run_sir_two_layer_bi_audit.py baseline_frzgamma baseline_zwd1_frzgamma recov28_frzgamma --n-bootstrap 500 --out-dir cache\summaries\two_layer_bi_centering_compare
.\.venv\Scripts\python.exe experiments\summarize_two_layer_bi_null_split.py --dir cache\summaries\two_layer_bi_centering_compare
.\.venv\Scripts\python.exe experiments\reproduce_random_country_5000_reference_draws.py  # exploratory BGR/HUN/SVK batch
```

### Full-refit calibration (`internal-p-full`)

For ordinary reproduction, rebuild the tables from the retained global ECDF:

```powershell
.\.venv\Scripts\python.exe experiments\summarize_full_refit_calibration.py
```

This uses all 5,000 saved global p-values. The six component KS Monte Carlo
p-values are recomputed from their retained KS distances, sample sizes and
rank grids; component rejection rates and moments remain the retained
summaries because the individual component p-values are unavailable.
Both KS calculations compare integer CDF-difference numerators, preserving
exact ties. They use 100,000 simulations, global seed 20260711, and component
seeds `20260711 + 50000 + 1009*(j+1)` in the script's fixed component order.
The corrected global KS p-value is 0.4909. This reporting step does not rerun
model fitting or reconstruct unavailable per-replicate component results.

The default settings are the reported complete fit--select--test design. The
explicit command below writes the reported `_final` output directory:

```powershell
.\.venv\Scripts\python.exe experiments\bi_synthetic_full_refit_internal_pvalue_calibration.py --outer-replicates 5000 --train-paths 32 --selection-paths 8 --steps 96 --pilot-paths 500 --evaluation-paths 999 --epochs 100 --seed 20260710 --out-dir cache\summaries\full_refit_internal_pvalue_calibration_r5000_p500_e999_final
```

This is an expensive refit experiment. For ordinary reproduction, use the
retained result directory from the matching bundle rather than rerunning 5,000
fits.
When resuming a new full-refit run, keep the original `--seed`; changing it
requires a new `--out-dir` or explicit `--overwrite`.

### Supporting comparisons and earlier analyses

The commands below reproduce supporting experiments and earlier analyses.
The current manuscript summaries are identified in docs/REPRODUCTION_MAP.md.
Runners use the matching bundle where required; deterministic reconstruction
audits need only `cache/owid_covid_data.csv`.

```powershell
.\.venv\Scripts\python.exe experiments\bi_synthetic_incremental_power_benchmark.py --seed 20260716 --out-dir cache\summaries\synthetic_incremental_power_split_r200  # supplementary benchmark
.\.venv\Scripts\python.exe experiments\run_coordinate_ablation.py --condition-tag cond1_lo --kappa 0.5 --extra-sigma1 0.020 --scale-factor 1.06 --seed 20260828 --out-dir cache\summaries\coordinate_ablation  # coordinate-ablation tables; repeat for all six rows below
.\.venv\Scripts\python.exe experiments\run_coordinate_ablation.py --condition-tag cond1_hi --kappa 0.5 --extra-sigma1 0.030 --scale-factor 1.08 --seed 20260829 --out-dir cache\summaries\coordinate_ablation
.\.venv\Scripts\python.exe experiments\run_coordinate_ablation.py --condition-tag cond2_lo --kappa 2.0 --extra-sigma1 0.120 --scale-factor 1.05 --seed 20260830 --out-dir cache\summaries\coordinate_ablation
.\.venv\Scripts\python.exe experiments\run_coordinate_ablation.py --condition-tag cond2_hi --kappa 2.0 --extra-sigma1 0.160 --scale-factor 1.10 --seed 20260831 --out-dir cache\summaries\coordinate_ablation
.\.venv\Scripts\python.exe experiments\run_coordinate_ablation.py --condition-tag cond3_lo --kappa 2.0 --nonlinear-kappa --extra-sigma1 0.050 --scale-factor 1.05 --seed 20260832 --out-dir cache\summaries\coordinate_ablation
.\.venv\Scripts\python.exe experiments\run_coordinate_ablation.py --condition-tag cond3_hi --kappa 2.0 --nonlinear-kappa --extra-sigma1 0.060 --scale-factor 1.10 --seed 20260833 --out-dir cache\summaries\coordinate_ablation
.\.venv\Scripts\python.exe experiments\audit_sir_gamma_reconstruction.py --output tmp\gamma_reconstruction.json  # gamma-reconstruction table
.\.venv\Scripts\python.exe experiments\bi_synthetic_solver_calibration.py   # solver-sensitivity check
.\.venv\Scripts\python.exe experiments\run_sir_tangent_map_comparison.py    # direct one-day sim / positive-part branch / geometry precision
.\.venv\Scripts\python.exe experiments\audit_sir_te7_deletion.py --output tmp\te7_deletion.json  # lag-7 component-deletion check
.\.venv\Scripts\python.exe experiments\run_sir_sde_native_short_window_audit.py --phase freeze  # freeze the 30-increment schedule
.\.venv\Scripts\python.exe experiments\run_sir_sde_native_short_window_audit.py --phase evaluate  # evaluate the frozen schedule
```

The SDE-native country and temporal extensions use the same explicit phase
pattern when regenerated from a fresh artifact root:

```powershell
.\.venv\Scripts\python.exe experiments\run_sir_sde_native_country_audit.py --phase develop
.\.venv\Scripts\python.exe experiments\run_sir_sde_native_country_audit.py --phase evaluate
.\.venv\Scripts\python.exe experiments\run_sir_sde_native_temporal_audit.py --phase freeze
.\.venv\Scripts\python.exe experiments\run_sir_sde_native_temporal_audit.py --phase evaluate
.\.venv\Scripts\python.exe experiments\run_sir_geometry_aware_country_bi.py --val FIN NOR SWE --n-bootstrap 500 --seed 20260706
.\.venv\Scripts\python.exe experiments\run_sir_geometry_precision_country_bi.py --val FIN NOR SWE --evaluation-paths 5000
```

### Derived SIR reporting tables

These reporting-layer commands consume the cached diagnostic inputs; they do
not refit models. The matching bundle already retains the paper-facing tables.
It also includes `bi_bootstrap/sir_bi_bootstrap_draws.csv` and the five
`summary_<experiment>.csv` forecast inputs under `cache/summaries/`.
Missing or non-finite forecast inputs stop the table builder before it writes.
The SWD runner uses seven checkpoints: baseline and recov28 with trainable or
frozen gamma, frozen-gamma zwd0p1 and zwd1, and recov28_dyn14. All seven are
included under the bundle's `models/` directory.
Their archived validation-loss columns use the one-day Euler approximation,
explicitly fixed in `_validation_nll`; the SWD forecast paths still use the
projected substep simulator. This historical reporting convention does not
change the corrected model's matched-substep training and SDE-native audit.

```powershell
.\.venv\Scripts\python.exe experiments\run_sir_swd_kernel_diagnostic.py
.\.venv\Scripts\python.exe experiments\run_sir_trec_gamma_decoupling.py
.\.venv\Scripts\python.exe experiments\build_sir_bi_audit_scorecard.py
.\.venv\Scripts\python.exe experiments\build_sir_bi_global_fitted_null_score.py
.\.venv\Scripts\python.exe experiments\build_sir_crossfit_main_calibration.py
.\.venv\Scripts\python.exe experiments\build_country_bi_evidence_tables.py --input-root cache\summaries\corrected_country_bi\fixednorm_matchedsolver --artifact-prefix corrected --output-stem corrected --case recov28_frzgamma --countries FIN NOR SWE  # corrected component-rank table
```

To regenerate the main BI draw file instead of using the bundle's copy:

```powershell
.\.venv\Scripts\python.exe experiments\run_sir_bi_bootstrap.py baseline_frzgamma recov28_frzgamma baseline_zwd1_frzgamma --n-bootstrap 500
```

The coordinate-ablation intermediates (reference banks, per-step covariances,
per-replicate fits) were not retained; the released exports and the six
commands above reproduce the supplementary coordinate/calibration comparison.

### Summarize the coordinate-ablation tables

After supplying the six trial CSVs and their metadata, or running the six
commands above, rebuild the strength-specific rates, pooled null rates,
paired power differences and their uncertainty with:

```powershell
.\.venv\Scripts\python.exe experiments\summarize_coordinate_ablation.py
```

The script reads `cache/summaries/coordinate_ablation/` and writes
`output/tables/coordinate_ablation.json`. `strengths` supplies the
`coordinate-ablation-strengths` table; `pooled_null_calibration` and
`pooled_power_comparisons` supply Table `coordinate-ablation` and its
reported tests. Multiply probabilities and differences by 100 for the
displayed percentages and percentage points.

The calculation uses 20,000 paired bootstrap resamples within each strength,
seed 2026082803, exact McNemar tests with Holm adjustment over six power
comparisons, and a separate Holm family of nine exact null-rate tests.
The original A/C null-grid KS calculations are also retained in the output:
they occur between settings in the shared random stream, which preserves
the reported bootstrap intervals. No fitting is performed. Use `--input-dir`
and `--output` for a separate set of trial results and summary.

### Regenerate the corrected country-BI table inputs

For the archived table `corrected-full`, use the corrected checkpoints, all
three countries, 500 paths per country and seed 20260706. Run the following
in a fresh artifact root when regenerating the retained inputs:

```powershell
foreach ($country in @('FIN', 'NOR', 'SWE')) {
    .\.venv\Scripts\python.exe experiments\run_confirmatory_country_bi.py --val $country --case recov28_frzgamma --n-bootstrap 500 --seed 20260706 --model-root models\country_cv\corrected_fixednorm_matchedsolver --out-dir cache\summaries\corrected_country_bi\fixednorm_matchedsolver --artifact-prefix corrected --analysis-role posthoc_corrected_reanalysis
}
.\.venv\Scripts\python.exe experiments\build_country_bi_evidence_tables.py --input-root cache\summaries\corrected_country_bi\fixednorm_matchedsolver --artifact-prefix corrected --output-stem corrected --case recov28_frzgamma --countries FIN NOR SWE
```

The runner's historical defaults point to the earlier `confirmatory` analysis;
the explicit overrides above select the corrected models and the filenames
expected by the table builder. On Linux/macOS, run the first Python command
once for each `--val FIN`, `--val NOR` and `--val SWE`, with the same remaining
arguments. These runs use the retained plug-in residual map; the paired
tangent-map calculation gives the same displayed component ranks, as stated
in the table caption. The paired map comparison is reproduced with
`experiments/run_sir_tangent_map_comparison.py` in the commands above.

### New random-country window-sensitivity result

This retrospective result tests three neighboring windows and three common
calendar starts for each of LTU, MDA, and SVN. It generated 18 new fitted-null
banks; all 18 global ranks are below 0.05. The retained design, summaries, and
window-level CSV files are under
`cache/summaries/random_country_selection_sensitivity/`.

To regenerate it from the matching bundle:

```powershell
.\.venv\Scripts\python.exe experiments\run_random_country_selection_sensitivity.py --phase freeze
.\.venv\Scripts\python.exe experiments\run_random_country_selection_sensitivity.py --phase evaluate
```

To reproduce the country-eligibility and window-placement audit retained in the research records (the 22-country pool
reconstruction and the window-shift table), from an OWID snapshot placed at
`cache/owid_covid_data.csv`:

```powershell
.\.venv\Scripts\python.exe experiments\audit_country_selection_rules.py --json docs\results\country_selection_audit.json
```

For a fresh registered LTU/MDA/SVN protocol root, the country draw runner
must be executed in its recorded phase order:

```powershell
.\.venv\Scripts\python.exe experiments\run_registered_random_country_replication.py --phase draw
.\.venv\Scripts\python.exe experiments\run_registered_random_country_replication.py --phase evaluate
```

### Fixed-calendar CZE and GRC holdouts

The fixed-calendar result uses a data-only 16-country European pool, one target
draw, and a common calendar interval fixed before the selected continuation is
scored.  The retained result is \(S_{\mathrm{CE}}=2.7789\) with global reference
rank \(0.06118\); the global decision is do not reject at 0.05, with
`bracket_I_energy_acf1` as the dominant component.

The second holdout reuses that window and analysis, excludes CZE as a prior
target, and draws once from the remaining eligible pool. It selected GRC and
gave \(S_{\mathrm{CE}}=5.9226\), above all 2,500 reference scores, for rank
\(1/2501\). Both selected countries are reported regardless of outcome.

To regenerate the selection and then the 5,000-path bank:

Use these commands from a fresh artifact root; when the retained files are
already present, use them directly instead of overwriting them.

```powershell
.\.venv\Scripts\python.exe experiments\run_fixed_calendar_holdout.py --phase freeze
.\.venv\Scripts\python.exe experiments\run_fixed_calendar_holdout.py --phase evaluate
.\.venv\Scripts\python.exe experiments\run_second_fixed_calendar_holdout.py --phase freeze
.\.venv\Scripts\python.exe experiments\run_second_fixed_calendar_holdout.py --phase evaluate
```

The exact candidate pool, selected target, window, seeds and score settings are
in `cache/summaries/fixed_calendar_holdout/causal_window/holdout_freeze.json`.

## 4. Reproduce the twelve cited figures

```powershell
.\.venv\Scripts\python.exe reproduce.py figures
```

Inputs are the matching `cache/` and `models/` bundle, the included nonlinear
`coupling/` and `separation/` tables,
`output/neural_training_comparison/combined_results.csv`, and
`experiments/feedback_weight_selection/results/fit00/` through `fit15/`.
The twelve cited figures comprise five in the main paper and seven in the
supplement. Eleven are generated from numerical inputs; the conceptual
Figure 2 is copied from `assets/score_comparisons.pdf` without simulation.
The command also retains three earlier comparisons:
`paper1_geometry_protocol_comparison`, `paper1_nordic_sde_native_audit` and
`paper1_random_country_replication`.
The command reconstructs window-energy ratios from the archived component
summaries. If CZE/GRC residual files are missing, it runs
`experiments/replay_fixed_calendar_residual_paths.py`, replaying the recorded
5,000-path banks and comparing all metrics and ranks against retained results.
It does not refit the epidemic model.

Outputs go to `output/figures/`. The submission figure filenames are listed
in `docs/REPRODUCTION_MAP.md`.
The individual plotting commands are:

```powershell
.\.venv\Scripts\python.exe experiments/replay_fixed_calendar_residual_paths.py
.\.venv\Scripts\python.exe experiments/make_paper1_core_figures.py --out-dir output/figures
.\.venv\Scripts\python.exe experiments/make_manuscript_figures.py
```

The last command writes `output/figures/`; `--additions-only`
rebuilds the neural, MDA-dependence, nonlinear-weight and feedback-weight
figures. The older
core builder also writes historical figures not cited by the current manuscript.
To regenerate the optional finite-step and linear simulation caches first:

```powershell
.\.venv\Scripts\python.exe experiments/finite_step_geometry_synthetic.py --out-dir cache/summaries/finite_step_geometry_synthetic
.\.venv\Scripts\python.exe experiments/run_linear_geometry_mechanism.py run
```

The exact current figure/table mapping is in `docs/REPRODUCTION_MAP.md`.

## 5. Current nonlinear weight-selection experiment

The cited directory is
`experiments/nonlinear_weight_selection/`.
Its scripts and small numerical outputs are included. Supplement Table `nonlinear-weights` uses
`coupling/results.csv`, grouping 64 fits at each training size and method,
with `null_rejection_cv` for conditional rejection and `weak_matched` for
true-null-calibrated power. The supplementary strong-power table uses
`strong_matched`. Both coupling values and all fits are retained.

To rerun the reported comparison (262,144 pilot paths, 32,768 evaluation paths,
64 fits, seed 2026092121), then the independent interval verification
(262,144 and 2,097,152 nested paths, seed 2026092141):

```powershell
.\.venv\Scripts\python.exe experiments/nonlinear_weight_selection/coupling.py
.\.venv\Scripts\python.exe experiments/nonlinear_weight_selection/verify_separation.py
```

These full commands regenerate the results in the cited directory. For a
separate trial use `--out-dir output/nonlinear_trial`; the validation script
still deliberately reads the published first n=1024 fit from `coupling/fits.csv`.
Changing sample counts creates a different experiment, not the paper result.
See the included README and `separation.md` for the exact estimators and bounds.

## 6. Current neural training comparison

Supplement Table `ce-conditional` and Supplement Figure `ce-training-variance` use the included
`output/neural_training_comparison/combined_results.csv`.
To recompute this file from per-path scores, supply the two archived directories
`output/ce_conditional_fit_pilot/` and
`output/ce_conditional_tangent_fit/`, or generate them with:

```powershell
.\.venv\Scripts\python.exe experiments/run_ce_conditional_fit_pilot.py --train-variance plugin --training-implementation standard --out-dir output/ce_conditional_fit_pilot
.\.venv\Scripts\python.exe experiments/run_ce_conditional_fit_pilot.py --train-variance tangent --training-implementation batched --out-dir output/ce_conditional_tangent_fit
.\.venv\Scripts\python.exe experiments/compare_ce_training_variances.py
```

Run fresh fits into empty directories. After an interrupted run, repeat the same
command with `--resume`: it checks the saved settings, keeps complete replicate
groups, and reruns incomplete groups with their original seeds. Both
arms use seed 2026090907, 12 training groups, 48/192 training paths, 112 steps,
3,000 Adam updates, 10 Euler substeps, a 60-step continuation, 2,500 pilot paths,
and 8,192 paths for each conditional score distribution. The reference size
is 2,500. Each fit is assessed using tangent-standardized residuals, regardless
of its training variance. These are controlled synthetic fits, not SIR refits.

For the numerical equivalence check of standard and batched losses/gradients,
run `.\.venv\Scripts\python.exe experiments/check_ce_training_calculation.py` after supplying the
plug-in output. `.\.venv\Scripts\python.exe experiments/run_ce_conditional_fit_pilot.py --check-only
--out-dir output/ce_checks` checks the feature and rank calculations without
running the full fits.

## Numerical checks

The unit tests cover model normalization, solver settings, country/window
scoring, table aggregation and the retained feedback calculations. Run the
unit-test command in section 1 after installation.

The feedback summary replays the 16 final tests from their recorded seeds;
the joint-parameter command recomputes five exclusion bounds. The coordinate
summary recomputes paired intervals and multiplicity adjustments from retained
trial results. The CZE/GRC figure replay compares regenerated residual metrics
and ranks with the archived results before writing the paths.

Table aggregation does not refit models. The full-refit component p-values and
the coordinate-ablation intermediate arrays were not retained; the available
summaries and the commands for new runs are described above. Same-seed
agreement checks reproducibility, not robustness to new samples.
