# Data and artifact sources

## OWID COVID-19 data

The analysis uses the Our World in Data COVID-19 data pipeline. The official
download and catalog documentation is:

- [OWID COVID-19 data documentation](https://docs.owid.io/projects/etl/api/covid/)

The matching Zenodo bundle supplies the archived file at
`cache/owid_covid_data.csv`. The preprocessing code reads that file directly;
it does not download a newer copy during reproduction. Check the applicable
OWID and upstream-provider terms before redistributing the data.

The snapshot was checked against the [official archived CSV](https://raw.githubusercontent.com/owid/covid-19-data/f0d53320b19703cf565940d8269181a8423d7032/public/data/owid-covid-data.csv)
on 2026-09-23. It is byte-for-byte identical to that version, last updated
upstream on 2024-08-19: 98,391,483 bytes, 429,435 rows and 67 columns, with
recorded dates from 2020-01-01 through 2024-08-14. Its SHA-256 is
`8473d0f0fdf962e1ffbd5b85b18726fc96a49bab109e271186c339725a12b10c`.

Source agreement does not remove missing observations or reporting errors.
`sir_data._select_incidence_array` replaces missing `new_cases_smoothed`
values with zero and clips negative incidence to zero before reconstruction.
Under the reported 28-day reconstruction and active-period filter, the nine
training countries' retained daily series include 337 such missing values:
DEU 90, FRA 134, ITA 3 and ESP 110. These are country-days, not independent
training sequences, and zero filling is a preprocessing assumption rather
than evidence of zero infections. The CZE/GRC fixed-calendar windows and
their 20-state conditioning histories have no missing smoothed incidence.
The S/I inputs are reconstructed proxies, not directly measured epidemic states.

## Model checkpoints

The matching bundle supplies the fitted `.eqx` checkpoints used by the
reported analyses. The checkpoint filenames are storage labels only. The
reported SDE-native fit uses the nine development countries
`DEU, FRA, ITA, ESP, NLD, BEL, AUT, CHE, GBR`; `FIN`, `NOR`, and `SWE` are the
held-out countries used for descriptive analyses with outcome-selected windows.
The later Nordic centring--energy score was developed after inspection.
The case, solver, optimiser, and random seeds
are listed in `REPRODUCE.md` and are set in the analysis runners.

## Generated results

The matching bundle supplies retained fitted-null draws, result tables, and
expensive simulation outputs under `cache/summaries/`. The source release
contains the scripts that regenerate them and
`docs/REPRODUCTION_MAP.md` maps each computational Figure/Table to its script
and command. Figures and compiled documents are regenerated under `output/`.

The current source release also includes the small per-fit neural comparison
tables under `output/neural_training_comparison/` and the
manuscript-cited nonlinear weight study under
`experiments/nonlinear_weight_selection/`.
These reproduce the main neural comparison table and the supplementary
nonlinear comparison tables without the large artifact bundle.
The full neural per-path scores and checkpoints are separate artifacts, or
can be regenerated with the commands in `REPRODUCE.md`.

The self-contained feedback study is in `experiments/feedback_weight_selection/`.
Its `results/` directory retains all 16 repetitions' settings, counts, bounds
and final ranks; `joint_results/` contains the training ranks and probability
checks for the supplementary two-parameter exclusion table. The included
scripts rebuild the summaries and replay the final tests using only NumPy
and SciPy. Full simulation and its additional cost are described in that
directory's README.

The source-code licence applies to this repository's original code.
OWID data and any third-party material remain subject to their own terms;
consult the accompanying data-deposit metadata before reuse.
