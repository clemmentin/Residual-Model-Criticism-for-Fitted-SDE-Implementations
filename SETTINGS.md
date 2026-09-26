# Key settings and the `config.py` caveat

`config.py` is the development baseline template, so it is not edited to
carry the paper-specific split. Read this file first.

To avoid editing or combining settings manually, run:

```text
.\.venv\Scripts\python.exe reproduce.py sir-settings
.\.venv\Scripts\python.exe reproduce.py sir-fit
```

The first command prints all resolved fields; the second fits with those
fields and writes new checkpoints under `output/sir_fit/`. Both reuse
`run_sir_sde_native_country_audit._config()` with GBR as a diagnostic
validation country inside the nine-country training cohort. The fit does
not select a checkpoint using validation scores. See the README for the
current table/figure entry points.

## `config.py` holds the development *baseline* template, not the paper's settings

The `Config` dataclass defaults describe the `baseline` case only. The
reported models are produced by layering named overrides on top of those
defaults, so **do not read the training/test split from `config.py`**:

| `config.py` default | Reported runs | Where the override lives |
| --- | --- | --- |
| `TRAIN_COUNTRIES` = 8 countries (no GBR) | 9 countries **including GBR** | `DEVELOPMENT_COUNTRIES` in `experiments/run_sir_sde_native_country_audit.py` |
| `VAL_COUNTRY = "GBR"` | GBR is a **training** country; FIN/NOR/SWE are held out for descriptive analyses; LTU/MDA/SVN are the registered random draw, also analysed descriptively | `_config()` in the same runner; `REPRODUCE.md` |
| `SIR_RECOVERY_DAYS = 14.0` | `28.0` (`recov28_frzgamma`) | `CASES` in `experiments/run_sir_bi_bootstrap.py` |

`_bi_audit_common.apply_config_updates()` applies these by mutating a fresh
`Config()`; the defaults are the starting point, not the final per-run
configuration. "Validation" in the baseline is diagnostic only: training runs
a fixed `NUM_EPOCHS` with no early stopping or best-checkpoint selection.

## Authoritative sources, in order

1. `REPRODUCE.md` -> **"Key settings"** — the run-by-run table (seeds,
   substeps, cohorts, optimiser) with manuscript cross-references.
2. The analysis runners and their explicit command-line defaults.
3. The Supplementary Material, "Fitted implementation".

## What matches the manuscript directly

* Solver: 10 projected Euler substeps/day, `SDE_SUBSTEP_DT = 0.1`,
  `FAST_EULER_LOSS = False`; asserted at load in
  `experiments/run_sir_sde_native_country_audit.py`.
* Optimiser: 60 fixed epochs, batch 512, AdamW weight decay `1e-4`,
  gradient clipping at 1. The cosine schedule starts at `5e-3` and has a
  `5e-4` endpoint at update 1,500. With 12,661 training sequences, each epoch
  uses 24 full batches and drops the remainder after shuffling: 1,440 updates
  in total, with a last applied learning rate of approximately `5.1834e-4`.
* Training-data seed `SEED = 42`, set in `config.py`.
* Evaluation-bank seeds: draw `20260813`, Nordic bank `20260814`,
  geometry-precision FIN/NOR/SWE `20260720/21/22`, synthetic master
  `2026082807` — set in the matching analysis runners.
