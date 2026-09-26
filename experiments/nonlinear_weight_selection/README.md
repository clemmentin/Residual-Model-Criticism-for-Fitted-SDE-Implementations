# Nonlinear weight selection

This directory reproduces the current manuscript's nonlinear coupling experiment,
Supplement Table `nonlinear-weights`, the supplementary strong-direction power table, and the independent
interval verification. The model uses two Euler substeps per observation interval:
`dX = -X dt + dW`, `dY = kappa tanh(X) dt`. Only kappa is fitted.

Run from the release root with the environment in `requirements.txt`:

```text
.\.venv\Scripts\python.exe experiments/nonlinear_weight_selection/coupling.py
.\.venv\Scripts\python.exe experiments/nonlinear_weight_selection/verify_separation.py
```

The first command uses 64 fits at each of n=64, 1024 and 16384, 262,144 pilot
paths and 32,768 evaluation paths, with seed 2026092121. It writes `coupling/`.
The second validates the first n=1024 fit from `coupling/fits.csv` using independent
seed 2026092141 and 262,144 / 2,097,152 nested paths; it writes `separation/`.
These commands regenerate the included tables. Use `--out-dir` for separate
outputs; the validation input remains `coupling/fits.csv` next to the script.
`--plot-only` redraws existing tables. `--evaluate-only` on coupling.py reuses
its pilot bounds and requires the same experiment settings.

## Files

- `coupling.py`: exact training interval, weight selection and paired evaluation.
- `verify_separation.py`: continuous-interval bounds and independent validation.
- `helpers.py`: shared simulation, quadratic-score, CDF and plotting functions.
- `coupling/`: settings, pilot bounds, 192 training fits, 1,536 evaluation rows,
  24 summary rows and numerical checks.
- `separation/`: endpoint and interval bounds, integration calculation and checks.
- `separation.md`: interpretation of the interval verification.

Supplement Table `nonlinear-weights` uses kappa=0.1, maximum `gap`, the range of `null_rejection_cv`,
and mean `weak_matched`. The supplementary strong-power table uses
`strong_matched`. Detection rates use a separate true-model null calibration.
Both coupling values and all fits are retained, including the interval missing
the truth. The rules were developed during the study; this is not a preregistered
experiment. The interval check validates one fixed training interval, not all fits.

Full derivations and assumptions are in the manuscript and supplement.
Saved metadata records the numerical settings and run environment.
