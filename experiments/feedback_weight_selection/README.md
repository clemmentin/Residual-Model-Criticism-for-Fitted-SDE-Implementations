# Weight selection with nonlinear feedback

The main experiment is the four-substep example in Section 4.3 and Supplementary
Section "Weight selection with feedback". Only kappa is unknown. Training follows the
model; the specified departure adds independent Y noise to the future path.
Every final reference path comes from the same fitted model used for selection.

The 16 complete repetitions were fixed before generating this batch. Their
seeds are `976000001 + 1000*i`, for `i=0,...,15`. Each repetition includes new
training, simulated training inversion, preparation of weights, selection and
one final test under each of the null and departure. No repetition is omitted.

## Reproduce the reported results

From the release root, run:

```powershell
.\.venv\Scripts\python.exe experiments/feedback_weight_selection/summarize.py
```

This recomputes the paired binomial bounds from saved counts, checks selection,
replays the final tests and writes `fits.csv` and `summary.csv` under
`output/feedback_weight_selection/`. It does not rerun fitting or selection.
The reported counts are 16 selections, 1 null rejection and 4 departure
rejections. Each displayed 95% binomial interval is a separate statement.
The internal rank level is 0.024. The theoretical overall null upper bound
is 0.024 + 0.02 + 2/1024 + 0.0025 = 0.048453125, below 5%.
The empirical null frequency does not establish this guarantee.

`results/fit00/` through `fit15/` contain the original settings, training-set
calculation, per-cell counts, candidate bounds, final ranks and numerical
checks. Output paths in the settings are relative to the repository root.

## Check the numerical calculations

```powershell
.\.venv\Scripts\python.exe experiments/feedback_weight_selection/four_step.py --check-only --out-dir output/feedback_weight_selection/check
```

The checks compare the score with a separate literal Euler calculation and
check training and score interval inclusion on finite grids. The supplement
gives the continuous inclusion argument in real arithmetic. The double
precision interval code uses outward rounding and numerical guards; it is
not a formally certified interval-arithmetic implementation.

## Repeat the experiment

One original complete repetition is:

```powershell
.\.venv\Scripts\python.exe experiments/feedback_weight_selection/four_step.py --training 1024 --training-simulations 2047 --interval-tolerance .001 --pilot 2048 --cells 16 --evaluation 0 --marginal-test --alpha .024 --bound-method extrema --state-form direct --seed 976000001 --out-dir output/feedback_weight_selection/rerun/fit00
```

For the other repetitions, change the seed and output directory using the
formula above. Each fit uses 2,047 simulated training datasets of size 1,024
and 1,021,952 Q reference paths during selection. The summary command is the
short route to the published numbers; full simulation is substantially more
expensive. Pass `--source output/feedback_weight_selection/rerun` to summarize
a completed rerun of all 16 fits.

The fixed 16-cell cover and five candidate weights use `extrema` bounds:
each binomial tail has error 0.0025/(6*5). The guarantee covers the three
final extrema for all candidates; individual cell intervals are not
simultaneous. The supplement gives the short argument at a fixed worst
cell. Settings were chosen using an earlier fit before this batch's training
data were generated. Earlier exploratory batches are not pooled with these
16 repetitions.

The numerical functions are copied from the research implementation, with
imports collected in `support.py`. The saved runs used Python 3.13.6,
NumPy 2.4.2 and SciPy 1.17.0. Only NumPy and SciPy are needed for this example.

## When kappa and sigma are both unknown

Supplementary Table `joint-feedback-exclusion` reports a separate, retrospective
comparison for one training dataset of 1,024 paths with four observations each.
It uses the same feedback model with X noise loading sigma, parameters in
[0.2, 3] x [0.5, 1.5], and generating point (1, 1). The internal rank level
is 0.024, the error tolerance is 0.02 and required departure power is 0.12.
At two parameters accepted by the joint training ranks, independent probability
bounds exclude all five prepared weights. This is an exclusion for this
training set and candidate class, not a general impossibility or a successful
two-parameter selector.

Recompute the table from the included small result files:

```powershell
.\.venv\Scripts\python.exe experiments/feedback_weight_selection/joint_point_check.py
```

The command checks membership of both points, recovers integer rejection
counts from the saved rates and sample sizes, and recomputes the binomial
bounds. It writes `output/feedback_weight_selection/joint_exclusion.csv`.
The first probability check covers five weights, two accepted points and
null/departure rejection probabilities. Each binomial tail gets 0.0025/40.
The independent targeted check covers two lower bounds, each with tail
error 0.0025/2. Together their five reported exclusions
have at least 99.5% simulation confidence conditional on training and weights.

The training, weight-preparation and two probability-check seeds are
989000001, 990000021, 990000001 and 991000001, respectively. Observed and
simulated training inputs use training-seed offsets 10 and 11; future target
noise uses checking-seed offset 1. Candidate parameters share each stage's
Gaussian inputs. The scripts retain the complete original stream layout.

To regenerate the fitted moments, all saved training ranks, prepared weights
and both probability checks, add `--replay`. This uses the original seeds
and about 24.5 million fitted reference paths. Preparation uses 1,024
independent fitted paths. The first check uses 16,384 independent groups of
499 references; the second uses 32,768 groups. The files under
`joint_results/` retain the exact parameter values, settings and rates.
The numerical functions are copied from the corresponding research scripts;
the table calculation needs only NumPy and SciPy.
