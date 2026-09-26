"""Four-substep feedback model with simulation-inverted training uncertainty.

The training statistic is the mean squared first Y observation. Its finite-n
distribution is simulated, with no Gaussian pivot or transition density.
Confidence-set computation retains all unresolved parameter cells.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support as loop
import numpy as np
from scipy.optimize import brentq

H = .25
A, B, C = .75, .875, .125
NAMES = ("raw", "pooled", "pooled_ridge_0.01", "pooled_ridge_0.03", "pooled_ridge_0.1")
RIDGES = (0., .01, .03, .1)


def training_y(noise, kappa):
    x, y = 0.*kappa, 0.*kappa
    for z in noise:
        x, y = A*x+C*loop.tanh(y)+.5*z, B*y+H*kappa*loop.tanh(x)
    return y


def prepare_training(noise):
    """Unroll the same four updates once; this is not a distributional formula."""
    x1 = .5*noise[0]
    x2 = A*x1+.5*noise[1]
    t1, t2 = np.tanh(x1), np.tanh(x2)
    return B*B*t1+B*t2, A*x2+.5*noise[2], t1


def prepared_training_y(prepared, kappa):
    fixed, base, t1 = prepared
    return H*kappa*(fixed+loop.tanh(base+C*loop.tanh(H*kappa*t1)))


def training_stat(noise, kappa):
    y = training_y(noise, kappa)
    return np.mean(y*y, axis=-1)


def prepared_training_stat(prepared, kappa):
    y = prepared_training_y(prepared, kappa)
    return np.mean(y*y, axis=-1)


def training_bounds(noise, lo, hi, prepared=None):
    prepared = prepare_training(noise) if prepared is None else prepared
    y = prepared_training_y(prepared, loop.Interval(lo, hi, 1.)).square()
    value_lo, value_hi = y.lo.mean(axis=-1), y.hi.mean(axis=-1)
    gradient = np.maximum(abs(y.dl.mean(axis=-1)), abs(y.dh.mean(axis=-1)))
    center = prepared_training_stat(prepared, (lo+hi)/2)
    width = (hi-lo)/2*gradient
    guard = 1e-10*(1+abs(center))
    return (np.maximum(0., np.maximum(value_lo, center-width)-guard),
            np.minimum(value_hi, center+width)+guard)


def fit_and_interval(observed_y, seed, simulations=2047, tolerance=.001, domain=(.2, 3.)):
    n = len(observed_y)
    statistic = float(np.mean(observed_y**2))
    z = np.random.default_rng(seed).standard_normal((4, simulations, n))
    prepared = prepare_training(z)
    gamma = .0025
    j = int(np.floor(gamma/2*(simulations+1)))
    if j < 1:
        raise ValueError("Insufficient simulation ranks for the requested training coverage.")
    def objective(kappa):
        return float(np.median(prepared_training_stat(prepared, kappa))-statistic)
    f0, f1 = objective(domain[0]), objective(domain[1])
    if f0*f1 <= 0:
        fitted = float(brentq(objective, *domain, xtol=1e-7))
    else:
        fitted = domain[0] if abs(f0) <= abs(f1) else domain[1]
    pending, kept, rows = [domain], [], []
    started = time.perf_counter()
    while pending:
        lo, hi = pending.pop()
        low, high = training_bounds(z, lo, hi, prepared=prepared)
        # Order statistics preserve coordinatewise inequalities.
        lower_possible = float(np.partition(low, j-1)[j-1])
        lower_sufficient = float(np.partition(high, j-1)[j-1])
        upper_possible = float(np.partition(high, simulations-j)[simulations-j])
        upper_sufficient = float(np.partition(low, simulations-j)[simulations-j])
        rejected = statistic < lower_possible or statistic > upper_possible
        accepted = lower_sufficient <= statistic <= upper_sufficient
        if rejected:
            decision = "reject"
        elif accepted or hi-lo <= tolerance:
            decision = "accept" if accepted else "retain_unresolved"
            kept.append((lo, hi))
        else:
            decision = "split"
            middle = (lo+hi)/2
            pending.extend(((middle, hi), (lo, middle)))
        rows.append(dict(lower=lo, upper=hi, statistic=statistic,
                         lower_quantile_lower=lower_possible,
                         upper_quantile_upper=upper_possible, decision=decision))
        if len(rows) % 8 == 0:
            print(f"Training inversion: {len(rows)} cells, {len(pending)} pending, "
                  f"{time.perf_counter()-started:.1f}s", flush=True)
    merged = []
    for lo, hi in sorted(kept):
        if merged and lo <= merged[-1][1]+1e-14:
            merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))
    metadata = dict(training=n, statistic=statistic, fitted_kappa=fitted,
                    simulations=simulations, tail_rank=j,
                    training_failure_bound=2*j/(simulations+1),
                    parameter_domain=list(domain), outer_intervals=merged,
                    cells_evaluated=len(rows), tolerance=tolerance,
                    inversion_seconds=time.perf_counter()-started)
    return fitted, merged, rows, metadata


def noise_sample(rng, count, horizon=20):
    return rng.standard_normal((horizon, 4, 2, count))


def score_noise(noise, kappa, fitted, weights, weak=0., state_form="direct"):
    """Fixed matrices: keep the same fitted center on target and Q paths."""
    x, y = 0.*kappa, 0.*kappa
    totals = [0.*kappa for _ in weights]
    for shocks in noise:
        mx, my = x, y
        actual_x, actual_y = x, y
        ex, ey = 0.*kappa, 0.*kappa
        for zx, zy in shocks:
            tx, ty = loop.tanh(mx), loop.tanh(my)
            if state_form == "prediction_residual":
                actual_x, actual_y = mx+ex, my+ey
            actual_tx, actual_ty = loop.tanh(actual_x), loop.tanh(actual_y)
            nex = A*ex+C*(actual_ty-ty)+.5*zx
            ney = B*ey+H*(kappa-fitted)*actual_tx
            ney = ney+H*fitted*(actual_tx-tx)+weak*.5*kappa*zy
            if state_form == "direct":
                actual_x, actual_y = (A*actual_x+C*actual_ty+.5*zx,
                                      B*actual_y+H*kappa*actual_tx+weak*.5*kappa*zy)
            mx, my = A*mx+C*ty, B*my+H*fitted*tx
            ex, ey = nex, ney
        for i, (q, slope, variance) in enumerate(weights):
            strong = 0. if q is None else loop.square(ex)/q
            totals[i] = totals[i]+strong+loop.square(ey-slope*ex)/variance
        x, y = (actual_x, actual_y) if state_form == "direct" else (mx+ex, my+ey)
    return [score/len(noise) for score in totals]


def prepare_weights(fitted, seed):
    z = noise_sample(np.random.default_rng(seed), 1024)
    history, _ = loop.paths(z[:, :, 0], fitted)
    covariance = np.zeros((2, 2))
    for k in range(20):
        start = history[:, k]
        _, _, cov, _ = loop.predicted(start, np.zeros_like(start), fitted, substeps=4)
        covariance += cov.mean(axis=0)/20
    q = covariance[0, 0]
    slope = covariance[0, 1]/q
    variance = covariance[1, 1]-covariance[0, 1]**2/q
    qraw = H*sum(A**(2*j) for j in range(4))
    weights = [(qraw, 0., qraw)]
    weights.extend((q, slope, variance+fitted*fitted*ell) for ell in RIDGES)
    return weights, covariance


def envelopes(noise, fitted, weights, lo, hi, weak=0., state_form="direct"):
    interval = score_noise(noise, loop.Interval(lo, hi, 1.), fitted, weights, weak, state_form)
    center = score_noise(noise, (lo+hi)/2, fitted, weights, weak, state_form)
    result = []
    for box, value in zip(interval, center):
        width = (hi-lo)/2*np.maximum(abs(box.dl), abs(box.dh))
        guard = 1e-10*(1+abs(value))
        result.append((np.maximum(0., np.maximum(box.lo, value-width)-guard),
                       np.minimum(box.hi, value+width)+guard))
    return result


def thresholds(fitted, weights, count, seed, generating=None, alpha=.05):
    generating = fitted if generating is None else generating
    rank = int(np.floor(alpha*500))
    if not 1 <= rank <= 499:
        raise ValueError("The rank level must be attainable with 499 references.")
    rng, parts = np.random.default_rng(seed), [[] for _ in weights]
    for start in range(0, count, 32):
        size = min(32, count-start)
        scores = score_noise(noise_sample(rng, size*499), generating, fitted, weights)
        for destination, values in zip(parts, scores):
            destination.append(np.partition(values.reshape(size, 499), 499-rank, axis=1)[:, 499-rank])
    return [np.concatenate(p) for p in parts]


def certify(fitted, weights, intervals, seed, count, cells, out, state_form="direct", names=None,
            selection="first", power_intervals=None, bound_method="all_cells", alpha=.05):
    names = NAMES if names is None else tuple(names)
    if len(names) != len(weights):
        raise ValueError("Every candidate weight needs a name.")
    if selection not in ("first", "max_power"):
        raise ValueError("Unknown selection rule.")
    if bound_method not in ("all_cells", "extrema"):
        raise ValueError("Unknown probability bound method.")
    # A hull is an explicit outer set; it does not discard any accepted component.
    lower, upper = intervals[0][0], intervals[-1][1]
    edges = np.linspace(lower, upper, cells+1)
    power_edges = (edges if power_intervals is None else
                   np.linspace(power_intervals[0][0], power_intervals[-1][1], cells+1))
    cutoff = thresholds(fitted, weights, count, seed, alpha=alpha)
    z = noise_sample(np.random.default_rng(seed+1), count)
    matched = score_noise(z, fitted, fitted, weights)
    baseline = [s > t for s, t in zip(matched, cutoff)]
    # extrema controls the three final extrema for every candidate, as proved
    # in the supplement. Its cellwise intervals are not simultaneous.
    tail = (.0025/(6*len(weights)) if bound_method == "extrema" else
            .0025/(12*len(weights)*cells))
    rows = []
    started = time.perf_counter()
    for cell, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        null = envelopes(z, fitted, weights, lo, hi, state_form=state_form)
        alt = envelopes(z, fitted, weights, power_edges[cell], power_edges[cell+1],
                        weak=.2, state_form=state_form)
        for name, limits, alimits, origin, threshold in zip(names, null, alt, baseline, cutoff):
            row = dict(method=name, cell=cell, lower_kappa=lo, upper_kappa=hi,
                       power_lower_kappa=power_edges[cell], power_upper_kappa=power_edges[cell+1])
            for label, score in (("lower", limits[0]), ("upper", limits[1]), ("power", alimits[0])):
                low, high, n10, n01 = loop.paired_difference(score > threshold, origin, tail)
                row.update({label+"_low":low, label+"_high":high,
                            label+"_n10":n10, label+"_n01":n01})
            row["envelope_width_mean"] = float(np.mean(limits[1]-limits[0]))
            rows.append(row)
        if (cell+1) % 8 == 0:
            print(f"Four-step envelope {cell+1}/{cells}, {time.perf_counter()-started:.1f}s",
                  flush=True)
    summary, selected = [], "none"
    for name in names:
        group = [row for row in rows if row["method"] == name]
        lo = min(row["lower_low"] for row in group)
        hi = max(row["upper_high"] for row in group)
        error = max(0., -lo, hi)
        power = max(0., alpha+min(row["power_low"] for row in group))
        passed = error <= .02 and power >= .12
        if passed and selected == "none":
            selected = name
        summary.append(dict(method=name, calibration_bound=error, power_lower=power,
                            rejection_lower=max(0., alpha+lo), rejection_upper=min(1., alpha+hi),
                            passes=passed))
    if selection == "max_power":
        eligible = [row for row in summary if row["passes"]]
        selected = max(eligible, key=lambda row: row["power_lower"])["method"] if eligible else "none"
    loop.write_csv(out/"cells.csv", rows)
    loop.write_csv(out/"bounds.csv", summary)
    return selected


def evaluate(fitted, weights, seed, count, selected, names=None, alpha=.05):
    names = NAMES if names is None else tuple(names)
    if len(names) != len(weights):
        raise ValueError("Every candidate weight needs a name.")
    ref = thresholds(fitted, weights, count, seed, alpha=alpha)
    oracle = thresholds(fitted, weights, count, seed+1, generating=1., alpha=alpha)
    z = noise_sample(np.random.default_rng(seed+2), count)
    null, alt = score_noise(z, 1., fitted, weights), score_noise(z, 1., fitted, weights, weak=.2)
    rows = []
    for name, sn, sa, threshold, true_threshold in zip(names, null, alt, ref, oracle):
        row = dict(method=name, selected=name == selected)
        for label, indicator in (("null", sn > threshold), ("power", sa > threshold),
                                 ("oracle_null", sn > true_threshold),
                                 ("oracle_power", sa > true_threshold)):
            low, high = loop.cp_bounds(int(indicator.sum()), count, .025)
            row.update({label:float(indicator.mean()), label+"_low":low, label+"_high":high})
        rows.append(row)
    return rows


def one_final_test(fitted, weights, selected, seed, names=None, alpha=.05):
    """One fresh final test per complete fit, including selection failures."""
    if selected == "none":
        return dict(selected=selected, null_p=1., power_p=1.,
                    null_reject=0, power_reject=0)
    names = NAMES if names is None else tuple(names)
    chosen = [weights[names.index(selected)]]
    rng = np.random.default_rng(seed)
    reference = score_noise(noise_sample(rng, 499), fitted, fitted, chosen)[0]
    inputs = noise_sample(rng, 1)
    null = float(score_noise(inputs, 1., fitted, chosen)[0][0])
    alt = float(score_noise(inputs, 1., fitted, chosen, weak=.2)[0][0])
    null_p = (1+int(np.sum(reference >= null)))/500
    power_p = (1+int(np.sum(reference >= alt)))/500
    return dict(selected=selected, null_p=null_p, power_p=power_p,
                null_reject=int(null_p <= alpha), power_reject=int(power_p <= alpha))


def checks(out, seed):
    rng = np.random.default_rng(seed)
    z = noise_sample(rng, 19)
    fitted = 1.02
    weights, _ = prepare_weights(fitted, seed+1)
    worst, matrix_error = 0., 0.
    for weak in (0., .2):
        bound = envelopes(z, fitted, weights, .96, 1.06, weak)
        for kappa in np.linspace(.96, 1.06, 11):
            values = score_noise(z, kappa, fitted, weights, weak)
            for value, (lo, hi) in zip(values, bound):
                worst = max(worst, float(np.max(lo-value)), float(np.max(value-hi)))
        # A separate literal Euler implementation and deterministic prediction.
        x, y = np.zeros(19), np.zeros(19)
        actual = np.zeros((len(weights), 19))
        kappa = .99
        for shocks in z:
            start = np.column_stack((x, y))
            mean, _, _, _ = loop.predicted(start, np.zeros_like(start), fitted, substeps=4)
            for zx, zy in shocks:
                x, y = A*x+C*np.tanh(y)+.5*zx, B*y+H*kappa*np.tanh(x)+weak*.5*kappa*zy
            rx, ry = (np.column_stack((x, y))-mean).T
            for j, (q, slope, variance) in enumerate(weights):
                actual[j] += (rx*rx/q+(ry-slope*rx)**2/variance)/20
        expected = score_noise(z, kappa, fitted, weights, weak)
        matrix_error = max(matrix_error, float(np.max(np.abs(actual-np.array(expected)))))
    training_noise = rng.standard_normal((4, 17, 16))
    lo, hi = training_bounds(training_noise, .8, 1.2)
    training_violation = 0.
    unrolling_error = 0.
    for kappa in np.linspace(.8, 1.2, 31):
        value = training_stat(training_noise, kappa)
        optimized = prepared_training_stat(prepare_training(training_noise), kappa)
        unrolling_error = max(unrolling_error, float(np.max(abs(value-optimized))))
        training_violation = max(training_violation, float(np.max(lo-value)), float(np.max(value-hi)))
    assert max(worst, training_violation) <= 0 and max(matrix_error, unrolling_error) < 1e-10
    result = dict(score_envelope_violation=worst, training_envelope_violation=training_violation,
                  literal_euler_score_error=matrix_error, training_unrolling_error=unrolling_error)
    (out/"checks.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=loop.HERE/"four_step_pilot")
    parser.add_argument("--training", type=int, default=1024)
    parser.add_argument("--training-simulations", type=int, default=2047)
    parser.add_argument("--interval-tolerance", type=float, default=.001)
    parser.add_argument("--pilot", type=int, default=2048)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--evaluation", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=976000001)
    parser.add_argument("--alpha", type=float, default=.024)
    parser.add_argument("--bound-method", choices=("all_cells", "extrema"),
                        default="extrema")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--marginal-test", action="store_true")
    parser.add_argument("--reuse-training", type=Path)
    parser.add_argument("--state-form", choices=("direct", "prediction_residual"), default="direct")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(checks(args.out_dir, args.seed)), flush=True)
    if args.check_only:
        return
    started = time.perf_counter()
    saved_preparation = None
    if args.reuse_training:
        metadata = json.loads((args.reuse_training/"training.json").read_text(encoding="utf-8"))
        saved = json.loads((args.reuse_training/"settings.json").read_text(encoding="utf-8"))
        if metadata["training"] != args.training or metadata["simulations"] != args.training_simulations:
            raise ValueError("The reused training design does not match the requested design.")
        fitted, intervals = metadata["fitted_kappa"], metadata["outer_intervals"]
        with (args.reuse_training/"training_cells.csv").open(encoding="utf-8") as stream:
            cells = list(csv.DictReader(stream))
        saved_preparation = saved.get("preparation")
        metadata["reused_from"] = str(args.reuse_training)
    else:
        observed_y = training_y(np.random.default_rng(args.seed+10).standard_normal((4, args.training)), 1.)
        fitted, intervals, cells, metadata = fit_and_interval(
            observed_y, args.seed+11, args.training_simulations, args.interval_tolerance)
    metadata["covers"] = any(lo <= 1. <= hi for lo, hi in intervals)
    loop.write_csv(args.out_dir/"training_cells.csv", cells)
    (args.out_dir/"training.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    selected = "none"
    if not args.training_only:
        if saved_preparation and "weights" in saved_preparation:
            weights = saved_preparation["weights"]
            covariance = np.array(saved_preparation["pooled_covariance"])
        else:
            weights, covariance = prepare_weights(fitted, args.seed+20)
        if intervals:
            selected = certify(fitted, weights, intervals, args.seed+30, args.pilot,
                               args.cells, args.out_dir, args.state_form,
                               bound_method=args.bound_method, alpha=args.alpha)
        print(f"Selected: {selected}", flush=True)
        metadata.update(pooled_covariance=covariance.tolist(), weights=weights)
        if args.evaluation:
            rows = evaluate(fitted, weights, args.seed+40, args.evaluation, selected,
                            alpha=args.alpha)
            loop.write_csv(args.out_dir/"evaluation.csv", rows)
        if args.marginal_test:
            final = one_final_test(fitted, weights, selected, args.seed+80, alpha=args.alpha)
            (args.out_dir/"final_test.json").write_text(json.dumps(final, indent=2)+"\n",
                                                     encoding="utf-8")
    settings = vars(args).copy()
    settings.update(out_dir=str(args.out_dir),
                    reuse_training=str(args.reuse_training) if args.reuse_training else None,
                    selected=selected, true_kappa=1.,
                    horizon=20, substeps=4, alpha=args.alpha, references=499,
                    gamma=.0025, eta=.0025, epsilon=.02, required_power=.12,
                    weak_loading_ratio=.2, parameter_domain=[.2, 3.],
                    seconds=time.perf_counter()-started, preparation=metadata,
                    candidate_order=list(NAMES))
    (args.out_dir/"settings.json").write_text(json.dumps(settings, indent=2)+"\n", encoding="utf-8")
    print(f"Finished in {settings['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
