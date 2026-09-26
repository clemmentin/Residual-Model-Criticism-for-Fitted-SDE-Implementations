"""Reproduce the two-parameter accepted-point comparison.

Numerical functions are retained from the research scripts. This checks one
fixed training set and five prepared weights; it is not a full joint selector.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import four_step as old

HERE = Path(__file__).resolve().parent
DOMAIN = ((.2, 3.), (.5, 1.5))
NAMES = old.NAMES
tanh, square = old.loop.tanh, old.loop.square


def first_observation(noise, kappa, sigma):
    x, y = 0.*kappa+0.*sigma, 0.*kappa+0.*sigma
    for z in noise:
        tx, ty = tanh(x), tanh(y)
        x, y = (old.A*x+old.C*ty+.5*sigma*z,
                old.B*y+old.H*kappa*tx)
    return x, y


def moments(noise, kappa, sigma):
    x, y = first_observation(noise, kappa, sigma)
    return square(x), square(y)


def averages(noise, kappa, sigma):
    x2, y2 = moments(noise, kappa, sigma)
    return x2.mean(axis=-1), y2.mean(axis=-1)


def fit_moments(noise, observed, domain=DOMAIN):
    scale = np.maximum(np.asarray(observed), 1e-8)

    def criterion(theta):
        expected = np.asarray(averages(noise, theta[0], theta[1])).mean(axis=1)
        return float(np.sum(((expected-np.asarray(observed))/scale)**2))

    center = [(lo+hi)/2 for lo, hi in domain]
    result = minimize(criterion, center, bounds=domain, method="L-BFGS-B",
                      options={"ftol": 1e-12})
    return result.x.tolist(), float(result.fun), bool(result.success)


def components(noise, kappa, sigma):
    x, y = 0.*kappa+0.*sigma, 0.*kappa+0.*sigma
    x_square, y_quadratic, y_linear = 0.*x, 0.*x, 0.*x
    for shocks in noise:
        start_x, start_y = x, y
        for z in shocks:
            tx, ty = np.tanh(x), np.tanh(y)
            x, y = (old.A*x+old.C*ty+.5*sigma*z,
                    old.B*y+old.H*kappa*tx)
        response = y-old.B**4*start_y
        x_square = x_square+x*x
        y_quadratic = y_quadratic+response*response
        y_linear = y_linear+np.tanh(start_x)*response
    return (x_square.mean(axis=-1)/len(noise),
            y_quadratic.mean(axis=-1), y_linear.mean(axis=-1))


def observed_statistics(observed_components, kappa):
    x, y2, xy = observed_components
    return np.asarray((x, y2-.6*kappa*xy))


def simulated_statistics(noise, kappa, sigma):
    x, y2, xy = components(noise, kappa, sigma)
    return x, y2-.6*kappa*xy


def score_noise(noise, kappa, sigma, fitted, weights, weak=0.):
    fitted_k, _ = fitted
    x, y = 0.*kappa+0.*sigma, 0.*kappa+0.*sigma
    totals = [0.*x for _ in weights]
    for shocks in noise:
        mx, my = x, y
        actual_x, actual_y = x, y
        ex, ey = 0.*x, 0.*x
        for zx, zy in shocks:
            tx, ty = old.loop.tanh(mx), old.loop.tanh(my)
            atx, aty = old.loop.tanh(actual_x), old.loop.tanh(actual_y)
            next_ex = old.A*ex+old.C*(aty-ty)+.5*sigma*zx
            next_ey = (old.B*ey+old.H*(kappa-fitted_k)*atx+
                       old.H*fitted_k*(atx-tx)+weak*.5*kappa*zy)
            actual_x, actual_y = (old.A*actual_x+old.C*aty+.5*sigma*zx,
                                  old.B*actual_y+old.H*kappa*atx+weak*.5*kappa*zy)
            mx, my = old.A*mx+old.C*ty, old.B*my+old.H*fitted_k*tx
            ex, ey = next_ex, next_ey
        for i, (q, slope, variance) in enumerate(weights):
            strong = 0. if q is None else old.loop.square(ex)/q
            totals[i] = totals[i]+strong+old.loop.square(ey-slope*ex)/variance
        x, y = actual_x, actual_y
    return [value/len(noise) for value in totals]


def prepare_weights(fitted, seed):
    kappa, sigma = fitted
    z = old.noise_sample(np.random.default_rng(seed), 1024)
    history, _ = old.loop.paths(z[:, :, 0], kappa, sigma=sigma)
    covariance = np.zeros((2, 2))
    for step in range(20):
        start = history[:, step]
        _, _, cov, _ = old.loop.predicted(
            start, np.zeros_like(start), kappa, substeps=4, sigma=sigma)
        covariance += cov.mean(axis=0)/20
    q = covariance[0, 0]
    slope = covariance[0, 1]/q
    variance = covariance[1, 1]-covariance[0, 1]**2/q
    qraw = sigma*sigma*old.H*sum(old.A**(2*j) for j in range(4))
    weights = [(qraw, 0., qraw)]
    weights.extend((q, slope, variance+kappa*kappa*sigma*sigma*ell)
                   for ell in old.RIDGES)
    return weights, covariance


def thresholds(fitted, weights, count, seed, alpha=.024):
    rank = int(np.floor(alpha*500))
    if not 1 <= rank <= 499:
        raise ValueError("Unattainable rank level.")
    rng, parts = np.random.default_rng(seed), [[] for _ in weights]
    for start in range(0, count, 32):
        size = min(32, count-start)
        z = old.noise_sample(rng, size*499)
        scores = score_noise(z, *fitted, fitted, weights)
        for dest, values in zip(parts, scores):
            dest.append(np.partition(values.reshape(size, 499), 499-rank, axis=1)
                        [:, 499-rank])
    return [np.concatenate(part) for part in parts]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def count_from_rate(rate, trials):
    value = float(rate)*trials
    count = round(value)
    if not 0 <= count <= trials or abs(value-count) > 1e-8:
        raise ValueError("A retained rejection rate does not give an integer count.")
    return count


def summarize(source, out):
    training = read_json(source/"training_settings.json")
    points = read_csv(source/"training_points.csv")
    settings = read_json(source/"point_settings.json")
    rates = read_csv(source/"point_rates.csv")
    targeted = read_json(source/"targeted_settings.json")
    if (settings["fitted"] != training["fitted"]
            or targeted["fitted"] != settings["fitted"]):
        raise ValueError("The comparisons must use the same fit.")
    for _, kappa, sigma in settings["points"]:
        row = next(row for row in points
                   if float(row["kappa"]) == kappa and float(row["sigma"]) == sigma)
        if not (row["accepted"] == "True"
                and 2 <= int(row["x_rank"]) <= 2047
                and 2 <= int(row["y_rank"]) <= 2047):
            raise ValueError("A displayed parameter is not in the joint rank set.")
    if targeted["point"] != settings["points"][0]:
        raise ValueError("The second check must use the same high-sigma point.")

    # All original bounds are recomputed, including unused checks in the family.
    for row in rates:
        for kind in ("null", "power"):
            count = count_from_rate(row[kind+"_estimate"], settings["pilot"])
            bounds = old.loop.cp_bounds(count, settings["pilot"], settings["per_tail"])
            saved = [float(row[kind+"_lower"]), float(row[kind+"_upper"])]
            if not np.allclose(bounds, saved, rtol=1e-9, atol=1e-12):
                raise ValueError("A saved probability interval differs on recomputation.")

    result = []
    for name in NAMES:
        if name in targeted["candidates"]:
            row = next(row for row in targeted["results"] if row["method"] == name)
            trials, tail = targeted["pilot"], targeted["per_comparison_tail"]
            count = count_from_rate(row["estimate"], trials)
            bound = old.loop.cp_bounds(count, trials, tail)[0]
            if not np.isclose(bound, row["one_sided_lower"], rtol=1e-9, atol=1e-12):
                raise ValueError("A targeted lower bound differs on recomputation.")
            point, kind, side, threshold = "high_sigma", "null rejection", "lower", .044
            excluded = bound > threshold
        else:
            row = next(row for row in rates
                       if row["point"] == "low_sigma" and row["method"] == name)
            trials, tail = settings["pilot"], settings["per_tail"]
            count = count_from_rate(row["power_estimate"], trials)
            bound = old.loop.cp_bounds(count, trials, tail)[1]
            point, kind, side, threshold = "low_sigma", "power", "upper", .12
            excluded = bound < threshold
        result.append(dict(method=name, point=point, probability=kind,
                           successes=count, trials=trials, tail=tail,
                           bound_side=side, bound=bound,
                           requirement=threshold, excluded=excluded))
    failure = settings["eta"]+targeted["eta"]
    if not all(row["excluded"] for row in result):
        raise ValueError("The retained evidence does not exclude every candidate.")
    out.mkdir(parents=True, exist_ok=True)
    old.loop.write_csv(out/"joint_exclusion.csv", result)
    print(json.dumps(dict(fitted=training["fitted"], bounds=result,
                          conditional_simulation_confidence=1-failure), indent=2))
    return result


def replay_training(source):
    settings = read_json(source/"training_settings.json")
    points = read_csv(source/"training_points.csv")
    seed, n, b = settings["seed"], settings["training"], settings["simulations"]
    obs_noise = np.random.default_rng(seed+10).standard_normal((4, 4, n))
    sim_noise = np.random.default_rng(seed+11).standard_normal((4, 4, b, n))
    observed = components(obs_noise, 1., 1.)
    fitted, _, _ = fit_moments(sim_noise[0], averages(obs_noise[0], 1., 1.))
    if not np.allclose(fitted, settings["fitted"], rtol=0, atol=2e-6):
        raise ValueError("The fitted moments differ on replay.")
    for row in points:
        kappa, sigma = float(row["kappa"]), float(row["sigma"])
        simulated = simulated_statistics(sim_noise, kappa, sigma)
        target = observed_statistics(observed, kappa)
        ranks = [1+int(np.sum(values < actual))
                 for values, actual in zip(simulated, target)]
        if ranks != [int(row["x_rank"]), int(row["y_rank"])]:
            raise ValueError("A training rank differs on replay.")
    settings = read_json(source/"point_settings.json")
    weights, covariance = prepare_weights(settings["fitted"], settings["prepare_seed"])
    if not np.allclose(weights, settings["weights"], rtol=1e-9, atol=1e-12):
        raise ValueError("Prepared weights differ on replay.")
    if not np.allclose(covariance, settings["covariance"], rtol=1e-9, atol=1e-12):
        raise ValueError("Prepared covariance differs on replay.")
    print("Training ranks, fitted moments and prepared weights reproduced.", flush=True)


def replay_probabilities(source):
    settings = read_json(source/"point_settings.json")
    rates = read_csv(source/"point_rates.csv")
    fitted, weights = settings["fitted"], settings["weights"]
    count, seed = settings["pilot"], settings["seed"]
    cutoff = thresholds(fitted, weights, count, seed, settings["alpha"])
    noise = old.noise_sample(np.random.default_rng(seed+1), count)
    for label, kappa, sigma in settings["points"]:
        for kind, weak in (("null", 0.), ("power", .2)):
            scores = score_noise(noise, kappa, sigma, fitted, weights, weak)
            for name, values, threshold in zip(NAMES, scores, cutoff):
                row = next(row for row in rates if row["point"] == label and row["method"] == name)
                if int(np.sum(values > threshold)) != count_from_rate(row[kind+"_estimate"], count):
                    raise ValueError("An original pointwise count differs on replay.")
    targeted = read_json(source/"targeted_settings.json")
    names = targeted["candidates"]
    selected_weights = [weights[NAMES.index(name)] for name in names]
    count, seed = targeted["pilot"], targeted["seed"]
    cutoff = thresholds(fitted, selected_weights, count, seed, settings["alpha"])
    noise = old.noise_sample(np.random.default_rng(seed+1), count)
    _, kappa, sigma = targeted["point"]
    scores = score_noise(noise, kappa, sigma, fitted, selected_weights)
    for row, values, threshold in zip(targeted["results"], scores, cutoff):
        if int(np.sum(values > threshold)) != count_from_rate(row["estimate"], count):
            raise ValueError("A targeted count differs on replay.")
    print("Both independent probability checks reproduced.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE/"joint_results")
    parser.add_argument("--out-dir", type=Path,
                        default=HERE.parents[1]/"output/feedback_weight_selection")
    parser.add_argument("--replay", action="store_true",
                        help="Regenerate training ranks, weights and both probability checks.")
    args = parser.parse_args()
    summarize(args.source, args.out_dir)
    if args.replay:
        replay_training(args.source)
        replay_probabilities(args.source)
