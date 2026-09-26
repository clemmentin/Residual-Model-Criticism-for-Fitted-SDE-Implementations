"""Reproduce the manuscript's 16 complete repetitions with overall level below 5%."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

import four_step as model
from support import cp_bounds, write_csv

HERE = Path(__file__).resolve().parent
REPETITIONS, SIMULATIONS, CELLS = 16, 2048, 16
ALPHA = .024


def count_limits(count, total, tail):
    count = np.asarray(count, dtype=int)
    lower = np.where(count == 0, 0., beta.ppf(tail, count, total-count+1))
    upper = np.where(count == total, 1., beta.ppf(1-tail, count+1, total-count))
    return lower, upper


def summarize(source, out):
    rows = []
    for index in range(REPETITIONS):
        folder = source / f"fit{index:02d}"
        settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        final = json.loads((folder / "final_test.json").read_text(encoding="utf-8"))
        expected = dict(seed=976000001+1000*index, training=1024,
                        training_simulations=2047, interval_tolerance=.001,
                        pilot=SIMULATIONS, cells=CELLS, evaluation=0, marginal_test=True,
                        state_form="direct", true_kappa=1., horizon=20,
                        substeps=4, alpha=ALPHA, references=499, eta=.0025,
                        bound_method="extrema",
                        epsilon=.02, required_power=.12, weak_loading_ratio=.2,
                        candidate_order=list(model.NAMES))
        for key, value in expected.items():
            if settings[key] != value:
                raise ValueError(f"{folder.name}: unexpected {key}")
        prep = settings["preparation"]
        if prep["training_failure_bound"] != 2/1024:
            raise ValueError(f"{folder.name}: unexpected training coverage")

        with (folder / "cells.csv").open(encoding="utf-8", newline="") as stream:
            cells = list(csv.DictReader(stream))
        with (folder / "bounds.csv").open(encoding="utf-8", newline="") as stream:
            saved = {r["method"]: r for r in csv.DictReader(stream)}
        if len(cells) != CELLS*len(model.NAMES):
            raise ValueError(f"{folder.name}: incomplete parameter cover")
        # The three extrema are simultaneous across candidates. Cellwise
        # intervals themselves are not simultaneous; see the supplement.
        tail = .0025/(6*len(model.NAMES))
        limits = {}
        for label in ("lower", "upper", "power"):
            n10 = [int(r[label+"_n10"]) for r in cells]
            n01 = [int(r[label+"_n01"]) for r in cells]
            l10, u10 = count_limits(n10, SIMULATIONS, tail)
            l01, u01 = count_limits(n01, SIMULATIONS, tail)
            limits[label] = (l10-u01, u10-l01)
            for bound, values in zip(("low", "high"), limits[label]):
                original = [float(r[label+"_"+bound]) for r in cells]
                if not np.allclose(values, original, rtol=1e-8, atol=1e-10):
                    raise ValueError(f"{folder.name}: paired probability bounds differ")

        selected = "none"
        chosen = None
        for name in model.NAMES:
            take = np.array([r["method"] == name for r in cells])
            if take.sum() != CELLS:
                raise ValueError(f"{folder.name}: missing cells for {name}")
            group = [r for r in cells if r["method"] == name]
            if {int(r["cell"]) for r in group} != set(range(CELLS)):
                raise ValueError(f"{folder.name}: repeated or missing cell indices")
            edges = np.array([[float(r["lower_kappa"]), float(r["upper_kappa"])]
                              for r in sorted(group, key=lambda r: int(r["cell"]))])
            if (not np.allclose(edges[:-1, 1], edges[1:, 0], atol=1e-12, rtol=0)
                    or edges[0, 0] > prep["outer_intervals"][0][0]
                    or edges[-1, 1] < prep["outer_intervals"][-1][1]):
                raise ValueError(f"{folder.name}: parameter cells do not cover training set")
            error = max(0., -limits["lower"][0][take].min(), limits["upper"][1][take].max())
            power = max(0., ALPHA+limits["power"][0][take].min())
            if not np.allclose([error, power],
                               [float(saved[name]["calibration_bound"]),
                                float(saved[name]["power_lower"])],
                               rtol=1e-8, atol=1e-10):
                raise ValueError(f"{folder.name}: candidate bounds differ")
            if selected == "none" and error <= .02 and power >= .12:
                selected, chosen = name, (error, power)
        if selected != settings["selected"] or selected != final["selected"]:
            raise ValueError(f"{folder.name}: selected candidate differs")
        replay = model.one_final_test(prep["fitted_kappa"], prep["weights"],
                                      selected, settings["seed"]+80, alpha=ALPHA)
        if replay != final:
            raise ValueError(f"{folder.name}: final ranks differ on replay")
        rows.append(dict(index=index, seed=settings["seed"], selected=selected,
                         fitted_kappa=prep["fitted_kappa"],
                         calibration_bound=chosen[0] if chosen else "",
                         power_lower=chosen[1] if chosen else "",
                         null_p=final["null_p"], power_p=final["power_p"],
                         null_reject=final["null_reject"],
                         power_reject=final["power_reject"]))
    metrics = []
    for name, count in (("selection", sum(r["selected"] != "none" for r in rows)),
                        ("null_rejection", sum(r["null_reject"] for r in rows)),
                        ("alternative_rejection", sum(r["power_reject"] for r in rows))):
        lower, upper = cp_bounds(count, REPETITIONS, .025)
        metrics.append(dict(metric=name, count=count, repetitions=REPETITIONS,
                            estimate=count/REPETITIONS, cp_95_lower=lower, cp_95_upper=upper))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "fits.csv", rows)
    write_csv(out / "summary.csv", metrics)
    print(json.dumps(dict(metrics=metrics, marginal_null_upper=ALPHA+.02+2/1024+.0025,
                          output=str(out)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "results")
    parser.add_argument("--out-dir", type=Path,
                        default=HERE.parents[1] / "output/feedback_weight_selection")
    args = parser.parse_args()
    summarize(args.source, args.out_dir)
