"""Audit what the country-eligibility screen and the window rule actually removed.

Backs Supplementary Section "What the selection rules removed": rebuilds the
2026-08-12 eligibility screen from the archived OWID snapshot, and measures how
far the midpoint window rule moves when its two trajectory-dependent inputs are
perturbed.  This documents the size of the departure; it does not restore
calibration for the already-evaluated targets.

Usage:  python experiments/audit_country_selection_rules.py [--json PATH]
"""
from __future__ import annotations


import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "cache/owid_covid_data.csv"

START = pd.Timestamp("2020-03-15")      # Config.TRAINING_START_DATE
ACTIVE_THRESHOLD = 1e-5                 # Config.SIR_ACTIVE_THRESHOLD
RECOVERY_DAYS = 28.0                    # recov28_frzgamma case
LOOKBACK = 20                           # max(Config.SIGNATURE_PATH_LENGTHS)
HORIZON = 60                            # Config.FORECAST_HORIZON
MIN_POPULATION = 1_000_000

# Excluded at the 2026-08-12 screen (random_country_smoke_metadata.json).
PREVIOUSLY_USED = ("DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "CHE", "GBR",
                   "DNK", "FIN", "NOR", "SWE")
RECORDED_FIRST_DRAW_POOL = (
    "ALB", "BGR", "BIH", "BLR", "CZE", "EST", "GRC", "HRV", "HUN", "IRL", "LTU",
    "LVA", "MDA", "MKD", "POL", "PRT", "ROU", "RUS", "SRB", "SVK", "SVN", "UKR")
EVALUATED = ("FIN", "NOR", "SWE", "LTU", "MDA", "SVN")


def _load() -> pd.DataFrame:
    cols = ["iso_code", "continent", "date", "population",
            "total_cases", "new_cases_smoothed"]
    raw = pd.read_csv(SNAPSHOT, usecols=cols, parse_dates=["date"])
    return raw[(raw["continent"] == "Europe")
               & (~raw["iso_code"].str.startswith("OWID"))]


def _series(cdf: pd.DataFrame, threshold: float, recovery_days: float):
    """Active-period block and audit window for one country, per train_sde.py."""
    cdf = cdf[cdf["date"] >= START].sort_values("date")
    pop_vals = cdf["population"].dropna()
    if pop_vals.empty:
        return None
    pop = float(pop_vals.iloc[0])
    x = cdf["new_cases_smoothed"].fillna(0).clip(lower=0).values.astype(float)
    alpha = float(np.exp(-1.0 / recovery_days))
    active = np.zeros(len(x))
    for t in range(len(x)):
        active[t] = x[t] + alpha * (active[t - 1] if t > 0 else 0.0)
    on = np.where(active / pop > threshold)[0]
    if on.size == 0:
        return None
    t0, t1 = int(on[0]), int(on[-1]) + 1
    dates = pd.to_datetime(cdf["date"].values)
    if (t1 - t0) < 60:
        return None
    n_val = (t1 - t0) - (LOOKBACK - 1)
    start_idx = max(n_val // 2, LOOKBACK)
    first = t0 + LOOKBACK - 1 + start_idx
    if first + HORIZON >= len(dates):
        return None
    return {"cdf": cdf, "pop": pop, "dates": dates, "t0": t0, "t1": t1,
            "n_val": n_val, "window": (dates[first], dates[first + HORIZON])}


def screen(raw: pd.DataFrame) -> dict:
    """Rebuild the eligibility screen and attribute every exclusion to a rule."""
    entities = eligible_population = 0
    rows, pool = [], []
    for iso, cdf in raw.groupby("iso_code"):
        entities += 1
        pop_vals = cdf.loc[cdf["date"] >= START, "population"].dropna()
        if pop_vals.empty or float(pop_vals.iloc[0]) < MIN_POPULATION:
            continue
        eligible_population += 1
        if iso in PREVIOUSLY_USED:
            continue
        info = _series(cdf, ACTIVE_THRESHOLD, RECOVERY_DAYS)
        if info is None:
            rows.append({"iso": iso, "verdict": "no complete 60-increment window"})
            continue
        lo, hi = info["window"]
        src = info["cdf"].set_index("date").loc[lo:hi]
        missing_days = int((hi - lo).days) + 1 - len(src)
        missing_smoothed = int(src["new_cases_smoothed"].isna().sum())
        zero_smoothed = int((src["new_cases_smoothed"].fillna(0) == 0).sum())
        ok = not (missing_days or missing_smoothed or zero_smoothed)
        rows.append({"iso": iso, "verdict": "eligible" if ok else "fails completeness",
                     "missing_calendar_days": missing_days,
                     "missing_smoothed": missing_smoothed,
                     "zero_smoothed": zero_smoothed})
        if ok:
            pool.append(iso)
    return {
        "european_entities": entities,
        "population_at_least_1e6": eligible_population,
        "candidates_after_prior_use": len(rows),
        "reconstructed_pool": sorted(pool),
        "recorded_pool": sorted(RECORDED_FIRST_DRAW_POOL),
        "in_recorded_not_reconstructed":
            sorted(set(RECORDED_FIRST_DRAW_POOL) - set(pool)),
        "in_reconstructed_not_recorded":
            sorted(set(pool) - set(RECORDED_FIRST_DRAW_POOL)),
        "removed_by_completeness":
            [r["iso"] for r in rows if r["verdict"] == "fails completeness"],
        "per_country": rows,
    }


def window_placement(raw: pd.DataFrame) -> list:
    """Shift in the window start date under perturbations of the two inputs."""
    perturbations = [("threshold_x0.5", dict(threshold=0.5 * ACTIVE_THRESHOLD)),
                     ("threshold_x2", dict(threshold=2 * ACTIVE_THRESHOLD)),
                     ("threshold_x5", dict(threshold=5 * ACTIVE_THRESHOLD)),
                     ("memory_14d", dict(recovery_days=14.0)),
                     ("memory_56d", dict(recovery_days=56.0))]
    out = []
    for iso in EVALUATED:
        cdf = raw[raw["iso_code"] == iso]
        base = _series(cdf, ACTIVE_THRESHOLD, RECOVERY_DAYS)
        row = {"iso": iso,
               "nominal_start": str(base["window"][0].date()),
               "nominal_end": str(base["window"][1].date()),
               "series_end": str(base["dates"][base["t1"] - 1].date()),
               "record_end": str(base["dates"][-1].date()),
               "series_ends_at_record_end": base["t1"] == len(base["dates"])}
        for label, kw in perturbations:
            alt = _series(cdf, kw.get("threshold", ACTIVE_THRESHOLD),
                          kw.get("recovery_days", RECOVERY_DAYS))
            row[label] = (None if alt is None
                          else (alt["window"][0] - base["window"][0]).days)
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    raw = _load()
    report = {"snapshot": SNAPSHOT.relative_to(ROOT).as_posix(),
              "eligibility": screen(raw),
              "window_placement": window_placement(raw)}

    e = report["eligibility"]
    print(f"European entities                    : {e['european_entities']}")
    print(f"  population >= {MIN_POPULATION:,}          : {e['population_at_least_1e6']}")
    print(f"  minus previously used              : {e['candidates_after_prior_use']}")
    print(f"  removed by completeness conditions : {len(e['removed_by_completeness'])}")
    print(f"  reconstructed pool                 : {len(e['reconstructed_pool'])}")
    print(f"  vs recorded pool, either direction : "
          f"{len(e['in_recorded_not_reconstructed'])} / "
          f"{len(e['in_reconstructed_not_recorded'])}")
    print()
    print(pd.DataFrame(report["window_placement"]).to_string(index=False))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
