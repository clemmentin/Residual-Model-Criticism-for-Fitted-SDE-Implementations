"""Shared, experiment-agnostic helpers for fitted-null BI audit runners."""

from __future__ import annotations


import pickle

import numpy as np

import config as config_module
import sir_data

from experiments.audit_statistics import upper_rank


def apply_config_updates(updates: dict[str, object]) -> config_module.Config:
    cfg = config_module.Config()
    for key, value in updates.items():
        setattr(cfg, key, value)
    return cfg


def load_cached_data(cfg: config_module.Config) -> config_module.TrainingData:
    cache_path = config_module.get_cache_path(cfg)
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)["data"]
    return sir_data.load_or_create_training_data(cfg)


def empirical_pvalue(observed: float, null_values: np.ndarray, tail: str) -> float:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(observed) or values.size == 0:
        return float("nan")
    if tail == "upper":
        return upper_rank(observed, values)
    if tail == "centered":
        center = float(np.median(values))
        observed_distance = abs(observed - center)
        null_distances = np.abs(values - center)
        return upper_rank(observed_distance, null_distances)
    raise ValueError(f"Unknown tail mode: {tail}")


def summarize_metric(
    *,
    case_name: str,
    experiment: str,
    start_date: str,
    n_bootstrap: int,
    metric: str,
    observed: float,
    null_values: np.ndarray,
    metric_tails: dict[str, str],
) -> dict[str, object]:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    tail = metric_tails[metric]
    percentile = (
        float(np.mean(values <= observed))
        if values.size and np.isfinite(observed)
        else float("nan")
    )
    pvalue = empirical_pvalue(observed, values, tail)
    return {
        "case": case_name,
        "experiment": experiment,
        "start_date": start_date,
        "n_bootstrap": n_bootstrap,
        "metric": metric,
        "tail": tail,
        "observed": observed,
        "null_mean": float(np.mean(values)),
        "null_std": float(np.std(values, ddof=1)),
        "null_q025": float(np.quantile(values, 0.025)),
        "null_q500": float(np.quantile(values, 0.500)),
        "null_q975": float(np.quantile(values, 0.975)),
        "empirical_percentile": percentile,
        "empirical_pvalue": pvalue,
        "significant_05": bool(pvalue < 0.05),
    }
