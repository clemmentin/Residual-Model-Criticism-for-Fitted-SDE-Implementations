"""Read retained SIR paths and build the manuscript's window-energy table."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/sir_figure_data"
SUMMARIES = ROOT / "cache/summaries"
BATCH = SUMMARIES / "sde_native_random_country_replication/random_country_batch2_frozen/evaluation"
WINDOWS = SUMMARIES / "random_country_selection_sensitivity/fixed_calendar_and_neighbor_windows/evaluation"
METRIC = "bracket_I_mean_z2"

def load_paths(country: str) -> dict:
    if country in ("CZE", "GRC"):
        folder, prefix = {
            "CZE": ("causal_window", "holdout"),
            "GRC": ("second_holdout", "second_holdout"),
        }[country]
        source = SUMMARIES / "fixed_calendar_holdout" / folder
        row = pd.read_csv(source / f"{prefix}_results.csv").iloc[0]
        draws = pd.read_csv(source / f"{prefix}_reference_draws.csv")
        metrics = {k.removeprefix("observed_"): float(v)
                   for k, v in row.items() if k.startswith("observed_")}
        score = {
            "global_rank_pvalue": float(row.global_rank_pvalue),
            "martingale_rank_pvalue": float(row.martingale_rank_pvalue),
        }
        start, end = row.evaluation_start, row.evaluation_end
        cache = OUT / "residuals" / f"{country}_residual_paths.npz"
    else:
        source = BATCH / country
        meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        metrics, score = meta["observed_metrics"], meta["score"]
        start, end = meta["start_date"], meta["end_date"]
        draws = pd.read_csv(source / "null_metric_draws.csv")
        cache = SUMMARIES / "six_country_path_visualization" / f"{country}_residual_paths.npz"
    with np.load(cache, allow_pickle=False) as saved:
        observed = np.asarray(saved["observed_z_I"], float)
        reference = np.asarray(saved["evaluation_z_I"], float)
        state_paths = {}
        if country in ("CZE", "GRC"):
            state_paths = dict(
                observed_log_I=np.asarray(saved["observed_log_I"], float),
                reference_log_I=np.asarray(saved["evaluation_log_I"], float),
                state_dates=pd.DatetimeIndex(pd.to_datetime(saved["state_dates"])),
            )
            assert state_paths["observed_log_I"].shape == (61,)
            assert state_paths["reference_log_I"].shape == (2500, 61)
            assert np.isfinite(state_paths["observed_log_I"]).all()
            assert np.isfinite(state_paths["reference_log_I"]).all()
            np.testing.assert_array_equal(
                state_paths["reference_log_I"][:, 0],
                np.full(2500, state_paths["observed_log_I"][0]),
            )
            assert state_paths["state_dates"].equals(pd.date_range(start, end, freq="D"))
    assert observed.shape == (60,) and reference.shape == (2500, 60)
    assert np.isfinite(observed).all() and np.isfinite(reference).all()
    evaluation = draws.loc[draws.bootstrap_id.astype(int).mod(2).eq(1)]
    # The older three-country reference cache was saved as float32.
    for name, obs_value, ref_values in [
        (METRIC, np.mean(observed**2), np.mean(reference**2, axis=1)),
        ("martingale_I_max_abs_cum", np.max(np.abs(np.cumsum(observed)))/np.sqrt(60),
         np.max(np.abs(np.cumsum(reference, axis=1)), axis=1)/np.sqrt(60)),
        ("bracket_I_max_abs_cum", np.max(np.abs(np.cumsum(observed**2-1)))/np.sqrt(60),
         np.max(np.abs(np.cumsum(reference**2-1, axis=1)), axis=1)/np.sqrt(60)),
    ]:
        np.testing.assert_allclose(obs_value, metrics[name], rtol=0, atol=1e-9)
        np.testing.assert_allclose(ref_values, evaluation[name], rtol=2e-6, atol=2e-7)
    reference_energy = float(evaluation[METRIC].median())
    return dict(country=country, observed=observed, reference=reference, metrics=metrics,
                score=score, start=start, end=end, reference_energy=reference_energy,
                source=str(source.relative_to(ROOT)).replace("\\", "/"), **state_paths)


def window_energy_table() -> pd.DataFrame:
    rows = []
    for item in pd.read_csv(WINDOWS / "window_scores.csv").itertuples(index=False):
        source = (BATCH / item.country if item.window_label == "published_midpoint"
                  else WINDOWS / item.country / item.window_label)
        table = pd.read_csv(source / "component_summary.csv")
        component = table.loc[table.metric.eq(METRIC)].iloc[0]
        median = float(component.evaluation_q500)
        rows.append(dict(country=item.country, window=item.window_label,
                         family=item.window_family, start_date=item.start_date,
                         end_date=item.end_date, observed=float(component.observed),
                         reference_median=median,
                         energy_ratio=float(component.observed)/median,
                         reference_low=float(component.evaluation_q025)/median,
                         reference_high=float(component.evaluation_q975)/median,
                         global_reference_rank=float(item.global_rank_pvalue),
                         dominant_metric=item.dominant_metric))
    result = pd.DataFrame(rows).sort_values(["country", "start_date"])
    assert len(result) == 21
    additional = result.loc[result.window.ne("published_midpoint")]
    assert len(additional) == 18 and additional.global_reference_rank.lt(.05).all()
    assert additional.loc[additional.country.eq("MDA"), "energy_ratio"].lt(1).sum() == 5
    result.to_csv(OUT / "window_energy_ratios.csv", index=False)
    return result
