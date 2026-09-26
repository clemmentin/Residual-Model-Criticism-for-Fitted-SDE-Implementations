"""Read-only audit of the SIR reconstruction/NLS removal-rate identity."""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import sir_data

from experiments import (
    audit_sir_beta_clipping as beta_audit,
    run_sir_sde_native_country_audit as native,
)


MEMORY_HORIZONS = (14.0, 28.0, 56.0)


def _stack_pairs(country_data):
    s_all, i_all, ds_all, di_all = [], [], [], []
    for frame in country_data.values():
        s = frame["S"].to_numpy(dtype=np.float64)
        i = frame["I"].to_numpy(dtype=np.float64)
        s_all.append(s[:-1])
        i_all.append(i[:-1])
        ds_all.append(np.diff(s))
        di_all.append(np.diff(i))
    return tuple(map(np.concatenate, (s_all, i_all, ds_all, di_all)))


def audit() -> list[dict[str, float | int]]:
    cfg = native._config("FIN")
    raw = beta_audit.raw_snapshot()
    rows = []
    for horizon in MEMORY_HORIZONS:
        country_data = sir_data.prepare_sir_countries(
            raw,
            list(cfg.TRAIN_COUNTRIES),
            cfg.TRAINING_START_DATE,
            recovery_days=horizon,
            active_threshold=cfg.SIR_ACTIVE_THRESHOLD,
            incidence_source=cfg.SIR_INCIDENCE_SOURCE,
            incidence_blend_weight=cfg.SIR_INCIDENCE_BLEND_WEIGHT,
        )
        beta_hat, gamma_hat = sir_data.fit_sir_baseline(country_data)
        s, i, delta_s, delta_i = _stack_pairs(country_data)
        decay = 1.0 - np.exp(-1.0 / horizon)
        reconstruction_error = delta_i + delta_s + decay * i
        r_s = delta_s + beta_hat * s * i
        normal_rhs = float(np.sum(i * r_s) / np.sum(i * i))
        normal_lhs = float(gamma_hat - decay)
        rows.append(
            {
                "memory_horizon": horizon,
                "n_pairs": int(i.size),
                "decay": float(decay),
                "beta_hat": float(beta_hat),
                "gamma_hat": float(gamma_hat),
                "gamma_over_decay": float(gamma_hat / decay),
                "reconstruction_identity_max_abs_error": float(
                    np.max(np.abs(reconstruction_error))
                ),
                "normal_equation_lhs": normal_lhs,
                "normal_equation_rhs": normal_rhs,
                "normal_equation_abs_error": float(abs(normal_lhs - normal_rhs)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "tmp").resolve()):
        raise ValueError("Diagnostic output must be under repository tmp/.")
    if output.exists():
        raise FileExistsError(output)
    rows = audit()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
