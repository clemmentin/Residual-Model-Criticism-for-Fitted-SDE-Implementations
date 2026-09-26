from __future__ import annotations

from pathlib import Path

import equinox as eqx

import config as config_module
import sir_data
import sir_training
from sir_training import build_like_models, fit_sir_priors  # retained for existing callers
from neural_sde import NeuralSDE

def load_model_ensemble_and_data(
    cfg: config_module.Config,
    *,
    force_refresh: bool = False,
) -> tuple[list[NeuralSDE], config_module.TrainingData, Path]:
    data = sir_data.load_or_create_training_data(cfg, force_refresh=force_refresh)
    template = build_like_models(cfg, data)
    save_path = config_module.get_model_save_path(cfg)
    if not save_path.exists():
        raise FileNotFoundError(f"Missing cached model: {save_path}")

    loaded = deserialize_models(save_path, template)
    for model in loaded:
        sir_training.assert_model_normalization(model, data)
    return loaded, data, save_path


def load_first_model(cfg: config_module.Config, data) -> tuple[NeuralSDE, Path]:
    """Load the first retained model; the caller owns normalization checks."""
    model_path = config_module.get_model_save_path(cfg)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing cached model: {model_path}")
    model = deserialize_models(model_path, build_like_models(cfg, data))[0]
    return model, model_path


def deserialize_models(model_path: Path, template: list[NeuralSDE]) -> list[NeuralSDE]:
    """Read the retained ensemble or historical single-model checkpoint layout."""
    try:
        loaded = eqx.tree_deserialise_leaves(model_path, like=template)
    except Exception:
        return [eqx.tree_deserialise_leaves(model_path, like=template[0])]
    return loaded if isinstance(loaded, list) else [loaded]
