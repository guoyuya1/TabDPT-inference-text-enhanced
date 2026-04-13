from __future__ import annotations

import os
from dataclasses import MISSING, dataclass

import yaml


@dataclass(frozen=True)
class ModelConfig:
    device: str | None
    model_weight_path: str | None
    text_attn_layers: list[int]
    use_flash: bool
    compile_model: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "text_attn_layers", _validate_text_attn_layers(self.text_attn_layers))


@dataclass(frozen=True)
class TuningConfig:
    epochs: int
    gate_lr: float
    text_attn_lr: float
    gate_logit_clamp: float | None
    tune_batch_size: int
    max_context: int | None
    debug_text_effect: bool
    log_text_mixing_params: bool
    step_log_every: int
    early_stopping_patience: int
    loss_type: str = "l1"
    log_text_score_stats: bool = False
    text_score_sample_size: int = 8
    early_stopping_metric: str = "mae"


@dataclass(frozen=True)
class DataConfig:
    data_path: str
    date_column: str | None
    numeric_features: list[str]
    target_column: str
    embedding_lags: list[int]
    embedding_columns: list[str] | None
    embedding_column_template: str | None
    max_rows: int | None
    context_ratio: float
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    model: ModelConfig
    tuning: TuningConfig
    prediction_window: int = 1


def _validate_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _validate_text_attn_layers(text_attn_layers: object) -> list[int]:
    if not isinstance(text_attn_layers, list):
        raise ValueError("model.text_attn_layers must be a list of 1-based encoder layer numbers.")
    if not text_attn_layers:
        raise ValueError("model.text_attn_layers must be non-empty.")

    validated_layers: list[int] = []
    seen_layers: set[int] = set()
    for layer in text_attn_layers:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError("model.text_attn_layers must contain only integers.")
        if layer <= 0:
            raise ValueError("model.text_attn_layers must contain only positive 1-based layer numbers.")
        if layer in seen_layers:
            raise ValueError("model.text_attn_layers must not contain duplicate layer numbers.")
        validated_layers.append(layer)
        seen_layers.add(layer)
    return validated_layers


def load_dataset_config(config_path: str, dataset_name: str | None) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if dataset_name is None:
        return config
    return config[dataset_name]


def load_fine_tune_config(config_path: str, dataset_name: str | None) -> DataConfig:
    dataset_cfg = dict(load_dataset_config(config_path, dataset_name))
    dataset_cfg.setdefault("prediction_window", 1)
    dataset_cfg.setdefault("embedding_lags", [])
    dataset_cfg.setdefault("embedding_columns", None)
    dataset_cfg.setdefault("embedding_column_template", None)
    missing_data_keys = [
        key
        for key, field in DataConfig.__dataclass_fields__.items()
        if key not in {"model", "tuning"}
        and key not in dataset_cfg
        and field.default is MISSING
        and field.default_factory is MISSING
    ]
    if missing_data_keys:
        raise ValueError(
            "Fine-tune config is missing required data keys: "
            + ", ".join(sorted(missing_data_keys))
        )
    model_cfg = dict(dataset_cfg["model"])
    if "text_enhanced" in model_cfg:
        raise ValueError(
            "Fine-tune config must use model.text_attn_layers and must not define the legacy "
            "model.text_enhanced flag."
        )
    missing_model_keys = [
        key
        for key, field in ModelConfig.__dataclass_fields__.items()
        if key not in model_cfg
        and field.default is MISSING
        and field.default_factory is MISSING
    ]
    if missing_model_keys:
        raise ValueError(
            "Fine-tune config is missing required model keys: "
            + ", ".join(sorted(missing_model_keys))
        )
    unknown_model_keys = sorted(set(model_cfg) - set(ModelConfig.__dataclass_fields__))
    if unknown_model_keys:
        raise ValueError(
            "Fine-tune config has unsupported model keys: "
            + ", ".join(unknown_model_keys)
        )
    dataset_cfg["model"] = ModelConfig(**model_cfg)
    missing_tuning_keys = [
        key
        for key, field in TuningConfig.__dataclass_fields__.items()
        if key not in dataset_cfg["tuning"]
        and field.default is MISSING
        and field.default_factory is MISSING
    ]
    if missing_tuning_keys:
        raise ValueError(
            "Fine-tune config is missing required tuning keys: "
            + ", ".join(sorted(missing_tuning_keys))
        )
    dataset_cfg["tuning"] = TuningConfig(
        **{
            key: dataset_cfg["tuning"][key]
            for key in TuningConfig.__dataclass_fields__
            if key in dataset_cfg["tuning"]
        }
    )
    dataset_cfg = {
        key: dataset_cfg[key]
        for key in DataConfig.__dataclass_fields__
        if key in dataset_cfg
    }
    dataset_cfg["prediction_window"] = _validate_positive_int(
        "prediction_window",
        dataset_cfg["prediction_window"],
    )
    return DataConfig(**dataset_cfg)
