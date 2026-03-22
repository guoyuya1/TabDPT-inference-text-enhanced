from __future__ import annotations

import os
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class ModelConfig:
    device: str | None
    model_weight_path: str | None
    text_enhanced: bool
    use_flash: bool
    compile_model: bool


@dataclass(frozen=True)
class TuningConfig:
    epochs: int
    gate_lr: float
    gate_logit_clamp: float | None
    tune_batch_size: int
    max_context_for_tune: int | None
    max_context_for_eval: int | None
    max_context_for_tune_eval: int | None
    eval_each_epoch: bool
    debug_text_effect: bool
    step_log_every: int


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
    tune_ratio: float
    eval_ratio: float
    seed: int
    model: ModelConfig
    tuning: TuningConfig


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
    dataset_cfg.setdefault("embedding_lags", [])
    dataset_cfg.setdefault("embedding_columns", None)
    dataset_cfg.setdefault("embedding_column_template", None)
    dataset_cfg["model"] = ModelConfig(**dataset_cfg["model"])
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
    return DataConfig(**dataset_cfg)
