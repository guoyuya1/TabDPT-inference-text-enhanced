from __future__ import annotations

from dataclasses import MISSING, dataclass

from .fine_tune_configs import (
    DataConfig,
    ModelConfig,
    TuningConfig,
    _validate_positive_int,
    load_dataset_config,
)


def _validate_gpu_ids(name: str, values: object) -> list[int] | None:
    if values is None:
        return None
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list when provided.")
    if not values:
        raise ValueError(f"{name} must be non-empty when provided.")

    validated_values: list[int] = []
    seen_values: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must contain only non-negative integers.")
        if value < 0:
            raise ValueError(f"{name} must contain only non-negative integers.")
        if value in seen_values:
            raise ValueError(f"{name} must not contain duplicate GPU ids.")
        validated_values.append(value)
        seen_values.add(value)
    return validated_values


@dataclass(frozen=True)
class FeatureSearchConfig:
    max_context: list[int | None]
    target_lag_count: list[int]
    covariate_lag_count: list[int]
    include_calendar_features: bool = True
    include_seasonal_features: bool = True
    normalize: bool = True
    max_parallel_trialsper_gpu: int = 1
    gpu_ids: list[int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_context",
            _validate_choice_ints(
                "feature_search.max_context",
                self.max_context,
                allow_none_values=True,
            ),
        )
        object.__setattr__(
            self,
            "target_lag_count",
            _validate_choice_ints("feature_search.target_lag_count", self.target_lag_count),
        )
        object.__setattr__(
            self,
            "covariate_lag_count",
            _validate_choice_ints(
                "feature_search.covariate_lag_count",
                self.covariate_lag_count,
                allow_zero_values=True,
            ),
        )
        for field_name in ("include_calendar_features", "include_seasonal_features", "normalize"):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise ValueError(f"feature_search.{field_name} must be a boolean.")
        object.__setattr__(
            self,
            "max_parallel_trialsper_gpu",
            _validate_positive_int(
                "feature_search.max_parallel_trialsper_gpu",
                self.max_parallel_trialsper_gpu,
            ),
        )
        object.__setattr__(
            self,
            "gpu_ids",
            _validate_gpu_ids("feature_search.gpu_ids", self.gpu_ids),
        )


@dataclass(frozen=True, kw_only=True)
class FeatureSearchDataConfig(DataConfig):
    feature_search: FeatureSearchConfig


def _validate_choice_ints(
    name: str,
    values: object,
    *,
    allow_none_values: bool = False,
    allow_zero_values: bool = False,
) -> list[int | None]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list.")
    if not values:
        raise ValueError(f"{name} must be non-empty.")

    validated_values: list[int | None] = []
    for value in values:
        if value is None:
            if not allow_none_values:
                raise ValueError(f"{name} must not contain null values.")
            validated_values.append(None)
            continue
        if allow_zero_values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must contain integers.")
            if value < 0:
                raise ValueError(f"{name} must contain only non-negative integers.")
            validated_values.append(value)
            continue
        validated_values.append(_validate_positive_int(name, value))
    return validated_values


def _default_feature_search_model_config() -> ModelConfig:
    return ModelConfig(
        device="auto",
        model_weight_path=None,
        text_attn_layers=[16],
        use_flash=True,
        compile_model=True,
    )


def _default_feature_search_tuning_config() -> TuningConfig:
    return TuningConfig(
        epochs=1,
        gate_lr=0.1,
        text_attn_lr=0.01,
        gate_logit_clamp=None,
        tune_batch_size=1,
        max_context=None,
        debug_text_effect=False,
        log_text_mixing_params=False,
        step_log_every=1,
        early_stopping_patience=1,
        early_stopping_metric="mae",
    )


def load_feature_search_config(config_path: str, dataset_name: str | None) -> FeatureSearchDataConfig:
    dataset_cfg = dict(load_dataset_config(config_path, dataset_name))
    feature_search_cfg = dataset_cfg.pop("feature_search", None)
    if not isinstance(feature_search_cfg, dict):
        raise ValueError("Feature-search config must define a 'feature_search' mapping.")

    if "base_config" in dataset_cfg or "base_dataset" in dataset_cfg:
        raise ValueError(
            "Feature-search config must be standalone and must not define 'base_config' or 'base_dataset'."
        )

    dataset_cfg.setdefault("prediction_window", 1)
    dataset_cfg.setdefault("calendar_frequency", None)
    dataset_cfg.setdefault("seasonality_k", 3)
    dataset_cfg.setdefault("seasonality_L", None)
    dataset_cfg.setdefault("embedding_lags", [])
    dataset_cfg.setdefault("embedding_columns", None)
    dataset_cfg.setdefault("embedding_column_template", None)
    dataset_cfg.setdefault("target_mode", "original")
    dataset_cfg.setdefault("model", _default_feature_search_model_config())
    dataset_cfg.setdefault("tuning", _default_feature_search_tuning_config())

    if not isinstance(dataset_cfg["model"], ModelConfig):
        model_cfg = dict(dataset_cfg["model"])
        dataset_cfg["model"] = ModelConfig(**model_cfg)
    if not isinstance(dataset_cfg["tuning"], TuningConfig):
        tuning_cfg = dict(dataset_cfg["tuning"])
        if "qk_norm_type" in tuning_cfg:
            if "text_qk_norm" in tuning_cfg:
                raise ValueError("tuning must not define both 'qk_norm_type' and 'text_qk_norm'.")
            tuning_cfg["text_qk_norm"] = tuning_cfg.pop("qk_norm_type")
        dataset_cfg["tuning"] = TuningConfig(
            **{
                key: tuning_cfg[key]
                for key in TuningConfig.__dataclass_fields__
                if key in tuning_cfg
            }
        )

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
            "Feature-search config is missing required data keys: "
            + ", ".join(sorted(missing_data_keys))
        )

    dataset_cfg["seasonality_k"] = _validate_positive_int("seasonality_k", dataset_cfg["seasonality_k"])
    dataset_cfg["prediction_window"] = _validate_positive_int("prediction_window", dataset_cfg["prediction_window"])
    target_mode = str(dataset_cfg["target_mode"]).strip().lower()
    allowed_target_modes = {"original", "target_differencing"}
    if target_mode not in allowed_target_modes:
        raise ValueError(
            "target_mode must be one of "
            + ", ".join(sorted(allowed_target_modes))
            + f". Got: {dataset_cfg['target_mode']!r}"
        )
    dataset_cfg["target_mode"] = target_mode

    return FeatureSearchDataConfig(
        **{
            key: dataset_cfg[key]
            for key in DataConfig.__dataclass_fields__
        },
        feature_search=FeatureSearchConfig(**feature_search_cfg),
    )
