from __future__ import annotations

from dataclasses import MISSING, dataclass
from pathlib import Path

import yaml

try:
    from .fine_tune_configs import _validate_text_attn_layers
except ImportError:
    from fine_tune_configs import _validate_text_attn_layers


def _validate_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _validate_positive_int(name: str, value: object) -> int:
    validated = _validate_int(name, value)
    if validated <= 0:
        raise ValueError(f"{name} must be positive.")
    return validated


def _validate_non_negative_int(name: str, value: object) -> int:
    validated = _validate_int(name, value)
    if validated < 0:
        raise ValueError(f"{name} must be non-negative.")
    return validated


def _validate_choice_list(name: str, values: object) -> list[object] | None:
    if values is None:
        return None
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list when provided.")
    if not values:
        raise ValueError(f"{name} must be non-empty when provided.")
    return values


def _validate_text_attn_layer_choices(name: str, values: object) -> list[list[int]] | None:
    raw_values = _validate_choice_list(name, values)
    if raw_values is None:
        return None
    return [_validate_text_attn_layers(value) for value in raw_values]


def _validate_int_choices(
    name: str,
    values: object,
    *,
    allow_none_values: bool = False,
) -> list[int | None] | None:
    raw_values = _validate_choice_list(name, values)
    if raw_values is None:
        return None

    validated_values: list[int | None] = []
    for value in raw_values:
        if value is None:
            if not allow_none_values:
                raise ValueError(f"{name} must not contain null values.")
            validated_values.append(None)
            continue
        validated_value = _validate_positive_int(name, value)
        validated_values.append(validated_value)
    return validated_values


def _validate_float_choices(
    name: str,
    values: object,
    *,
    allow_none_values: bool = False,
) -> list[float | None] | None:
    raw_values = _validate_choice_list(name, values)
    if raw_values is None:
        return None

    validated_values: list[float | None] = []
    for value in raw_values:
        if value is None:
            if not allow_none_values:
                raise ValueError(f"{name} must not contain null values.")
            validated_values.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must contain only numeric values.")
        validated_value = float(value)
        if validated_value <= 0.0:
            raise ValueError(f"{name} must contain only positive numeric values.")
        validated_values.append(validated_value)
    return validated_values


def _validate_gpu_ids(name: str, values: object) -> list[int] | None:
    raw_values = _validate_choice_list(name, values)
    if raw_values is None:
        return None

    validated_values: list[int] = []
    seen_values: set[int] = set()
    for value in raw_values:
        validated_value = _validate_non_negative_int(name, value)
        if validated_value in seen_values:
            raise ValueError(f"{name} must not contain duplicate GPU ids.")
        validated_values.append(validated_value)
        seen_values.add(validated_value)
    return validated_values


@dataclass(frozen=True)
class SearchSpaceConfig:
    text_attn_layers: list[list[int]] | None = None
    epochs: list[int] | None = None
    gate_lr: list[float] | None = None
    text_attn_lr: list[float] | None = None
    gate_logit_clamp: list[float | None] | None = None
    tune_batch_size: list[int] | None = None
    max_context: list[int | None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "text_attn_layers",
            _validate_text_attn_layer_choices("search_space.text_attn_layers", self.text_attn_layers),
        )
        object.__setattr__(self, "epochs", _validate_int_choices("search_space.epochs", self.epochs))
        object.__setattr__(self, "gate_lr", _validate_float_choices("search_space.gate_lr", self.gate_lr))
        object.__setattr__(
            self,
            "text_attn_lr",
            _validate_float_choices("search_space.text_attn_lr", self.text_attn_lr),
        )
        object.__setattr__(
            self,
            "gate_logit_clamp",
            _validate_float_choices(
                "search_space.gate_logit_clamp",
                self.gate_logit_clamp,
                allow_none_values=True,
            ),
        )
        object.__setattr__(
            self,
            "tune_batch_size",
            _validate_int_choices("search_space.tune_batch_size", self.tune_batch_size),
        )
        object.__setattr__(
            self,
            "max_context",
            _validate_int_choices(
                "search_space.max_context",
                self.max_context,
                allow_none_values=True,
            ),
        )
        if not self.has_dimensions():
            raise ValueError("Random search config must define at least one non-empty search dimension.")

    def has_dimensions(self) -> bool:
        return any(
            getattr(self, field_name) is not None
            for field_name in self.__dataclass_fields__
        )


@dataclass(frozen=True)
class RandomSearchConfig:
    base_config: str
    trials: int
    seed: int
    search_space: SearchSpaceConfig
    base_dataset: str | None = None
    max_parallel_trials: int = 1
    gpu_ids: list[int] | None = None
    top_k: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.base_config, str) or not self.base_config.strip():
            raise ValueError("base_config must be a non-empty string path.")
        if self.base_dataset is not None and not isinstance(self.base_dataset, str):
            raise ValueError("base_dataset must be a string or null.")
        object.__setattr__(self, "trials", _validate_positive_int("trials", self.trials))
        object.__setattr__(self, "seed", _validate_int("seed", self.seed))
        object.__setattr__(
            self,
            "max_parallel_trials",
            _validate_positive_int("max_parallel_trials", self.max_parallel_trials),
        )
        object.__setattr__(self, "gpu_ids", _validate_gpu_ids("gpu_ids", self.gpu_ids))
        object.__setattr__(self, "top_k", _validate_positive_int("top_k", self.top_k))


def load_random_search_config(config_path: str) -> RandomSearchConfig:
    resolved_config_path = Path(config_path).expanduser().resolve()
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Random search config not found: {config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Random search config must be a YAML mapping.")

    missing_keys = [
        key
        for key, field in RandomSearchConfig.__dataclass_fields__.items()
        if key not in config
        and field.default is MISSING
        and field.default_factory is MISSING
    ]
    if missing_keys:
        raise ValueError(
            "Random search config is missing required keys: "
            + ", ".join(sorted(missing_keys))
        )

    unknown_keys = sorted(set(config) - set(RandomSearchConfig.__dataclass_fields__))
    if unknown_keys:
        raise ValueError(
            "Random search config has unsupported keys: "
            + ", ".join(unknown_keys)
        )

    search_space_cfg = config["search_space"]
    if not isinstance(search_space_cfg, dict):
        raise ValueError("search_space must be a mapping.")

    unknown_search_keys = sorted(set(search_space_cfg) - set(SearchSpaceConfig.__dataclass_fields__))
    if unknown_search_keys:
        raise ValueError(
            "Random search config has unsupported search_space keys: "
            + ", ".join(unknown_search_keys)
        )

    base_config_path = Path(config["base_config"]).expanduser()
    if not base_config_path.is_absolute():
        base_config_path = (resolved_config_path.parent / base_config_path).resolve()
    else:
        base_config_path = base_config_path.resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base fine-tune config not found: {base_config_path}")

    return RandomSearchConfig(
        base_config=str(base_config_path),
        base_dataset=config.get("base_dataset"),
        trials=config["trials"],
        seed=config["seed"],
        max_parallel_trials=config.get("max_parallel_trials", 1),
        gpu_ids=config.get("gpu_ids"),
        top_k=config.get("top_k", 10),
        search_space=SearchSpaceConfig(
            **{
                key: search_space_cfg[key]
                for key in SearchSpaceConfig.__dataclass_fields__
                if key in search_space_cfg
            }
        ),
    )
