"""
Create embeddings with lagged text features for the climate CSV. Should be extended to other dataframes easily in the future.

Note: onlt raw data should be uploaded to our repo.

MZ: Missing text is filled with the literal string ``n/a`` so we never use an all-zero
vector placeholder for unavailable text features., e.g. for the first few rows where lagged text is not available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Sequence

import pandas as pd
import torch
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer

# Fixed configuration (edit here if needed)
csv_path = "./Time-MMD/numerical/Economy/Economy_with_text.csv"
text_columns = None  # e.g. ["text"] or ["report_fact", "search_fact"]
date_column = "date"
model_name = "Qwen/Qwen3-Embedding-8B" # "Qwen/Qwen3-Embedding-0.6B"
lag_days = 3  # number of prior days to include (in addition to current)
batch_size = 16
max_length = 1024
device = "cuda" if torch.cuda.is_available() else "cpu"
output_csv_path = None
normalize_embeddings = False
na_text = "n/a"
text_column_hints = ("text", "fact", "report", "search", "headline", "summary")


def sanitize_text_series(series: pd.Series, na_text: str) -> pd.Series:
    # MZ: Replace NaN or empty strings with na_text. Make it a function for reusability and flexibility.
    series = series.fillna("").astype(str).str.strip()
    return series.mask(series == "", na_text)


def parse_columns(columns_arg: str | Sequence[str] | None) -> List[str] | None:
    if columns_arg is None:
        return None
    if isinstance(columns_arg, str):
        if not columns_arg.strip():
            return None
        return [col.strip() for col in columns_arg.split(",") if col.strip()]
    return [str(col).strip() for col in columns_arg if str(col).strip()]


def parse_text_columns(text_columns_arg: str | Sequence[str] | None) -> List[str] | None:
    return parse_columns(text_columns_arg)


def load_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}

    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = OmegaConf.load(path)
    if config is None:
        return {}
    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict):
        raise ValueError("Config must be a mapping of keys to values.")
    return config_dict


def resolve_config_path(value: str | None, config_dir: Path | None) -> str | None:
    if not value or not config_dir:
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return value
    return str((config_dir / candidate).resolve())


def resolve_csv_path(csv_path: str) -> Path:
    raw_path = Path(csv_path)
    candidates = []

    def add_candidate(candidate: Path) -> None:
        candidates.append(candidate)

    add_candidate(raw_path)
    if raw_path.suffix == "":
        add_candidate(raw_path.with_suffix(".csv"))

    if "TimeMMD" in str(raw_path):
        dashed = Path(str(raw_path).replace("TimeMMD", "Time-MMD"))
        add_candidate(dashed)
        if dashed.suffix == "":
            add_candidate(dashed.with_suffix(".csv"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    if raw_path.is_dir():
        matches = sorted(raw_path.glob("*with_text*.csv"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple '*with_text*.csv' files found in {raw_path}: "
                f"{', '.join(str(match) for match in matches)}"
            )

    search_name = raw_path.name
    if raw_path.suffix == "":
        search_name = f"{search_name}.csv"

    for root in [Path("Time-MMD/numerical"), Path("Time-MMD")]:
        if not root.is_dir():
            continue
        matches = sorted(root.rglob(search_name))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple '{search_name}' files found under {root}: "
                f"{', '.join(str(match) for match in matches)}"
            )

    raise FileNotFoundError(
        "CSV not found. Tried: " + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_text_columns(
    df: pd.DataFrame, requested: Sequence[str] | None
) -> List[str]:
    if requested:
        missing = [col for col in requested if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing text columns {missing}. Available columns: {list(df.columns)}"
            )
        return list(requested)

    object_cols = [col for col in df.columns if df[col].dtype == object]
    if not object_cols:
        raise ValueError("No object columns found to auto-detect text fields.")

    hinted = [
        col
        for col in object_cols
        if any(hint in col.lower() for hint in text_column_hints)
    ]
    if hinted:
        return hinted

    raise ValueError(
        "No text columns specified and none matched hints "
        f"{text_column_hints}. Pass --text-columns. Object columns: {object_cols}"
    )


def add_combined_text_column(
    df: pd.DataFrame, text_columns: List[str], left: str, right: str, na_text: str
) -> List[str]:
    if left not in text_columns or right not in text_columns:
        return text_columns

    combined_col = f"{left}_{right}"
    if combined_col in df.columns:
        if combined_col not in text_columns:
            text_columns.append(combined_col)
        return text_columns

    left_series = df[left].fillna("").astype(str).str.strip()
    right_series = df[right].fillna("").astype(str).str.strip()
    combined_series = (left_series + " " + right_series).str.strip()
    df[combined_col] = sanitize_text_series(combined_series, na_text)
    text_columns.append(combined_col)
    return text_columns


def default_output_path(csv_path: Path, lag_days: int) -> Path:
    output_name = f"{csv_path.stem}_with_embeddings_lag_{lag_days}.csv"
    parts = list(csv_path.parts)
    if "source_data" in parts:
        idx = parts.index("source_data")
        processed_root = Path(*parts[:idx], "processed_data")
        relative_dir = Path(*parts[idx + 1 : -1])
        return processed_root / relative_dir / output_name
    return csv_path.with_name(output_name)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_parsed, _ = pre_parser.parse_known_args(args)
    config = load_config(pre_parsed.config)
    config_dir = (
        Path(pre_parsed.config).expanduser().resolve().parent
        if pre_parsed.config
        else None
    )
    config_source_path = config.get("source_csv_path") or config.get("csv_path")
    config_output_path = config.get("output_csv_path")
    config_source_path = resolve_config_path(config_source_path, config_dir)
    config_output_path = resolve_config_path(config_output_path, config_dir)

    parser = argparse.ArgumentParser(
        description="Create embeddings with lagged text features for a CSV."
    )
    parser.add_argument("--config", default=pre_parsed.config)
    parser.add_argument("--csv-path", default=argparse.SUPPRESS)
    parser.add_argument(
        "--text-columns",
        default=argparse.SUPPRESS,
        help="Comma-separated list of text columns; auto-detect if omitted.",
    )
    parser.add_argument(
        "--target-columns",
        default=argparse.SUPPRESS,
        help="Comma-separated list of target columns to validate (optional).",
    )
    parser.add_argument("--date-column", default=argparse.SUPPRESS)
    parser.add_argument("--lag-days", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--output-csv-path", default=argparse.SUPPRESS)
    parser.add_argument("--model-name", default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-length", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument(
        "--normalize-embeddings", action="store_true", default=argparse.SUPPRESS
    )
    parser.add_argument("--na-text", default=argparse.SUPPRESS)
    parsed = parser.parse_args(args)

    defaults = {
        "config": pre_parsed.config,
        "csv_path": config_source_path or csv_path,
        "text_columns": config.get("text_columns", text_columns),
        "target_columns": config.get("target_columns"),
        "date_column": config.get("date_column", date_column),
        "lag_days": config.get("lag_days", lag_days),
        "output_csv_path": config_output_path or output_csv_path,
        "model_name": config.get("model_name", model_name),
        "batch_size": config.get("batch_size", batch_size),
        "max_length": config.get("max_length", max_length),
        "device": config.get("device", device),
        "normalize_embeddings": config.get(
            "normalize_embeddings", normalize_embeddings
        ),
        "na_text": config.get("na_text", na_text),
    }
    overrides = vars(parsed)
    merged = {**defaults, **overrides}
    return argparse.Namespace(**merged)



def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    normalize_embeddings: bool,
    pool: dict | None = None,
) -> List[list[float]]:
    if pool is not None:
        embeddings = model.encode_multi_process(
            texts,
            pool,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings,
        )
        return embeddings.tolist()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        # MZ: TODO: try both with and without normalization
        normalize_embeddings=normalize_embeddings,
    )
    return embeddings.cpu().tolist()


def main(args: Sequence[str] | None = None) -> None:
    parsed = parse_args(args)
    resolved_csv_path = resolve_csv_path(parsed.csv_path)

    # read df
    df = pd.read_csv(resolved_csv_path)
    target_columns = parse_columns(parsed.target_columns)
    if target_columns:
        missing = [col for col in target_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing target columns {missing}. Available columns: {list(df.columns)}"
            )
    resolved_text_columns = resolve_text_columns(
        df, parse_text_columns(parsed.text_columns)
    )
    resolved_text_columns = add_combined_text_column(
        df,
        resolved_text_columns,
        "report_fact",
        "search_fact",
        parsed.na_text,
    )
    if parsed.date_column and parsed.date_column in df.columns:
        df = df.sort_values(parsed.date_column).reset_index(drop=True)

    # create lagged text columns.
    lag_text_cols = []
    for text_col in resolved_text_columns:
        for lag in range(parsed.lag_days + 1):
            col_name = f"{text_col}_lag{lag}"
            if lag == 0:
                df[col_name] = df[text_col]
            else:
                # get lag feature by shifting lag days
                df[col_name] = df[text_col].shift(lag)
            df[col_name] = sanitize_text_series(df[col_name], parsed.na_text)
            lag_text_cols.append(col_name)

    print(
        f"Loaded {len(df)} rows from {resolved_csv_path} on device {parsed.device} "
        f"with lags 0..{parsed.lag_days}."
    )
    print(f"Text columns: {', '.join(resolved_text_columns)}")

    target_devices = None
    if str(parsed.device).startswith("cuda"):
        target_devices = ["cuda:0", "cuda:1"]
        print(f"Using multi-process pool on devices: {', '.join(target_devices)}")

    model_device = "cpu" if target_devices else parsed.device
    if target_devices:
        print("Loading model on CPU in the main process for multi-GPU encoding.")
    model = SentenceTransformer(parsed.model_name, device=model_device)
    model.max_seq_length = parsed.max_length

    pool = None
    if target_devices:
        pool = model.start_multi_process_pool(target_devices=target_devices)

    try:
        # create embeddings for cols
        for col_name in lag_text_cols:
            print(f"Encoding column '{col_name}'...")
            col_embeddings = embed_texts(
                model,
                df[col_name].tolist(),
                batch_size=parsed.batch_size,
                normalize_embeddings=parsed.normalize_embeddings,
                pool=pool,
            )
            df[f"embedding_{col_name}"] = col_embeddings
    finally:
        if pool is not None:
            model.stop_multi_process_pool(pool)

    resolved_output_path = (
        Path(parsed.output_csv_path)
        if parsed.output_csv_path
        else default_output_path(resolved_csv_path, parsed.lag_days)
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(resolved_output_path, index=False)
    print(f"Wrote CSV with embeddings to {resolved_output_path}")


if __name__ == "__main__":
    main()
