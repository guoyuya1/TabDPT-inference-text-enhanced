from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


EMBEDDING_LAG_PATTERN = re.compile(r"^embedding_.*_lag(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim datasets to a fixed number of rows and save copies."
    )
    parser.add_argument(
        "--dataset-paths",
        nargs="+",
        required=True,
        help="One or more dataset file paths.",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        help="Optional number of rows to keep from the start of each dataset.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where trimmed datasets will be written.",
    )
    parser.add_argument(
        "--keep-embedding-lags",
        nargs="+",
        type=int,
        help="Optional embedding lag indices to keep, for example: 1 2 3.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def output_path_for(
    dataset_path: Path,
    output_dir: Path,
    num_rows: int | None,
    keep_embedding_lags: set[int] | None,
) -> Path:
    suffix = "".join(dataset_path.suffixes)
    stem = dataset_path.name[: -len(suffix)] if suffix else dataset_path.name
    row_suffix = f"_head_{num_rows}" if num_rows is not None else "_all_rows"
    embedding_suffix = ""
    if keep_embedding_lags is not None:
        lag_str = "_".join(str(lag) for lag in sorted(keep_embedding_lags))
        embedding_suffix = f"_embedding_lags_{lag_str}"
    return output_dir / f"{stem}{row_suffix}{embedding_suffix}{suffix}"

def keep_selected_embedding_lag_columns(
    df: pd.DataFrame,
    keep_embedding_lags: set[int] | None,
) -> pd.DataFrame:
    if keep_embedding_lags is None:
        return df

    columns_to_keep: list[str] = []
    for column in df.columns:
        match = EMBEDDING_LAG_PATTERN.match(column)
        if match is None:
            columns_to_keep.append(column)
            continue

        lag = int(match.group(1))
        if lag in keep_embedding_lags:
            columns_to_keep.append(column)

    return df.loc[:, columns_to_keep]


def trim_dataset(
    dataset_path: Path,
    output_dir: Path,
    num_rows: int | None,
    keep_embedding_lags: set[int] | None,
) -> tuple[Path, int]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV datasets are supported: {dataset_path}")

    df = pd.read_csv(dataset_path)
    trimmed_df = df.head(num_rows).copy() if num_rows is not None else df.copy()
    trimmed_df = keep_selected_embedding_lag_columns(trimmed_df, keep_embedding_lags)

    output_path = output_path_for(
        dataset_path,
        output_dir,
        num_rows,
        keep_embedding_lags,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed_df.to_csv(output_path, index=False)
    return output_path, len(trimmed_df)


def main() -> None:
    args = parse_args()
    if args.num_rows is not None and args.num_rows <= 0:
        raise ValueError("--num-rows must be a positive integer.")
    if args.keep_embedding_lags is not None and any(lag < 0 for lag in args.keep_embedding_lags):
        raise ValueError("--keep-embedding-lags values must be non-negative integers.")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_embedding_lags = (
        set(args.keep_embedding_lags) if args.keep_embedding_lags is not None else None
    )

    for raw_path in args.dataset_paths:
        dataset_path = resolve_path(raw_path)
        output_path, row_count = trim_dataset(
            dataset_path,
            output_dir,
            args.num_rows,
            keep_embedding_lags,
        )
        print(f"Saved trimmed dataset to {output_path} ({row_count} rows)")


if __name__ == "__main__":
    main()
