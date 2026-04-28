#!/usr/bin/env python3
"""Append time + calendar features (from tabdpt.feature_build) to processed CSVs.

Pass A loads only the date and target columns, sorts chronologically for
generate_time_features, then maps features back to original row order.
Pass B streams the full CSV with csv.reader/writer and appends feature columns
so wide embedding fields are not loaded into a single DataFrame.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tabdpt.feature_build import generate_time_features  # noqa: E402

logger = logging.getLogger(__name__)


def _ensure_large_csv_field_limit() -> None:
    """Python's csv default max field size is 128KiB; embedding columns exceed that."""
    limit = sys.maxsize
    while limit > 1:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV: {path}") from exc


def feature_column_names(dim: int, *, prefix: str = "time_cal_feat_") -> list[str]:
    width = max(3, len(str(dim - 1)))
    return [f"{prefix}{i:0{width}d}" for i in range(dim)]


def fmt_feature_cell(value: float) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return ""
    return f"{float(value):.12g}"


def compute_features_original_order(
    csv_path: Path,
    *,
    date_column: str,
    target_column: str,
    frequency: str,
    k: int,
    use_index: bool,
    L: int | None,
) -> tuple[np.ndarray, list[str]]:
    """Returns (features, names) with shape (n_rows, n_features) in original CSV row order."""
    header = read_csv_header(csv_path)
    for col in (date_column, target_column):
        if col not in header:
            raise KeyError(f"Column {col!r} not in {csv_path}; available: {header[:20]}...")

    df = pd.read_csv(csv_path, usecols=[date_column, target_column])
    if len(df) == 0:
        raise ValueError(f"No data rows in {csv_path}")

    n = len(df)
    df = df.copy()
    df["__row_id__"] = np.arange(n, dtype=np.int64)
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    bad = df[date_column].isna() | df[target_column].isna()
    if bad.any():
        raise ValueError(
            f"{csv_path}: {int(bad.sum())} row(s) have invalid dates or non-numeric targets "
            f"after coercion (columns {date_column!r}, {target_column!r})."
        )
    vals = df[target_column].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        n_bad = int(np.sum(~np.isfinite(vals)))
        raise ValueError(
            f"{csv_path}: target column {target_column!r} has {n_bad} non-finite value(s) (inf/nan)."
        )

    df_sorted = df.sort_values(date_column, kind="mergesort").reset_index(drop=True)
    timestamps = df_sorted[date_column]
    targets = np.asarray(df_sorted[target_column], dtype=float)
    row_ids = df_sorted["__row_id__"].to_numpy(dtype=np.int64)

    feats_sorted = generate_time_features(
        targets,
        timestamps,
        frequency,
        k,
        L=L,
        use_index=use_index,
    )
    if feats_sorted.ndim == 1:
        feats_sorted = feats_sorted.reshape(-1, 1)

    dim = feats_sorted.shape[1]
    feats_orig = np.empty((n, dim), dtype=float)
    feats_orig[row_ids] = feats_sorted
    names = feature_column_names(dim)
    return feats_orig, names


def stream_append_features(
    src: Path,
    dst: Path,
    features: np.ndarray,
    new_columns: list[str],
) -> None:
    """Write dst = src rows plus appended feature columns (same row order)."""
    _ensure_large_csv_field_limit()
    n = features.shape[0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", newline="") as fin, dst.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout, lineterminator="\n")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV: {src}") from exc

        writer.writerow(list(header) + new_columns)
        i = -1
        for i, row in enumerate(reader):
            if i >= n:
                raise ValueError(
                    f"{src}: more data rows ({i + 1}) than feature matrix rows ({n})."
                )
            extra = [fmt_feature_cell(x) for x in features[i]]
            writer.writerow(list(row) + extra)
        if i + 1 != n:
            raise ValueError(
                f"{src}: expected {n} data rows after header, found {i + 1}."
            )


def resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_spec(spec_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with spec_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Spec must be a mapping, got {type(raw)}")
    defaults = raw.get("defaults") or {}
    datasets = raw.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Spec must contain a non-empty 'datasets' list")
    return defaults, datasets


def merged_entry(defaults: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(defaults)
    cfg.update(entry)
    return cfg


def output_path_for(
    rel_path: str,
    *,
    processed_root: Path,
    output_root: Path,
) -> Path:
    rel = Path(rel_path)
    return (output_root / rel).resolve()


def process_one(
    entry: dict[str, Any],
    *,
    processed_root: Path,
    output_root: Path,
    dry_run: bool,
    overwrite: bool,
) -> None:
    rel = entry["path"]
    src = (processed_root / rel).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Missing input CSV: {src}")

    date_column = str(entry["date_column"])
    target_column = str(entry["target_column"])
    frequency = str(entry["frequency"])
    k = int(entry.get("k", 3))
    use_index = bool(entry.get("use_index", True))
    L = entry.get("L")
    L_int = int(L) if L is not None else None

    dst = output_path_for(rel, processed_root=processed_root, output_root=output_root)
    if dst.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Refusing to overwrite existing file: {dst} (use --overwrite)")

    feats, names = compute_features_original_order(
        src,
        date_column=date_column,
        target_column=target_column,
        frequency=frequency,
        k=k,
        use_index=use_index,
        L=L_int,
    )
    logger.info(
        "Prepared %s rows x %s features for %s -> %s",
        feats.shape[0],
        feats.shape[1],
        src.relative_to(REPO_ROOT) if src.is_relative_to(REPO_ROOT) else src,
        dst.relative_to(REPO_ROOT) if dst.is_relative_to(REPO_ROOT) else dst,
    )
    if dry_run:
        return
    stream_append_features(src, dst, feats, names)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--spec",
        type=str,
        default="data/processed_data/time_calendar_feature_spec.yaml",
        help="YAML manifest (repo-relative or absolute)",
    )
    p.add_argument(
        "--root",
        type=str,
        default="data/processed_data",
        help="Root directory for dataset paths in the manifest",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="data/processed_data_time_features",
        help="Output tree root (mirrors manifest paths)",
    )
    p.add_argument("--dry-run", action="store_true", help="Log actions without writing files")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    spec_path = resolve_repo_path(args.spec)
    processed_root = resolve_repo_path(args.root)
    output_root = resolve_repo_path(args.output_dir)
    defaults, datasets = load_spec(spec_path)

    for raw_entry in datasets:
        if not isinstance(raw_entry, dict):
            raise TypeError(f"Each datasets entry must be a mapping, got {type(raw_entry)}")
        entry = merged_entry(defaults, raw_entry)
        if "path" not in entry:
            raise KeyError("Each dataset entry must include 'path'")
        if not bool(entry.get("enabled", True)):
            logger.info("Skipping disabled dataset %s", entry["path"])
            continue
        for key in ("date_column", "target_column", "frequency"):
            if key not in entry:
                raise KeyError(f"Dataset {entry.get('path')} missing {key!r}")
        try:
            process_one(
                entry,
                processed_root=processed_root,
                output_root=output_root,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
        except Exception:
            logger.exception("Failed processing %s", entry.get("path"))
            raise


if __name__ == "__main__":
    main()
