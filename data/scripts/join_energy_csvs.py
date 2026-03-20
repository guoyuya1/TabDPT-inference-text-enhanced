from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_text_csv(path: Path, date_col: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop unnamed index column if present.
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}' in {path}")
    if "fact" not in df.columns:
        raise ValueError(f"Missing 'fact' column in {path}")
    fact_col = f"{prefix}_fact"
    df = df.rename(columns={"fact": fact_col})
    return df[[date_col, fact_col]]


def merge_energy(
    *,
    numeric_path: Path,
    search_path: Path | None,
    numeric_date_col: str,
    search_date_col: str,
    output_path: Path,
) -> None:
    numeric = pd.read_csv(numeric_path)
    if numeric_date_col not in numeric.columns:
        raise ValueError(f"Missing date column '{numeric_date_col}' in {numeric_path}")

    merged = numeric
    if search_path:
        search = _read_text_csv(search_path, search_date_col, "search")
        merged = merged.merge(
            search, left_on=numeric_date_col, right_on=search_date_col, how="left"
        )

    # Drop join keys and date range columns from the output.
    drop_cols = {
        search_date_col,
        "start_date",
        "end_date",
        "start_date_x",
        "start_date_y",
        "end_date_x",
        "end_date_y",
    }
    drop_cols = [col for col in drop_cols if col in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    if f"{numeric_date_col}_x" in merged.columns:
        merged = merged.rename(columns={f"{numeric_date_col}_x": numeric_date_col})
    if f"{numeric_date_col}_y" in merged.columns:
        merged = merged.drop(columns=[f"{numeric_date_col}_y"])

    if "search_fact" in merged.columns:
        merged["search_fact"] = merged["search_fact"].fillna("NA")
    merged.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Join Energy numeric + search CSVs.")
    parser.add_argument(
        "--numeric",
        default="Time-MMD/numerical/Energy/Energy.csv",
        help="Path to numeric Energy CSV.",
    )
    parser.add_argument(
        "--search",
        default="Time-MMD/textual/Energy/Energy_search.csv",
        help="Path to search text CSV (optional).",
    )
    parser.add_argument(
        "--numeric-date-col",
        default="date",
        help="Date column in numeric CSV to join on.",
    )
    parser.add_argument(
        "--search-date-col",
        default="start_date",
        help="Date column in search CSV to join on.",
    )
    parser.add_argument(
        "--output",
        default="Time-MMD/numerical/Energy/Energy_with_text.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    merge_energy(
        numeric_path=Path(args.numeric),
        search_path=Path(args.search) if args.search else None,
        numeric_date_col=args.numeric_date_col,
        search_date_col=args.search_date_col,
        output_path=Path(args.output),
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
