from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _concat_text(series: pd.Series) -> str:
    parts = [str(v).strip() for v in series.dropna().tolist()]
    parts = [p for p in parts if p and p != "NA"]
    return "; ".join(parts) if parts else "NA"


def _read_text_csv(
    path: Path,
    *,
    date_col: str,
    prefix: str,
    join_key: str,
    group_by_month: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop unnamed index column if present.
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}' in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    if group_by_month:
        df[join_key] = df[date_col].dt.to_period("M").astype(str)
    else:
        df[join_key] = df[date_col]

    rename_map = {}
    if "fact" in df.columns:
        rename_map["fact"] = f"{prefix}_fact"
    df = df.rename(columns=rename_map)

    keep_cols = [join_key] + [c for c in df.columns if c == f"{prefix}_fact"]
    df = df[keep_cols]

    text_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    if text_cols:
        df = df.groupby(join_key, as_index=False).agg({col: _concat_text for col in text_cols})
    return df


def _read_text_csv_by_month(
    path: Path,
    *,
    date_col: str,
    prefix: str,
    month_shift: int = 0,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}' in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df["__month__"] = (df[date_col].dt.to_period("M") + month_shift).astype(str)

    if "fact" in df.columns:
        df = df.rename(columns={"fact": f"{prefix}_fact"})
    keep_cols = ["__month__", f"{prefix}_fact"]
    df = df[keep_cols]

    df = df.groupby("__month__", as_index=False).agg({f"{prefix}_fact": _concat_text})
    return df


def _merge_climate(
    *,
    numeric_path: Path,
    search_path: Path,
    report_path: Path,
    numeric_date_col: str,
    text_date_col: str,
    group_by_month: bool,
) -> pd.DataFrame:
    numeric = pd.read_csv(numeric_path)
    if numeric_date_col not in numeric.columns:
        raise ValueError(f"Missing date column '{numeric_date_col}' in {numeric_path}")
    numeric[numeric_date_col] = pd.to_datetime(numeric[numeric_date_col])

    join_key = "__join_key__"
    if group_by_month:
        numeric[join_key] = numeric[numeric_date_col].dt.to_period("M").astype(str)
    else:
        numeric[join_key] = numeric[numeric_date_col]

    search = _read_text_csv(
        search_path,
        date_col=text_date_col,
        prefix="search",
        join_key=join_key,
        group_by_month=group_by_month,
    )
    report = _read_text_csv(
        report_path,
        date_col=text_date_col,
        prefix="report",
        join_key=join_key,
        group_by_month=group_by_month,
    )

    merged = numeric.merge(search, on=join_key, how="left").merge(
        report, on=join_key, how="left"
    )
    merged = merged.drop(columns=[join_key])

    text_cols = [c for c in merged.columns if c.startswith("search_") or c.startswith("report_")]
    if text_cols:
        merged[text_cols] = merged[text_cols].fillna("NA")
    return merged


def _merge_economy_two_stage(
    *,
    numeric_path: Path,
    search_path: Path,
    report_path: Path,
    numeric_date_col: str,
    report_date_col: str,
    search_date_col: str,
) -> pd.DataFrame:
    numeric = pd.read_csv(numeric_path)
    if numeric_date_col not in numeric.columns:
        raise ValueError(f"Missing date column '{numeric_date_col}' in {numeric_path}")
    numeric[numeric_date_col] = pd.to_datetime(numeric[numeric_date_col])
    numeric["__month__"] = numeric[numeric_date_col].dt.to_period("M").astype(str)

    report = _read_text_csv_by_month(
        report_path,
        date_col=report_date_col,
        prefix="report",
        month_shift=0,
    )
    stage1 = numeric.merge(report, on="__month__", how="left")

    search = _read_text_csv_by_month(
        search_path,
        date_col=search_date_col,
        prefix="search",
        month_shift=-1,
    )
    merged = stage1.merge(search, on="__month__", how="left").drop(columns=["__month__"])

    text_cols = [c for c in merged.columns if c.startswith("search_") or c.startswith("report_")]
    if text_cols:
        merged[text_cols] = merged[text_cols].fillna("NA")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Join numeric + text CSVs by date.")
    parser.add_argument(
        "--numeric",
        default="Time-MMD/numerical/Climate/Climate.csv",
        help="Path to numeric Climate CSV.",
    )
    parser.add_argument(
        "--search",
        default="Time-MMD/textual/Climate/Climate_search.csv",
        help="Path to search-summary text CSV.",
    )
    parser.add_argument(
        "--report",
        default="Time-MMD/textual/Climate/Climate_report.csv",
        help="Path to report-summary text CSV.",
    )
    parser.add_argument(
        "--numeric-date-col",
        default="start_date",
        help="Date column in the numeric CSV.",
    )
    parser.add_argument(
        "--text-date-col",
        default="start_date",
        help="Date column in the text CSVs.",
    )
    parser.add_argument(
        "--report-date-col",
        default="start_date",
        help="Date column in the report CSV (economy two-stage).",
    )
    parser.add_argument(
        "--search-date-col",
        default="end_date",
        help="Date column in the search CSV (economy two-stage).",
    )
    parser.add_argument(
        "--group-by-month",
        action="store_true",
        help="Group text rows by the month of start_date before joining.",
    )
    parser.add_argument(
        "--strategy",
        choices=["simple", "economy-two-stage"],
        default="simple",
        help="Join strategy to use.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="Time-MMD/numerical/Climate/Climate_with_text.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.strategy == "economy-two-stage":
        merged = _merge_economy_two_stage(
            numeric_path=Path(args.numeric),
            search_path=Path(args.search),
            report_path=Path(args.report),
            numeric_date_col=args.numeric_date_col,
            report_date_col=args.report_date_col,
            search_date_col=args.search_date_col,
        )
    else:
        merged = _merge_climate(
            numeric_path=Path(args.numeric),
            search_path=Path(args.search),
            report_path=Path(args.report),
            numeric_date_col=args.numeric_date_col,
            text_date_col=args.text_date_col,
            group_by_month=args.group_by_month,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Wrote {len(merged)} rows to {output_path}")


if __name__ == "__main__":
    main()
