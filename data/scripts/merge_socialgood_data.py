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
    start_date_col: str,
    end_date_col: str,
    prefix: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    if start_date_col not in df.columns:
        raise ValueError(f"Missing start date column '{start_date_col}' in {path}")
    if end_date_col not in df.columns:
        raise ValueError(f"Missing end date column '{end_date_col}' in {path}")

    df[start_date_col] = pd.to_datetime(df[start_date_col])
    df[end_date_col] = pd.to_datetime(df[end_date_col])

    rename_map = {}
    if "fact" in df.columns:
        rename_map["fact"] = f"{prefix}_fact"
    df = df.rename(columns=rename_map)

    keep_cols = [start_date_col, end_date_col]
    fact_col = f"{prefix}_fact"
    if fact_col in df.columns:
        keep_cols.append(fact_col)

    return df[keep_cols].copy()


def _collect_overlapping_text(
    *,
    numeric_start: pd.Timestamp,
    numeric_end: pd.Timestamp,
    text_df: pd.DataFrame,
    text_start_col: str,
    text_end_col: str,
    text_cols: list[str],
) -> dict[str, str]:
    overlap_mask = (
        (text_df[text_start_col] <= numeric_end)
        & (text_df[text_end_col] >= numeric_start)
    )
    matched = text_df.loc[overlap_mask, text_cols]

    if matched.empty:
        return {col: "NA" for col in text_cols}

    return {col: _concat_text(matched[col]) for col in text_cols}


def _merge_with_interval_overlap(
    *,
    numeric_path: Path,
    search_path: Path,
    report_path: Path,
    numeric_start_date_col: str,
    numeric_end_date_col: str,
    text_start_date_col: str,
    text_end_date_col: str,
    min_numeric_start_date: str | None = None,
    max_numeric_start_date: str | None = None,
) -> pd.DataFrame:
    numeric = pd.read_csv(numeric_path)
    numeric = numeric.loc[:, ~numeric.columns.str.contains(r"^Unnamed")]

    if numeric_start_date_col not in numeric.columns:
        raise ValueError(
            f"Missing numeric start date column '{numeric_start_date_col}' in {numeric_path}"
        )
    if numeric_end_date_col not in numeric.columns:
        raise ValueError(
            f"Missing numeric end date column '{numeric_end_date_col}' in {numeric_path}"
        )

    numeric[numeric_start_date_col] = pd.to_datetime(numeric[numeric_start_date_col])
    numeric[numeric_end_date_col] = pd.to_datetime(numeric[numeric_end_date_col])

    if min_numeric_start_date is not None:
        min_numeric_start_date = pd.to_datetime(min_numeric_start_date)
        numeric = numeric[numeric[numeric_start_date_col] > min_numeric_start_date].copy()

    if max_numeric_start_date is not None:
        max_numeric_start_date = pd.to_datetime(max_numeric_start_date)
        numeric = numeric[numeric[numeric_start_date_col] < max_numeric_start_date].copy()

    search = _read_text_csv(
        search_path,
        start_date_col=text_start_date_col,
        end_date_col=text_end_date_col,
        prefix="search",
    )
    report = _read_text_csv(
        report_path,
        start_date_col=text_start_date_col,
        end_date_col=text_end_date_col,
        prefix="report",
    )

    search_text_cols = [c for c in search.columns if c.startswith("search_")]
    report_text_cols = [c for c in report.columns if c.startswith("report_")]

    search_rows = []
    report_rows = []

    for _, row in numeric.iterrows():
        numeric_start = row[numeric_start_date_col]
        numeric_end = row[numeric_end_date_col]

        search_rows.append(
            _collect_overlapping_text(
                numeric_start=numeric_start,
                numeric_end=numeric_end,
                text_df=search,
                text_start_col=text_start_date_col,
                text_end_col=text_end_date_col,
                text_cols=search_text_cols,
            )
        )
        report_rows.append(
            _collect_overlapping_text(
                numeric_start=numeric_start,
                numeric_end=numeric_end,
                text_df=report,
                text_start_col=text_start_date_col,
                text_end_col=text_end_date_col,
                text_cols=report_text_cols,
            )
        )

    search_aligned = pd.DataFrame(search_rows)
    report_aligned = pd.DataFrame(report_rows)

    merged = pd.concat(
        [numeric.reset_index(drop=True), search_aligned, report_aligned],
        axis=1,
    )

    text_cols = [c for c in merged.columns if c.startswith("search_") or c.startswith("report_")]
    if text_cols:
        merged[text_cols] = merged[text_cols].fillna("NA")

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join numeric + text CSVs by interval overlap."
    )
    parser.add_argument(
        "--min-numeric-start-date",
        default="2000-01-01",
        help="Keep only numeric rows with start date greater than this value.",
    )
    parser.add_argument(
        "--max-numeric-start-date",
        default="2023-12-31",
        help="Keep only numeric rows with start date less than this value.",
    )
    parser.add_argument(
        "--numeric",
        default="Time-MMD/numerical/SocialGood/SocialGood.csv",
        help="Path to numeric CSV.",
    )
    parser.add_argument(
        "--search",
        default="Time-MMD/textual/SocialGood/SocialGood_search.csv",
        help="Path to search-summary text CSV.",
    )
    parser.add_argument(
        "--report",
        default="Time-MMD/textual/SocialGood/SocialGood_report.csv",
        help="Path to report-summary text CSV.",
    )
    parser.add_argument(
        "--numeric-start-date-col",
        default="start_date",
        help="Start date column in the numeric CSV.",
    )
    parser.add_argument(
        "--numeric-end-date-col",
        default="end_date",
        help="End date column in the numeric CSV.",
    )
    parser.add_argument(
        "--text-start-date-col",
        default="start_date",
        help="Start date column in the text CSVs.",
    )
    parser.add_argument(
        "--text-end-date-col",
        default="end_date",
        help="End date column in the text CSVs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="Time-MMD/numerical/SocialGood/SocialGood_with_text.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    merged = _merge_with_interval_overlap(
        numeric_path=Path(args.numeric),
        search_path=Path(args.search),
        report_path=Path(args.report),
        numeric_start_date_col=args.numeric_start_date_col,
        numeric_end_date_col=args.numeric_end_date_col,
        text_start_date_col=args.text_start_date_col,
        text_end_date_col=args.text_end_date_col,
        min_numeric_start_date=args.min_numeric_start_date,
        max_numeric_start_date=args.max_numeric_start_date,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Wrote {len(merged)} rows to {output_path}")


if __name__ == "__main__":
    main()