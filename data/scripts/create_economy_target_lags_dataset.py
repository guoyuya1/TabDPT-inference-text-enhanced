"""
Create a dataset with target lags and selected embedding columns for the economy CSV.
"""

from __future__ import annotations

import pandas as pd

input_csv_path = "./data/economy/Economy_with_text_with_embeddings_lag_3.csv"
output_csv_path = "./data/economy/Economy_with_text_with_embeddings_lag_3_target_lags.csv"

target_col = "OT"
target_lags = 5

embedding_columns = [
    "embedding_report_fact_lag3",
    "embedding_report_fact_lag2",
    "embedding_report_fact_lag1",
    "embedding_report_fact_lag0",
]


def main() -> None:
    df = pd.read_csv(input_csv_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_csv_path}.")

    missing_embeddings = [col for col in embedding_columns if col not in df.columns]
    if missing_embeddings:
        raise ValueError(
            f"Missing embedding columns in {input_csv_path}: {missing_embeddings}"
        )

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    for lag in range(1, target_lags + 1):
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)

    output_columns = []
    if "date" in df.columns:
        output_columns.append("date")
    output_columns += [target_col] + [f"{target_col}_lag{i}" for i in range(1, target_lags + 1)]
    output_columns += embedding_columns

    df_out = df[output_columns]
    df_out.to_csv(output_csv_path, index=False)
    print(f"Wrote dataset to {output_csv_path} with columns: {output_columns}")


if __name__ == "__main__":
    main()
