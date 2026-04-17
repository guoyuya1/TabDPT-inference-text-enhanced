"""Dedicated lag search entrypoint for TabDPT baseline."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from tabdpt_tabpfn import (
    build_lagged_arrays,
    evaluate_tabdpt_lag_combo,
    load_dataframe,
    parse_cfg,
    parse_lag_values,
    read_cfg,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tabular.yaml")
    parser.add_argument("--dataset")
    parser.add_argument(
        "--target-lag-range",
        type=int,
        nargs="+",
        required=True,
        help="Target lag days to try, e.g. --target-lag-range 0 1 2 3",
    )
    parser.add_argument(
        "--covariate-lag-range",
        type=int,
        nargs="+",
        required=True,
        help="Covariate lag days to try, e.g. --covariate-lag-range 1 3 5 7",
    )
    parser.add_argument("--tabpfn-model-path")
    parser.add_argument("--tabpfn-cache-dir")
    args = parser.parse_args()

    target_lags = parse_lag_values(args.target_lag_range)
    covariate_lags = parse_lag_values(args.covariate_lag_range)

    config_file, raw_cfgs = read_cfg(args.config, args.dataset)
    print(f"config={config_file}")
    print(f"target_lag_range={target_lags}")
    print(f"covariate_lag_range={covariate_lags}")

    for dataset_name, raw_cfg in raw_cfgs.items():
        cfg = parse_cfg(raw_cfg, args)
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        df = load_dataframe(cfg)

        print(f"\n{'=' * 16} Dataset: {dataset_name} {'=' * 16}")
        best_combo: tuple[int, int] | None = None
        best_mae = float("inf")
        trial_results: list[dict[str, float | int]] = []

        for target_lag in target_lags:
            for covariate_lag in covariate_lags:
                print(f"trying target_lag_days={target_lag}, covariate_lag_days={covariate_lag}")
                try:
                    X_rows, y_targets = build_lagged_arrays(
                        df,
                        cfg,
                        target_lag_days=target_lag,
                        covariate_lag_days=covariate_lag,
                    )
                    mean_mae, _ = evaluate_tabdpt_lag_combo(cfg, X_rows, y_targets)
                    print(f"mean_val_tabdpt_mae={mean_mae:.6f}")
                    trial_results.append(
                        {
                            "target_lag_days": target_lag,
                            "covariate_lag_days": covariate_lag,
                            "mean_val_tabdpt_mae": mean_mae,
                        }
                    )
                    if mean_mae < best_mae:
                        best_mae = mean_mae
                        best_combo = (target_lag, covariate_lag)
                except Exception as exc:
                    print(f"skipping invalid lag combo: {exc}")

        if best_combo is None:
            raise RuntimeError(f"No valid lag combination found for dataset '{dataset_name}'.")

        best_target_lag, best_covariate_lag = best_combo
        print(f"\nBest for {dataset_name}")
        print(f"target_lag_days={best_target_lag}")
        print(f"covariate_lag_days={best_covariate_lag}")
        print(f"mean_val_tabdpt_mae={best_mae:.6f}")
        print(f"tested_combinations={len(trial_results)}")


if __name__ == "__main__":
    main()
