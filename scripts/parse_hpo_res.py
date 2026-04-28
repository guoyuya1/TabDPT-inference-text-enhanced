import re
import pandas as pd

log_path = "/home/yuyan/tabdpt_mz/results/hpo/agriculture_v3_analyzed_time_features_random_search_20260426_025112/summary.log"  # change to your file path

rows = []
current_set = None
current_set_params = None
current_horizon = None

def parse_value(v: str):
    """Convert string values from the log into useful Python types."""
    v = v.strip()

    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False

    # Strip percentage signs if present
    v_no_pct = v.rstrip("%")

    try:
        if "." in v_no_pct or "e" in v_no_pct.lower():
            return float(v_no_pct)
        return int(v_no_pct)
    except ValueError:
        return v

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Example:
        # Shared parameter set 1: Used params | text_attn_layers=[1, 2] | ...
        m_set = re.match(r"^Shared parameter set\s+(\d+):\s*(.*)$", line)
        if m_set:
            current_set = int(m_set.group(1))
            current_set_params = m_set.group(2)
            current_horizon = None
            continue

        # Skip shared baseline section
        if current_set is None:
            continue

        # Example:
        # Model horizon 1:
        m_horizon = re.match(r"^Model horizon\s+(\d+):", line)
        if m_horizon:
            current_horizon = int(m_horizon.group(1))
            continue

        # Candidate rows start with tuning_batch_size
        if not line.startswith("tuning_batch_size"):
            continue

        if current_horizon is None:
            continue

        parts = [p.strip() for p in line.split("|")]

        row = {
            "shared_param_set": current_set,
            "shared_param_params": current_set_params,
            "horizon": current_horizon,
        }

        # Parse tuning batch size from first segment:
        # tuning_batch_size 1/8
        m_batch = re.match(r"tuning_batch_size\s+(\d+)/(\d+)", parts[0])
        if m_batch:
            row["tuning_batch_size_index"] = int(m_batch.group(1))
            row["tuning_batch_size"] = int(m_batch.group(2))

        # Parse all key=value fields
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key.strip()] = parse_value(value)

        rows.append(row)

df = pd.DataFrame(rows)

# Make sure required column exists
required_cols = [
    "shared_param_set",
    "horizon",
    "candidate_index",
    "tuning_batch_size",
    "best_epoch",
    "before_val_real_mae",
    "after_val_real_mae",
    "before_test_real_mae",
    "after_test_real_mae",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Find the best candidate per shared parameter set + horizon
idx = (
    df.groupby(["shared_param_set", "horizon"])["after_test_real_mae"]
      .idxmin()
)

best_per_set_horizon = (
    df.loc[idx, required_cols + ["shared_param_params"]]
      .sort_values(["shared_param_set", "horizon"])
      .reset_index(drop=True)
)

print(best_per_set_horizon)

# Optional: save result
best_per_set_horizon.to_csv(
    "best_after_test_real_mae_per_shared_param_set_horizon.csv",
    index=False
)

# Optional: average after_test_real_mae across horizons for each shared parameter set
avg_by_set = (
    best_per_set_horizon
    .groupby("shared_param_set", as_index=False)
    .agg(
        avg_after_test_real_mae=("after_test_real_mae", "mean"),
        horizons_found=("horizon", "nunique"),
    )
    .sort_values("avg_after_test_real_mae")
)

print(avg_by_set)