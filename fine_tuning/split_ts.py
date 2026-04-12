from __future__ import annotations

import numpy as np


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    text: np.ndarray,
    *,
    context_ratio: float,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[np.ndarray, ...]:
    """
    Split sequentially into context / train / val / test based on ratios.

    Ratios must sum to 1.0 and are applied in order.
    """
    ratio_sum = context_ratio + train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}")
    if context_ratio <= 0 or train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("Split ratios must be positive.")

    n = len(y)
    n_context = int(n * context_ratio)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_context - n_train - n_val
    if n_context <= 0 or n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes: n={n}, context={n_context}, train={n_train}, val={n_val}, test={n_test}"
        )

    X_context, y_context, text_context = X[:n_context], y[:n_context], text[:n_context]
    train_start = n_context
    train_end = train_start + n_train
    val_end = train_end + n_val

    X_train, y_train, text_train = (
        X[train_start:train_end],
        y[train_start:train_end],
        text[train_start:train_end],
    )
    X_val, y_val, text_val = (
        X[train_end:val_end],
        y[train_end:val_end],
        text[train_end:val_end],
    )
    X_test, y_test, text_test = X[val_end:], y[val_end:], text[val_end:]

    return (
        X_context,
        y_context,
        text_context,
        X_train,
        y_train,
        text_train,
        X_val,
        y_val,
        text_val,
        X_test,
        y_test,
        text_test,
    )
