from __future__ import annotations

import numpy as np


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    text: np.ndarray | None,
    *,
    context_ratio: float,
    tune_ratio: float,
    eval_ratio: float,
) -> tuple[np.ndarray | None, ...]:
    """
    Split sequentially into context / tune / eval based on ratios.

    Ratios must sum to 1.0 and are applied in order.
    """
    ratio_sum = context_ratio + tune_ratio + eval_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}")
    if context_ratio <= 0 or tune_ratio <= 0 or eval_ratio <= 0:
        raise ValueError("Split ratios must be positive.")

    n = len(y)
    n_context = int(n * context_ratio)
    n_tune = int(n * tune_ratio)
    n_eval = n - n_context - n_tune
    if n_context <= 0 or n_tune <= 0 or n_eval <= 0:
        raise ValueError(f"Invalid split sizes: n={n}, context={n_context}, tune={n_tune}, eval={n_eval}")

    if text is None:
        text_context = text_tune = text_eval = None
    else:
        text_context = text[:n_context]
        text_tune = text[n_context : n_context + n_tune]
        text_eval = text[n_context + n_tune :]

    X_context, y_context = X[:n_context], y[:n_context]
    X_tune, y_tune = X[n_context : n_context + n_tune], y[n_context : n_context + n_tune]
    X_eval, y_eval = X[n_context + n_tune :], y[n_context + n_tune :]

    return (
        X_context,
        y_context,
        text_context,
        X_tune,
        y_tune,
        text_tune,
        X_eval,
        y_eval,
        text_eval,
    )
