import pandas as pd
import numpy as np
from scipy.signal import detrend, convolve, windows, find_peaks
import logging
import torch

logger = logging.getLogger()

MEDIAN_QUANTILE = 0.5
DEFAULT_FREQ = 10_000
DEFAULT_SEAS_SMOOTHING_WINDOW = 7

FREQ_MAP = {
    "second": "S",
    "minute": "T",
    "hour": "H",
    "day": "D",
    "week": "W",
    "month": "M",
    "quarter": "Q",
    "year": "Y",
}


def map_freq_gluonts(freq: str):
    if freq not in FREQ_MAP:
        raise ValueError(f"Frequency {freq} is not supported.")
    return FREQ_MAP[freq]


def extend_future(df, timestamp_col, horizon, freq=None):
    """extends the dataframe's timestamp_col into the future by
    the horizon, generating a new dataframe
    """
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    if freq is None:
        freq = pd.infer_freq(df[timestamp_col].sort_values())
        if freq is None:
            raise ValueError("Could not infer frequency; please specify `freq` manually.")

    last_ts = df[timestamp_col].max()
    future_dates = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]

    return pd.DataFrame({timestamp_col: future_dates})


def generate_time_features(
    target_values: np.array, timestamps: pd.Series, frequency: str,
    k: int, L: int = None, use_index: bool = True
):
    calendar_features = generate_calendar_features(timestamps, frequency)
    seasonal_features = generate_seasonality_features(target_values, k, L)[-len(timestamps):]
    if use_index:
        index = np.arange(len(target_values))[-len(timestamps):, np.newaxis]
        all_timestamp_features = np.concatenate([calendar_features, seasonal_features, index], axis=-1)
    else:
        all_timestamp_features = np.concatenate([calendar_features, seasonal_features], axis=-1)

    logger.debug("***** Temporal Features: %s", all_timestamp_features.shape)
    return all_timestamp_features


def generate_seasonality_features(target_values, k: int, L: int = None):
    if k == 0:
        return np.empty((len(target_values), 0))

    top_k_periods = extract_top_k_seasonalities(
        target_values, k=k, L=L if L is not None else DEFAULT_SEAS_SMOOTHING_WINDOW
    )

    ts_features = []
    for period in top_k_periods:
        if period is None:
            ts_features += [np.nan, np.nan]
            continue

        ts_features.append(np.sin(2 * np.pi / period))
        ts_features.append(np.cos(2 * np.pi / period))

    # The above method extracts top k seasonalities given the whole context as input,
    # and then repeats it for each row in the context
    # TODO potentially improve above. For instance,
    # below is an attempt to extract seasonalities for each row given the context/input up to that row.
    # However, it would not work for the beginning rows with very limited context.
    # top_k_periods_list = [extract_top_k_seasonalities(
    #     target_values[:i], k=k, L=L if L is not None else DEFAULT_SEAS_SMOOTHING_WINDOW
    # ) for i in range(len(target_values))]
    # ts_features = []
    # for top_k_periods in top_k_periods_list:
    #     print('top_k_periods = ', top_k_periods)
    #     if len(top_k_periods) == 0 or not all(top_k_periods):
    #         ts_features.append([None]*2*k)
    #         continue
    #     ts_features.append([])
    #     for period in top_k_periods:
    #         ts_features[-1].append(np.sin(2 * np.pi / period))
    #         ts_features[-1].append(np.cos(2 * np.pi / period))

    # Append NaNs if top_k_periods has less than k periods
    while len(ts_features) < 2 * k:
        ts_features.append(np.nan)

    return np.tile(np.asarray(ts_features, dtype=float), (len(target_values), 1))


def generate_causal_seasonality_features(target_values, k: int, L: int = None):
    """
    Compute seasonality features causally for each row using only history up to that row.

    Row i is derived from target_values[: i + 1], so no future target information is used.
    """
    if k == 0:
        return np.empty((len(target_values), 0))

    values = np.asarray(target_values, dtype=float)
    features = np.empty((len(values), 2 * k), dtype=float)
    window = L if L is not None else DEFAULT_SEAS_SMOOTHING_WINDOW
    for idx in range(len(values)):
        prefix = values[: idx + 1]
        periods = extract_top_k_seasonalities(prefix, k=k, L=window)
        row: list[float] = []
        for period in periods:
            if period is None:
                row.extend([np.nan, np.nan])
                continue
            row.extend([np.sin(2 * np.pi / period), np.cos(2 * np.pi / period)])
        while len(row) < 2 * k:
            row.append(np.nan)
        features[idx] = np.asarray(row, dtype=float)
    return features


def generate_calendar_features(time_stamp, frequency):
    # Normalize frequency aliases
    frequency_map = {
        "monthly": "month",
        "daily": "day",
        "weekly": "week",
        "hourly": "hour",
        "quarterly": "quarter",
    }

    frequency = frequency_map.get(frequency, frequency)

    calendar_features = []
    frequencies = ["hour", "day", "week", "month", "quarter"]

    if frequency in frequencies[1:]:
        calendar_features += [("hour_of_day", 24)]

    if frequency in frequencies[2:]:
        calendar_features += [("dayofweek", 7), ("day", 31), ("dayofyear", 365)]

    if frequency in frequencies[3:]:
        calendar_features += [("weekofyear", 52)]

    if frequency in frequencies[4:]:
        calendar_features += [("month", 12)]

    if frequency in frequencies[5:]:
        calendar_features += [("quarter", 4)]

    logger.debug("***** calendar features = %s", calendar_features)
    ts_features = []

    
    dt_attr = {"hour_of_day": "hour"}
    for feature_name, seasonality in calendar_features:
        if feature_name == "weekofyear":
            feature = time_stamp.dt.isocalendar().week
        else:
            attr = dt_attr.get(feature_name, feature_name)
            feature = getattr(time_stamp.dt, attr)
        ts_features.append(np.sin(2 * np.pi * feature / seasonality))
        ts_features.append(np.cos(2 * np.pi * feature / seasonality))


    return np.stack(ts_features, axis=-1)


def extract_top_k_seasonalities(series, k, L):
    """
    Automatically extracting top k seasonalities from the time series.
    Adopted from the method described in "From Tables to Time: Extending TabPFN v2 to Time Series Forecasting"
    [https://arxiv.org/abs/2501.02945v3] by Frank Hutter's team.

    Parameters
    - series: pd.Series or np.array, the univariate time series.
    - k: int, the number of top periods to return.
    - L: int, the smoothing window size for Hann window.

    Returns:
    - List of the top k identified periods (as integers)
    """

    series = pd.Series(series).ffill()
    series = series.dropna()
    x = np.array(series, dtype=float)
    N = len(x)

    if k == 0:
        return []
    if N == 0:
        return [None] * k

    # 1. Detrend linearly: series[t] = series[t] - (at + b)
    # scipy.signal.detrend uses least squares by default
    x_detrended = detrend(x, type="linear")

    # 2. Apply Hann window to the detrended series
    hann_win = windows.hann(L)
    # Normalize windows to preserve signal scale
    hann_win /= hann_win.sum()
    x_smoothed = convolve(x_detrended, hann_win, mode="same")

    # 3. Double length by symm. zero-padding: [0..0, series, 0..0]
    # To reach length 2*N, we add N/2 zeros on each side
    pad_before = N // 2
    pad_after = N - pad_before
    x_padded = np.pad(x_smoothed, (pad_before, pad_after), mode="constant", constant_values=0)

    # 4. Fourier Transform
    n_fft = len(x_padded)
    fft_res = np.fft.fft(x_padded)
    mags = np.abs(fft_res)
    freqs = np.fft.fftfreq(n_fft)  # frequencies in cycles per sample
    # 5. remove the DC component
    mags[0] = 0

    # 6. find the peaks
    peak_indices, _ = find_peaks(mags)

    # 7. Inverse frequencies to periods and convert to integers
    with np.errstate(divide="ignore", invalid="ignore"):
        all_periods = np.floor(1.0 / freqs)

    # 8. Remove duplicate and zero periods from the peak indices
    # if multiple peak frequencies map to the same integer period, we keep the highest magnitude
    period_to_best_peak = {}
    for idx in peak_indices:
        p = all_periods[idx]
        if p > 0 and np.isfinite(p):
            if p not in period_to_best_peak or mags[idx] > mags[period_to_best_peak[p]]:
                period_to_best_peak[p] = idx

    unique_peak_indices = list(period_to_best_peak.values())

    # 9. Keep only top k peaks based on magnitude
    top_k_indices = sorted(unique_peak_indices, key=lambda idx: mags[idx], reverse=True)[:k]
    

    return [int(all_periods[idx]) for idx in top_k_indices]


def apply_rope(q, k, q_length, k_length):
    # q, k: (batch_size, num_heads, context_length, head_dim)

    device = q.device
    dim = q.shape[-1]

    # frequencies
    theta = DEFAULT_FREQ ** (-torch.arange(0, dim, 2, device=device) / dim)

    def rotate(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).reshape_as(x)

    def apply_rotary(x, seq_len):
        pos = torch.arange(seq_len, device=device)
        freqs = torch.einsum("t,d->td", pos, theta)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        return (x * cos.repeat_interleave(2, dim=-1)) + (rotate(x) * sin.repeat_interleave(2, dim=-1))

    q = apply_rotary(q, q_length)
    k = apply_rotary(k, k_length)

    return q, k
