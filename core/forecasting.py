"""Lightweight, transparent forecasting: linear trend (+ seasonality when enough history).

Deliberately simple and explainable; it reports its method, uncertainty band and caveats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FREQ = {"day": ("D", 7), "week": ("W", 52), "month": ("MS", 12), "quarter": ("QS", 4), "year": ("YS", 1)}


def forecast_series(dates, values, periods: int = 6, frequency: str = "month") -> dict:
    if frequency not in FREQ:
        raise ValueError(f"frequency must be one of {list(FREQ)}")
    periods = int(max(1, min(periods, 36)))
    alias, season = FREQ[frequency]

    s = pd.Series(pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(),
                  index=pd.to_datetime(pd.Series(list(dates)), errors="coerce"))
    s = s[s.index.notna()].dropna().sort_index()
    if s.empty:
        raise ValueError("No valid (date, value) pairs to forecast from.")
    s = s.groupby(pd.Grouper(freq=alias)).sum()
    y = s.to_numpy(dtype=float)
    n = len(y)
    if n < 6:
        raise ValueError(f"Need at least 6 {frequency}s of history to forecast; got {n}.")

    t = np.arange(n)
    slope, intercept = np.polyfit(t, y, 1)
    trend = slope * t + intercept
    use_seasonal = season > 1 and n >= 2 * season
    seas = np.zeros(max(season, 1))
    if use_seasonal:
        resid = y - trend
        seas = np.array([resid[t % season == k].mean() for k in range(season)])
        seas -= seas.mean()
    fitted = trend + (seas[t % season] if use_seasonal else 0)
    sigma = float(np.std(y - fitted, ddof=2)) if n > 2 else 0.0

    tf = np.arange(n, n + periods)
    pred = slope * tf + intercept + (seas[tf % season] if use_seasonal else 0)
    future_idx = pd.date_range(s.index[-1], periods=periods + 1, freq=alias)[1:]
    fmt = "%Y-%m" if frequency in ("month", "quarter") else "%Y-%m-%d"

    return {
        "method": "linear trend" + (f" + {season}-period seasonality" if use_seasonal else ""),
        "frequency": frequency,
        "history": [{"period": d.strftime(fmt), "value": round(float(v), 2)} for d, v in zip(s.index, y)],
        "forecast": [
            {"period": d.strftime(fmt), "value": round(float(p), 2),
             "lower": round(float(p - 1.96 * sigma), 2), "upper": round(float(p + 1.96 * sigma), 2)}
            for d, p in zip(future_idx, pred)
        ],
        "trend_per_period": round(float(slope), 2),
        "direction": "upward" if slope > 0 else "downward" if slope < 0 else "flat",
        "residual_std": round(sigma, 2),
        "caveat": "Statistical extrapolation of past patterns; it cannot anticipate promotions, "
                  "market shocks or other external events. Treat the band as a rough 95% range.",
    }
