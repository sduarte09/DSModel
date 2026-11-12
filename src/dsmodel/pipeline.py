"""Downscaling of one station: calibration, validation and application."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .models import fit_predict


@dataclass
class DownscaleResult:
    series: pd.Series  # observed until ``cal_end``, predicted afterwards, on the model's daily index
    method: str
    best_params: dict
    n_calibration: int
    validation: dict = field(default_factory=dict)  # metrics over (cal_end, val_end]; empty if not requested


def metrics(predicted, observed):
    """RMSE, MAE, bias (mean of predicted - observed), correlation and count over the common valid days."""
    p, o = np.asarray(predicted, dtype=float), np.asarray(observed, dtype=float)
    ok = ~(np.isnan(p) | np.isnan(o))
    p, o = p[ok], o[ok]
    if p.size == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "correlation": np.nan}
    corr = float(np.corrcoef(p, o)[0, 1]) if p.size > 1 and p.std() > 0 and o.std() > 0 else np.nan
    return {"n": int(p.size), "rmse": float(np.sqrt(np.mean((p - o) ** 2))), "mae": float(np.mean(np.abs(p - o))),
            "bias": float(np.mean(p - o)), "correlation": corr}


def downscale_station(station, model, cal_end, app_end, method="knn", val_end: Optional[str] = None,
                      clip_negative=False, **fit_kwargs):
    """Downscale one station with one method.

    Parameters
    ----------
    station : daily ``pandas.Series`` of observations (NaN or missing days allowed)
    model : daily ``pandas.Series`` of the climate-model cell, same units as the station
    cal_end : last day of the calibration period
    app_end : last day of the series to produce
    method : ``"knn"``, ``"forest"``, ``"linear"`` or ``"mlp"`` (see :mod:`dsmodel.models`)
    val_end : if given, the days in ``(cal_end, val_end]`` that have observations are scored
    clip_negative : set negative predictions to zero (useful for precipitation)
    fit_kwargs : passed to :func:`dsmodel.models.build_search` (``cv``, ``param_grid``, ``random_state``...)

    The model is fitted once on the days up to ``cal_end`` where both series have a value
    and is applied to every model day in ``(cal_end, app_end]``. The result keeps the
    observations up to ``cal_end``, as the legacy code did.
    """
    station, model = station.dropna(), model.dropna()
    if station.empty:
        raise ValueError("The station has no data")
    cal_end, app_end = pd.Timestamp(cal_end), pd.Timestamp(app_end)
    val_end = pd.Timestamp(val_end) if val_end is not None else None
    if app_end <= cal_end or (val_end is not None and not cal_end < val_end <= app_end):
        raise ValueError("Dates must satisfy cal_end < val_end <= app_end")

    start = station.index[0]
    paired = pd.concat([model.rename("model"), station.rename("station")], axis=1, join="inner")
    paired = paired.loc[start:cal_end]
    if paired.empty:
        raise ValueError("Station and model have no common days in the calibration period")
    new_dates = model.index[(model.index > cal_end) & (model.index <= app_end)]
    if new_dates.empty:
        raise ValueError("The model has no days after cal_end")

    predicted, search = fit_predict(method, paired["model"].values, paired["station"].values,
                                    model.loc[new_dates].values, **fit_kwargs)
    if clip_negative:
        predicted = np.clip(predicted, 0, None)

    observed_cal = station.loc[start:cal_end].reindex(model.loc[start:cal_end].index)
    series = pd.concat([observed_cal, pd.Series(predicted, index=new_dates)]).rename(station.name or "station")

    scores = {}
    if val_end is not None:
        in_val = (new_dates <= val_end)
        pred_val = pd.Series(predicted[in_val], index=new_dates[in_val])
        scores = metrics(pred_val.values, station.reindex(pred_val.index).values)
    return DownscaleResult(series, method, dict(search.best_params_), len(paired), scores)
