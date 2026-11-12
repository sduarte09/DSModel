"""Run the downscaling for every station of a catalogue and write CSV files and plots.

This is the Python 3 version of ``climatic_statistcal_downscaling`` in the legacy
``Statistical Downscaling Models.py``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import io
from .pipeline import downscale_station

DEFAULT_METHODS = ("knn", "forest", "linear")


@dataclass
class RunResult:
    predictions: dict  # method -> DataFrame (date index, one column per station)
    validation: pd.DataFrame  # one row per station and method; empty without val_date
    skipped: dict = field(default_factory=dict)  # station ID -> reason
    model_id: str = ""
    experiment_id: str = ""
    variable: str = ""
    files: list = field(default_factory=list)


def _plot_annual(series, variable, path, title):
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    annual = series.resample("YE").sum() if variable == "pr" else series.resample("YE").mean()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    annual.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel("mm/year" if variable == "pr" else "degrees C")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_catalog(station_data, station_catalog, model_historic, model_rcp, output_folder, cal_date, app_date,
                val_date=None, methods=DEFAULT_METHODS, variable=None, plots=True, **fit_kwargs):
    """Downscale all stations of a catalogue with several methods.

    Parameters
    ----------
    station_data : CSV/Excel with a ``Date`` column (dd/mm/yyyy) and one column per station ID
    station_catalog : CSV/Excel with station ``ID``, ``Latitud`` and ``Longitud`` (English names also work)
    model_historic, model_rcp : folders with the NetCDF files of the historic and scenario experiments
    output_folder : where CSV files and plots are written (created if missing)
    cal_date, app_date : last day of calibration and of the application
    val_date : optional last day of validation; adds a metrics table
    methods : any of ``knn``, ``forest``, ``linear``, ``mlp``
    variable : ``pr``, ``tas``, ``tasmin`` or ``tasmax``; detected from the files when omitted
    plots : also write one annual plot per station and method
    fit_kwargs : passed to the model search (``cv``, ``param_grid``, ``random_state``...)
    """
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    table = io.read_station_table(station_data)
    catalog = io.read_catalog(station_catalog)
    his_files, rcp_files = io.list_netcdf(model_historic), io.list_netcdf(model_rcp)
    variable = variable or io.detect_variable(rcp_files[0])
    info = io.model_info(rcp_files[0])
    tag = f"{info['model_id']}_{variable}_{info['experiment_id']}"
    lat, lon = io.read_grid(his_files[0])

    start = table.index[0]
    index = pd.date_range(start, app_date)
    predictions = {m: pd.DataFrame(index=index) for m in methods}
    rows, skipped, files = [], {}, []
    model_cache: dict = {}

    for station_id, point in catalog.iterrows():
        if station_id not in table.columns:
            skipped[station_id] = "no column in the station table"
            continue
        try:
            cell = io.locate_cell(lat, lon, point["latitude"], point["longitude"])
        except ValueError as err:
            skipped[station_id] = str(err)
            continue
        if cell not in model_cache:  # stations in the same model cell share one series
            model_cache[cell] = io.concat_model(io.read_cell_series(his_files, *cell, variable),
                                                io.read_cell_series(rcp_files, *cell, variable))
        model = model_cache[cell]
        obs = table[station_id].dropna().rename(station_id)
        for method in methods:
            try:
                res = downscale_station(obs, model, cal_date, app_date, method=method, val_end=val_date,
                                        clip_negative=(variable == "pr"), **fit_kwargs)
            except ValueError as err:
                skipped[f"{station_id}/{method}"] = str(err)
                continue
            predictions[method][station_id] = res.series.reindex(index)
            if val_date is not None:
                rows.append({"station": station_id, "method": method, **res.validation})
            if plots:
                png = out / f"{station_id}_{method}_{tag}.png"
                _plot_annual(res.series, variable, png, f"{station_id} - {method}")
                files.append(png)

    for method, frame in predictions.items():
        path = out / f"{method}_{tag}.csv"
        frame.to_csv(path, index_label="Index")
        files.append(path)
    validation = pd.DataFrame(rows)
    if val_date is not None:
        path = out / f"validation_{tag}.csv"
        validation.to_csv(path, index=False)
        files.append(path)
    for key, reason in skipped.items():
        warnings.warn(f"Skipped {key}: {reason}")
    return RunResult(predictions, validation, skipped, info["model_id"], info["experiment_id"], variable, files)
