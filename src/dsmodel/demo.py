"""A small synthetic dataset (climate-model NetCDF files, stations, catalogue) for tests and demos.

The data are random; they only show that the pieces fit together and say nothing about
the skill of the methods.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import netcdf_file

LATS = np.array([2.0, 4.0, 6.0, 8.0])
LONS = np.array([282.0, 284.0, 286.0, 288.0])


def _write_nc(path, dates, field, experiment):
    origin = pd.Timestamp("1850-01-01")
    days = (dates - origin) / pd.Timedelta(days=1) + 0.5
    with netcdf_file(str(path), "w") as f:
        f.model_id, f.experiment_id = "DEMO-GCM", experiment
        f.createDimension("time", len(dates)); f.createDimension("lat", len(LATS)); f.createDimension("lon", len(LONS))
        t = f.createVariable("time", "d", ("time",)); t[:] = np.asarray(days)
        t.units, t.calendar = "days since 1850-1-1 00:00:00", "proleptic_gregorian"
        la = f.createVariable("lat", "d", ("lat",)); la[:] = LATS
        lo = f.createVariable("lon", "d", ("lon",)); lo[:] = LONS
        pr = f.createVariable("pr", "f", ("time", "lat", "lon"))
        pr[:] = (field / 86400.0).astype("float32")  # mm/day -> kg m-2 s-1
        pr.units, pr._FillValue = "kg m-2 s-1", np.float32(1e20)


def make_demo_dataset(directory, seed=0, start="1996-01-01", cal_end="2005-12-31", end="2009-12-31"):
    """Write a demo dataset and return a dict of the arguments for :func:`dsmodel.runner.run_catalog`.

    Two historic files (split at 2000), one scenario file, a station table with three
    stations (two of them in the same model cell) and a catalogue. Station values are a
    noisy linear function of the model cell with some missing days.
    """
    rng = np.random.default_rng(seed)
    root = Path(directory)
    (root / "historic").mkdir(parents=True, exist_ok=True)
    (root / "rcp").mkdir(exist_ok=True)

    dates = pd.date_range(start, end)
    shape = (len(dates), len(LATS), len(LONS))
    field = rng.gamma(0.8, 5.0, shape)  # mm/day
    hist = dates <= pd.Timestamp(cal_end)
    split = dates < pd.Timestamp("2000-01-01")
    _write_nc(root / "historic" / "pr_day_DEMO_historical_1.nc", dates[hist & split], field[hist & split], "historical")
    _write_nc(root / "historic" / "pr_day_DEMO_historical_2.nc", dates[hist & ~split], field[hist & ~split], "historical")
    _write_nc(root / "rcp" / "pr_day_DEMO_rcp45.nc", dates[~hist], field[~hist], "rcp45")

    # (lat, lon, cell index) of the stations: A and B share a cell, C is elsewhere
    stations = {"1001": (4.3, 284.2, (1, 1)), "1002": (3.6, 283.4, (1, 1)), "1003": (6.4, 286.8, (2, 2))}
    table = pd.DataFrame(index=dates)
    for sid, (_, _, (i, j)) in stations.items():
        obs = 0.7 * field[:, i, j] + rng.gamma(1.0, 1.0, len(dates))
        obs[rng.random(len(dates)) < 0.03] = np.nan
        table[sid] = obs
    table = table.loc[:"2007-12-31"]  # observations also cover part of the application period (validation)
    table.index = table.index.strftime("%d/%m/%Y")
    table.to_csv(root / "stations.csv", index_label="Date")
    pd.DataFrame({"ID": list(stations), "Latitud": [v[0] for v in stations.values()],
                  "Longitud": [v[1] - 360 for v in stations.values()]}).to_csv(root / "catalog.csv", index=False)
    return {
        "station_data": root / "stations.csv", "station_catalog": root / "catalog.csv",
        "model_historic": root / "historic", "model_rcp": root / "rcp", "cal_date": cal_end,
        "val_date": "2007-12-31", "app_date": end,
    }
