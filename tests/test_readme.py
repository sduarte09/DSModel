import json
from pathlib import Path

import pandas as pd

from dsmodel import downscale_station, io


def test_single_station_snippet_from_the_readme_works(demo_data):
    _, args = demo_data
    nc_files = io.list_netcdf(args["model_historic"]) + io.list_netcdf(args["model_rcp"])
    cell = io.locate_cell(*io.read_grid(nc_files[0]), 4.3, -75.8)  # same point as station 1001
    model = io.read_cell_series(nc_files, *cell)
    station = io.read_station_table(args["station_data"])["1001"]
    res = downscale_station(station, model, cal_end="2005-12-31", app_end="2009-12-31", method="linear",
                            cv=3, param_grid={"alpha": [0.01], "solver": ["svd"]})
    assert res.series.index[-1] == pd.Timestamp("2009-12-31") and res.best_params["solver"] == "svd"


def test_run_notebook_is_valid_json():
    nb = json.loads((Path(__file__).parents[1] / "notebooks" / "run_dsmodel.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and any(c["cell_type"] == "code" for c in nb["cells"])
