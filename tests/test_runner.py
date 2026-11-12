import numpy as np
import pandas as pd
import pytest

from dsmodel import io, runner

FAST = {"alpha": [0.001, 0.01], "solver": ["svd"]}


@pytest.fixture(scope="module")
def run(demo_data, tmp_path_factory):
    _, args = demo_data
    out = tmp_path_factory.mktemp("out")
    calls = []
    original = io.read_cell_series

    def counting(*a, **k):
        calls.append(a[1:3])
        return original(*a, **k)

    io.read_cell_series = counting
    try:
        result = runner.run_catalog(output_folder=out, methods=("linear",), cv=3, param_grid=FAST, **args)
    finally:
        io.read_cell_series = original
    return result, out, calls


def test_outputs_and_names(run):
    result, out, _ = run
    assert (result.model_id, result.experiment_id, result.variable) == ("DEMO-GCM", "rcp45", "pr")
    assert (out / "linear_DEMO-GCM_pr_rcp45.csv").exists() and (out / "validation_DEMO-GCM_pr_rcp45.csv").exists()
    assert (out / "1001_linear_DEMO-GCM_pr_rcp45.png").exists()
    frame = pd.read_csv(out / "linear_DEMO-GCM_pr_rcp45.csv", index_col="Index", parse_dates=True)
    assert list(frame.columns) == ["1001", "1002", "1003"]
    assert frame.index[-1] == pd.Timestamp("2009-12-31")


def test_stations_in_the_same_cell_share_one_model_series(run):
    _, _, calls = run
    assert sorted(set(calls)) == [(1, 1), (2, 2)] and len(calls) == 4  # two cells x (historic, rcp)


def test_predictions_are_non_negative_for_precipitation_and_validated(run):
    result, _, _ = run
    frame = result.predictions["linear"]
    assert frame.loc["2006-01-01":].min().min() >= 0
    v = result.validation
    assert set(v["station"]) == {"1001", "1002", "1003"} and (v["correlation"] > 0.9).all()


def test_unknown_stations_and_points_outside_the_grid_are_skipped(demo_data, tmp_path):
    root, args = demo_data
    cat = pd.read_csv(args["station_catalog"])
    cat.loc[len(cat)] = ["9999", 5.0, -74.0]    # not in the station table
    cat.loc[len(cat)] = ["1001x", 50.0, -74.0]  # outside the grid
    table = pd.read_csv(args["station_data"])
    table["1001x"] = table["1001"]
    cat.to_csv(tmp_path / "cat.csv", index=False)
    table.to_csv(tmp_path / "st.csv", index=False)
    args = {**args, "station_catalog": tmp_path / "cat.csv", "station_data": tmp_path / "st.csv"}
    with pytest.warns(UserWarning, match="Skipped"):
        result = runner.run_catalog(output_folder=tmp_path / "o", methods=("linear",), plots=False, cv=3,
                                    param_grid=FAST, **args)
    assert "no column" in result.skipped["9999"] and "outside" in result.skipped["1001x"]
    assert list(result.predictions["linear"].columns) == ["1001", "1002", "1003"]
