import numpy as np
import pandas as pd
import pytest

from dsmodel import pipeline

GRID = {"alpha": [0.001, 0.01], "solver": ["svd"]}


@pytest.fixture
def pair():
    dates = pd.date_range("2000-01-01", "2002-12-31")
    rng = np.random.default_rng(0)
    model = pd.Series(rng.gamma(1.0, 4.0, len(dates)), index=dates)
    station = (0.8 * model + 1).rename("1001")
    station.iloc[::17] = np.nan  # gaps
    return station, model


def test_series_keeps_observations_then_predicts(pair):
    station, model = pair
    res = pipeline.downscale_station(station, model, "2001-12-31", "2002-12-31", cv=4, param_grid=GRID, method="linear")
    s = res.series
    assert s.index[0] == station.dropna().index[0] and s.index[-1] == pd.Timestamp("2002-12-31")
    obs_part = s.loc[:"2001-12-31"]
    pd.testing.assert_series_equal(obs_part.dropna(), station.loc[:"2001-12-31"].dropna(), check_names=False)
    pred = s.loc["2002-01-01":]
    assert pred.notna().all() and len(pred) == 365
    np.testing.assert_allclose(pred.values, 0.8 * model.loc["2002-01-01":].values + 1, atol=0.05)
    assert res.n_calibration == station.loc[:"2001-12-31"].dropna().size
    assert res.series.name == "1001" and res.method == "linear"


def test_validation_scores_only_days_with_observations(pair):
    station, model = pair
    res = pipeline.downscale_station(station, model, "2001-12-31", "2002-12-31", val_end="2002-06-30",
                                     cv=4, param_grid=GRID, method="linear")
    v = res.validation
    assert v["n"] == station.loc["2002-01-01":"2002-06-30"].dropna().size
    assert v["rmse"] < 0.05 and v["correlation"] > 0.999


def test_clip_negative(pair):
    station, model = pair
    station = -station  # the fit will predict negative values
    raw = pipeline.downscale_station(station, model, "2001-12-31", "2002-03-31", cv=4, param_grid=GRID, method="linear")
    clipped = pipeline.downscale_station(station, model, "2001-12-31", "2002-03-31", cv=4, param_grid=GRID,
                                         method="linear", clip_negative=True)
    assert raw.series.loc["2002-01-01":].min() < 0 and clipped.series.loc["2002-01-01":].min() == 0


def test_invalid_dates_and_empty_inputs(pair):
    station, model = pair
    with pytest.raises(ValueError):
        pipeline.downscale_station(station, model, "2002-06-30", "2002-01-31")
    with pytest.raises(ValueError):
        pipeline.downscale_station(station, model, "2001-12-31", "2002-12-31", val_end="2003-12-31")
    with pytest.raises(ValueError):
        pipeline.downscale_station(station.iloc[:0], model, "2001-12-31", "2002-12-31")
    with pytest.raises(ValueError):
        pipeline.downscale_station(station, model, "2002-12-31", "2003-12-31")  # model ends on 2002-12-31


def test_metrics_ignore_missing_values():
    m = pipeline.metrics([1.0, 2.0, np.nan, 4.0], [1.0, 3.0, 5.0, np.nan])
    assert m["n"] == 2 and m["rmse"] == pytest.approx(np.sqrt(0.5)) and m["bias"] == pytest.approx(-0.5)
    assert pipeline.metrics([np.nan], [1.0])["n"] == 0
