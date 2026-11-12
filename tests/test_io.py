import numpy as np
import pandas as pd
import pytest
from scipy.io import netcdf_file

from dsmodel import io


def _write_nc(path, dates, values, variable="pr", fill=None, lat=(0.0, 10.0), lon=(0.0, 10.0)):
    days = (dates - pd.Timestamp("1850-01-01")) / pd.Timedelta(days=1) + 0.5
    with netcdf_file(str(path), "w") as f:
        f.createDimension("time", len(dates)); f.createDimension("lat", len(lat)); f.createDimension("lon", len(lon))
        t = f.createVariable("time", "d", ("time",)); t[:] = np.asarray(days)
        t.units, t.calendar = "days since 1850-1-1 00:00:00", "proleptic_gregorian"
        f.createVariable("lat", "d", ("lat",))[:] = lat
        f.createVariable("lon", "d", ("lon",))[:] = lon
        v = f.createVariable(variable, "f", ("time", "lat", "lon"))
        data = np.full((len(dates), len(lat), len(lon)), values, dtype="float32")
        if fill is not None:
            data[1] = fill
            v._FillValue = np.float32(fill)
        v[:] = data


def test_read_station_table_accepts_spanish_headers_and_missing_values(tmp_path):
    (tmp_path / "s.csv").write_text("Fecha,1001,1002\n01/02/2000,1.5,\n02/02/2000,0,3\n", encoding="utf-8")
    t = io.read_station_table(tmp_path / "s.csv")
    assert t.index[0] == pd.Timestamp("2000-02-01")  # dd/mm/yyyy
    assert list(t.columns) == ["1001", "1002"] and np.isnan(t.loc["2000-02-01", "1002"])


def test_read_catalog_finds_spanish_columns(tmp_path):
    (tmp_path / "c.csv").write_text("ID,Latitud,Longitud,Altura\n1001 ,4.6,-74.1,2600\n1002,5,-75,10\n", encoding="utf-8")
    c = io.read_catalog(tmp_path / "c.csv")
    assert list(c.index) == ["1001", "1002"] and c.loc["1001", "latitude"] == 4.6 and c.loc["1002", "longitude"] == -75


def test_decode_time_removes_the_midday_offset_and_rejects_other_calendars():
    idx = io.decode_time([56978.5, 56979.5], "days since 1850-1-1 00:00:00", "proleptic_gregorian")
    assert list(idx) == [pd.Timestamp("2006-01-01"), pd.Timestamp("2006-01-02")]
    with pytest.raises(ValueError):
        io.decode_time([0.5], "days since 1850-1-1", "360_day")


def test_convert_units():
    assert io.convert_units("pr", np.array([1e-5]))[0] == pytest.approx(0.864)
    assert io.convert_units("tas", np.array([300.0]))[0] == pytest.approx(26.85)


def test_locate_cell_regular_grid_uses_midpoints_and_wraps_longitude():
    lat, lon = np.meshgrid([0.0, 2.0, 4.0], [10.0, 12.0, 14.0], indexing="ij")
    assert io.locate_cell(lat, lon, 2.9, 12.9) == (1, 1)
    assert io.locate_cell(lat, lon, 3.1, 12.9) == (2, 1)
    lat, lon = np.meshgrid([0.0, 2.0], [358.0, 2.0], indexing="ij")  # longitudes 358 and 2 are not monotonic: nearest
    assert io.locate_cell(lat, lon, 0.1, -2.0) == (0, 0)  # -2 -> 358
    with pytest.raises(ValueError):
        io.locate_cell(*np.meshgrid([0.0, 2.0], [10.0, 12.0], indexing="ij"), 50.0, 11.0)


def test_locate_cell_curvilinear_grid_uses_the_nearest_centre():
    lat = np.array([[0.0, 0.5], [1.0, 1.6]])
    lon = np.array([[10.0, 11.0], [10.4, 11.5]])
    assert io.locate_cell(lat, lon, 1.5, 11.4) == (1, 1)


def test_read_cell_series_joins_files_masks_fill_and_converts(tmp_path):
    a, b = pd.date_range("2000-01-01", periods=4), pd.date_range("2000-01-03", periods=4)  # two days overlap
    _write_nc(tmp_path / "a.nc", a, 1e-5, fill=1e20)
    _write_nc(tmp_path / "b.nc", b, 2e-5)
    s = io.read_cell_series([tmp_path / "a.nc", tmp_path / "b.nc"], 0, 0)
    assert s.index[0] == pd.Timestamp("2000-01-01") and len(s) == 6 and s.name == "pr"
    assert np.isnan(s.iloc[1])  # fill value masked
    assert s.iloc[0] == pytest.approx(0.864, rel=1e-4)
    assert s.iloc[2] == pytest.approx(1.728, rel=1e-4)  # later file wins on the overlap


def test_detect_variable_and_model_info(tmp_path):
    _write_nc(tmp_path / "t.nc", pd.date_range("2000-01-01", periods=2), 300.0, variable="tas")
    assert io.detect_variable(tmp_path / "t.nc") == "tas"
    assert io.model_info(tmp_path / "t.nc") == {"model_id": "model", "experiment_id": "experiment"}
    _write_nc(tmp_path / "x.nc", pd.date_range("2000-01-01", periods=2), 1.0, variable="foo")
    with pytest.raises(ValueError):
        io.detect_variable(tmp_path / "x.nc")


def test_list_netcdf_is_sorted_and_complains_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.list_netcdf(tmp_path)
    for n in ("b.nc", "a.nc"):
        (tmp_path / n).write_bytes(b"")
    assert [p.name for p in io.list_netcdf(tmp_path)] == ["a.nc", "b.nc"]
