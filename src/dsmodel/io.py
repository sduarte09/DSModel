"""Input for the downscaling models: station tables, catalogues and climate-model NetCDF files.

NetCDF reading uses ``netCDF4`` when it is installed and falls back to
``scipy.io.netcdf_file`` (classic NetCDF3 files such as the CMIP5 daily files).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400.0  # kg m-2 s-1 -> mm day-1
KELVIN = 273.15
KNOWN_VARIABLES = ("pr", "tas", "tasmin", "tasmax")


# ----------------------------------------------------------------------------- tables

def _read_table(path, **kwargs):
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, **kwargs)  # needs openpyxl
    return pd.read_csv(path, **kwargs)


def _find_column(table, candidates, position):
    lower = {str(c).strip().lower(): c for c in table.columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    return table.columns[position]


def read_station_table(path, date_column="Date", dayfirst=True):
    """Daily station table: a date column and one column per station.

    The date column is looked up by name (``Date``/``Fecha``, case-insensitive) and falls
    back to the first column. Dates are read as ``dd/mm/yyyy`` by default. Returns a
    DataFrame indexed by date with the station IDs (as strings) as columns.
    """
    table = _read_table(path)
    col = _find_column(table, (str(date_column).lower(), "date", "fecha"), 0)
    dates = pd.to_datetime(table[col], dayfirst=dayfirst)
    table = table.drop(columns=col)
    table.columns = [str(c).strip() for c in table.columns]
    table.index = pd.DatetimeIndex(dates, name="Date")
    return table.apply(pd.to_numeric, errors="coerce").sort_index()


def read_catalog(path):
    """Station catalogue with ``ID``, ``latitude`` and ``longitude`` columns.

    Column names are looked up case-insensitively in English or Spanish (``ID``,
    ``Latitude``/``Latitud``, ``Longitude``/``Longitud``); if not found, the first three
    columns are used in that order. Returns a DataFrame indexed by station ID (string).
    """
    table = _read_table(path)
    ident = _find_column(table, ("id", "name", "station", "estacion", "estación"), 0)
    lat = _find_column(table, ("latitude", "latitud", "lat"), 1)
    lon = _find_column(table, ("longitude", "longitud", "lon"), 2)
    out = pd.DataFrame({"latitude": pd.to_numeric(table[lat]), "longitude": pd.to_numeric(table[lon])})
    out.index = pd.Index(table[ident].astype(str).str.strip(), name="ID")
    return out


# ----------------------------------------------------------------------------- NetCDF

class _Reader:
    """Minimal common interface over netCDF4.Dataset and scipy's netcdf_file."""

    def __init__(self, path):
        try:
            import netCDF4

            self._ds = netCDF4.Dataset(str(path))
            self._kind = "netcdf4"
        except ImportError:
            from scipy.io import netcdf_file

            self._ds = netcdf_file(str(path), mmap=False)
            self._kind = "scipy"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def variables(self):
        return list(self._ds.variables.keys())

    def has(self, name):
        return name in self._ds.variables

    def array(self, name, index=None):
        var = self._ds.variables[name]
        data = var[:] if index is None else var[index]
        if self._kind == "netcdf4":
            return np.ma.filled(np.ma.asarray(data, dtype="float64"), np.nan)
        return np.array(data, dtype="float64")

    def attr(self, name, attr, default=None):
        value = getattr(self._ds.variables[name], attr, default)
        return value.decode() if isinstance(value, bytes) else value

    def global_attr(self, attr, default=None):
        value = getattr(self._ds, attr, default)
        return value.decode() if isinstance(value, bytes) else value

    def fill_value(self, name):
        for key in ("_FillValue", "missing_value"):
            value = getattr(self._ds.variables[name], key, None)
            if value is not None:
                return float(np.ravel(value)[0])
        return None

    def close(self):
        self._ds.close()


def _mask_fill(data, fill):
    """NaN where ``data`` equals the fill value (compared with a tolerance, since the fill
    attribute is often stored in single precision)."""
    if fill is not None:
        data[np.abs(data - fill) <= 1e-6 * abs(fill)] = np.nan
    return data


def list_netcdf(folder):
    """Sorted ``.nc`` files of a folder (the legacy code used ``os.listdir``, which is unordered)."""
    files = sorted(Path(folder).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files in {folder}")
    return files


def decode_time(values, units, calendar="standard", midday_offset=True):
    """Daily NetCDF time values to a DatetimeIndex of calendar days.

    Only calendars with a pandas equivalent are supported (standard, gregorian,
    proleptic_gregorian). Daily CMIP files stamp days at 12:00, so 0.5 day is removed,
    as in the legacy code (``time - 0.5``).
    """
    if calendar not in ("standard", "gregorian", "proleptic_gregorian"):
        raise ValueError(f"Unsupported calendar {calendar!r}; convert the file to a gregorian calendar first")
    unit, _, origin = units.partition(" since ")
    if unit.strip() != "days":
        raise ValueError(f"Only 'days since ...' time units are supported, got {units!r}")
    values = np.asarray(values, dtype="float64") - (0.5 if midday_offset else 0.0)
    return pd.DatetimeIndex(pd.Timestamp(origin.strip()) + pd.to_timedelta(np.floor(values + 1e-6), unit="D"))


def detect_variable(path):
    """First known climate variable (``pr``, ``tas``, ``tasmin``, ``tasmax``) in a file.

    The legacy code took the last variable of the file, which only works for some files.
    """
    with _Reader(path) as reader:
        for name in KNOWN_VARIABLES:
            if reader.has(name):
                return name
        raise ValueError(f"None of {KNOWN_VARIABLES} found in {path}; pass the variable name explicitly")


def convert_units(variable, values):
    """Precipitation kg m-2 s-1 -> mm/day; temperature K -> degrees C."""
    if variable == "pr":
        return values * SECONDS_PER_DAY
    return values - KELVIN


def model_info(path):
    """``model_id`` and ``experiment_id`` global attributes (used to name output files)."""
    with _Reader(path) as reader:
        return {
            "model_id": reader.global_attr("model_id", "model"),
            "experiment_id": reader.global_attr("experiment_id", "experiment"),
        }


def read_grid(path):
    """Latitude and longitude of a model file as 2-D arrays (n_lat, n_lon), longitudes in 0-360."""
    with _Reader(path) as reader:
        lat, lon = reader.array("lat"), reader.array("lon")
    if lat.ndim == 1:
        lat, lon = np.meshgrid(lat, lon, indexing="ij")
    lon = np.where(lon < 0, lon + 360.0, lon)
    return lat, lon


def _edges(centers):
    mid = (centers[1:] + centers[:-1]) / 2
    return np.concatenate([[centers[0] - (mid[0] - centers[0])], mid, [centers[-1] + (centers[-1] - mid[-1])]])


def locate_cell(lat, lon, latitude, longitude):
    """Grid indices ``(i, j)`` of the model cell that contains a point.

    ``lat``/``lon`` are the 2-D arrays from :func:`read_grid`. For a regular grid the cell
    is bounded by the midpoints between neighbouring centres (the legacy rule); for a
    curvilinear grid the nearest centre is used. Negative longitudes are converted to
    0-360. Raises ``ValueError`` if the point is outside the grid.
    """
    longitude = longitude + 360.0 if longitude < 0 else longitude
    regular = np.allclose(lat, lat[:, :1]) and np.allclose(lon, lon[:1, :])
    if regular:
        lat1, lon1 = lat[:, 0], lon[0, :]
        lat_e, lon_e = _edges(lat1), _edges(lon1)
        lo, hi = np.minimum(lat_e[:-1], lat_e[1:]), np.maximum(lat_e[:-1], lat_e[1:])
        in_lat = np.flatnonzero((lo <= latitude) & (latitude <= hi))
        lo, hi = np.minimum(lon_e[:-1], lon_e[1:]), np.maximum(lon_e[:-1], lon_e[1:])
        in_lon = np.flatnonzero((lo <= longitude) & (longitude <= hi))
        if in_lat.size == 0 or in_lon.size == 0:
            raise ValueError(f"Point ({latitude}, {longitude}) lies outside the model grid")
        return int(in_lat[0]), int(in_lon[0])
    dlon = (lon - longitude + 180.0) % 360.0 - 180.0
    dist = (lat - latitude) ** 2 + (dlon * np.cos(np.radians(latitude))) ** 2
    i, j = np.unravel_index(np.nanargmin(dist), dist.shape)
    return int(i), int(j)


def read_cell_series(paths, i, j, variable=None):
    """Daily series of one model cell from one or more files, joined in time order.

    Units are converted (see :func:`convert_units`). Where files overlap in time the later
    file wins. The series is named after the variable.
    """
    paths = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
    variable = variable or detect_variable(paths[0])
    parts = []
    for path in paths:
        with _Reader(path) as reader:
            data = _mask_fill(reader.array(variable, (slice(None), i, j)), reader.fill_value(variable))
            index = decode_time(reader.array("time"), reader.attr("time", "units"),
                                reader.attr("time", "calendar", "standard"))
        parts.append(pd.Series(convert_units(variable, data), index=index, name=variable))
    joined = pd.concat(parts)
    return joined[~joined.index.duplicated(keep="last")].sort_index()


def concat_model(historic, future):
    """Join a historic and a scenario series; the scenario wins where dates overlap."""
    joined = pd.concat([historic, future])
    return joined[~joined.index.duplicated(keep="last")].sort_index()
