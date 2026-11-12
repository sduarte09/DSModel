# Notes on the legacy code

The original scripts are in `legacy/` and are unchanged. They were written for Python 3.8 and
older pandas/scikit-learn (February 2022). This file lists what was found and which choices
`src/dsmodel` makes. The package was checked by reading the legacy code and by tests on
synthetic data; it was **not** compared with output of the legacy scripts.

## Things that no longer run on current libraries

- `pd.datetime` and the `date_parser` argument of `read_csv`, `resample(..., how='sum')`,
  `reindex_axis` (all removed from pandas).
- `his.variables.keys()[-1]` (a `dict` view cannot be indexed in Python 3).
- `sys.path.append` with a hard-coded Windows path, and `'\\'` as path separator.
- `legacy/extras/`: Keras 1 API (`init=`), `statsmodels.tsa.arima_model` (removed), and
  scripts that use undefined names (`series`, `steps`, `validate`, `aaa`, `data`). They also
  import `ML_Functions`, which is not in this repository, and are not ported.
  `Chaos_Statistics.py` is superseded by the SNLDA and CSD packages.

## Likely bugs in `Statistical Downscaling Models.py`

- **File order.** `os.listdir` is unordered, and the time units come from "the first file
  listed". `dsmodel.io.list_netcdf` sorts the files and times are decoded per file.
- **Cell search.** The loops index `[cell_x-1, cell_y-1]`, which wraps around to the last
  element for cell 0, mix the indices of the latitude and longitude tests, never reach the
  last row and column, and `cell_x = cell_x + 1` has no effect inside a `for` loop. When no
  cell is found the indices stay NaN and `int(nan)` fails. `dsmodel.io.locate_cell` uses the
  midpoints between neighbouring centres on each axis (the same rule), the nearest centre for
  curvilinear grids, and raises `ValueError` for a point outside the grid.
- **Model cache size.** `model = [None] * (ns - 1)` is one entry too short if every station
  sits in its own model cell. The package keeps a cache keyed by cell.
- **`his.close` and `rcp.close`** lack parentheses, so files are never closed explicitly.
- **Gaps in the station series.** `future[name] = estacion.loc[:cal_date]` is aligned to the
  model index, so days without observations become NaN in the training target, which
  scikit-learn refuses. The package fits on the days where both series have a value.
- **Unused `val_date`.** The validation date is accepted but never used. The package scores
  the days in `(cal_end, val_end]` that have observations (RMSE, MAE, bias, correlation).
- **Plot names.** The linear-regression plot is saved with the name `forest`, overwriting
  the forest plot.
- **Annual aggregation.** Annual values are sums, also for temperature. The package uses sums
  for `pr` and means for temperature.
- **Unseeded models.** Random forest and MLP are unseeded, so results change from run to run.
  The package takes a `random_state` (default 0).
- **MLP.** `multi_layer_perceptron.py` uses `time` without importing it, has code after
  `return`, and is only a placeholder `print("MLP")` in the run loop. `dsmodel.models` includes
  `mlp`, but it is not in the default method list.
- **Compiled files.** The attached `__pycache__/*.cpython-38.pyc` files are bytecode and do
  not belong in the repository (`.gitignore` excludes them; use `git rm --cached` once).

## Design choices kept from the legacy code

- Each method maps the model value of a day to the station value of the same day. There are
  no lagged or neighbouring values, so the downscaled series has the temporal structure of the
  model, not of the station.
- Hyper-parameter grids, the number of folds (k-NN 30, forest 25, ridge 30, MLP 20) and the
  scoring (`neg_median_absolute_error`) are those of `DSMethods/`.
- The folds are those of an integer `cv` in scikit-learn for regressors: contiguous,
  unshuffled blocks. For a series with autocorrelation this is a reasonable choice, but
  `cv` and `param_grid` can be changed.
- The result keeps the observations up to the calibration date and predictions after it.
- Predictions are not clipped. `clip_negative=True` sets negative values to zero (the runner
  does this for precipitation).
- The `Weather Generator` in the original README has no code in the repository.
