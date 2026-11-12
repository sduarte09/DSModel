"""Machine-learning regressors used for statistical downscaling.

Each method maps the climate-model value of a day to the station value of the same day.
The hyper-parameter grids and cross-validation folds are the ones of the legacy scripts
(``DSMethods/``); they can be overridden. Scoring is the median absolute error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


@dataclass(frozen=True)
class MethodSpec:
    estimator: type
    grid: dict
    cv: int
    seeded: bool = False  # estimator takes ``random_state``
    extra: dict = field(default_factory=dict)


METHODS: dict[str, MethodSpec] = {
    # DSMethods/k_Nearest_Neighbors.py (cv=30)
    "knn": MethodSpec(
        KNeighborsRegressor,
        {"weights": ["uniform"], "algorithm": ["ball_tree"], "n_neighbors": [1, 2, 3],
         "leaf_size": [5, 10, 15, 20, 25, 30]},
        cv=30,
    ),
    # DSMethods/forest_trees.py (cv=25)
    "forest": MethodSpec(
        RandomForestRegressor,
        {"n_estimators": [10, 50, 100], "min_samples_split": [2], "min_samples_leaf": [1]},
        cv=25, seeded=True,
    ),
    # DSMethods/linear_regression.py (Ridge, cv=30)
    "linear": MethodSpec(
        Ridge,
        {"alpha": np.linspace(0.001, 0.1, 50).tolist(), "solver": ["svd", "sag"]},
        cv=30, seeded=True,
    ),
    # DSMethods/multi_layer_perceptron.py (cv=20); not part of the legacy run loop
    "mlp": MethodSpec(
        MLPRegressor,
        {"hidden_layer_sizes": [(200,)], "activation": ["identity", "logistic"], "solver": ["adam"],
         "learning_rate": ["adaptive"], "learning_rate_init": [0.0001]},
        cv=20, seeded=True,
    ),
}


def build_search(method, cv=None, param_grid=None, scoring="neg_median_absolute_error",
                 random_state=0, n_jobs=None, n_samples=None):
    """A :class:`~sklearn.model_selection.GridSearchCV` for one method.

    ``cv`` defaults to the legacy number of folds and is capped by ``n_samples`` when given.
    ``random_state`` seeds the stochastic estimators (the legacy code was unseeded).
    """
    try:
        spec = METHODS[method]
    except KeyError:
        raise ValueError(f"Unknown method {method!r}; choose one of {sorted(METHODS)}") from None
    folds = spec.cv if cv is None else cv
    if n_samples is not None:
        folds = max(2, min(folds, n_samples))
    kwargs: dict[str, Any] = {"random_state": random_state} if spec.seeded else {}
    return GridSearchCV(spec.estimator(**kwargs), param_grid or spec.grid, cv=folds, scoring=scoring,
                        n_jobs=n_jobs, verbose=0)


def fit_predict(method, x_cal, y_cal, x_new, **kwargs):
    """Fit ``method`` on calibration pairs and predict for new model values.

    ``x_cal``, ``y_cal`` and ``x_new`` are 1-D. Returns ``(predictions, fitted_search)``.
    Extra keyword arguments go to :func:`build_search`.
    """
    x_cal = np.asarray(x_cal, dtype=float).reshape(-1, 1)
    y_cal = np.asarray(y_cal, dtype=float).ravel()
    if len(x_cal) != len(y_cal):
        raise ValueError("x_cal and y_cal must have the same length")
    if len(x_cal) < 4:
        raise ValueError("At least 4 calibration pairs are needed")
    search = build_search(method, n_samples=len(x_cal), **kwargs)
    search.fit(x_cal, y_cal)
    x_new = np.asarray(x_new, dtype=float).reshape(-1, 1)
    return search.predict(x_new), search
