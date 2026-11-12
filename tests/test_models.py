import numpy as np
import pytest

from dsmodel import models

GRID = {"alpha": [0.001, 0.01], "solver": ["svd"]}


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        models.build_search("svr")


def test_legacy_folds_are_the_default_and_are_capped_by_the_sample_size():
    assert models.build_search("knn").cv == 30
    assert models.build_search("forest").cv == 25
    assert models.build_search("linear", n_samples=10).cv == 10
    assert models.build_search("knn", cv=5, n_samples=100).cv == 5


def test_ridge_recovers_a_linear_relation():
    x = np.linspace(0, 10, 200)
    pred, search = models.fit_predict("linear", x, 2 * x + 1, [0.0, 5.0], cv=4, param_grid=GRID)
    np.testing.assert_allclose(pred, [1.0, 11.0], atol=0.05)
    assert search.best_params_["solver"] == "svd"


def test_knn_returns_the_value_of_the_nearest_calibration_day():
    x = np.arange(100.0)
    pred, _ = models.fit_predict("knn", x, x ** 2, [10.2, 50.4], cv=3,
                                 param_grid={"n_neighbors": [1], "algorithm": ["ball_tree"]})
    np.testing.assert_allclose(pred, [100.0, 2500.0])


def test_forest_is_reproducible_with_a_seed():
    rng = np.random.default_rng(1)
    x = rng.random(120)
    y = x + rng.normal(0, 0.1, 120)
    grid = {"n_estimators": [5]}
    a, _ = models.fit_predict("forest", x, y, x[:10], cv=3, param_grid=grid, random_state=7)
    b, _ = models.fit_predict("forest", x, y, x[:10], cv=3, param_grid=grid, random_state=7)
    np.testing.assert_array_equal(a, b)


def test_mlp_runs_on_a_tiny_problem():
    x = np.linspace(0, 1, 60)
    pred, _ = models.fit_predict("mlp", x, x, [0.5], cv=3, random_state=0,
                                 param_grid={"hidden_layer_sizes": [(5,)], "max_iter": [300], "activation": ["identity"]})
    assert pred.shape == (1,) and np.isfinite(pred).all()


def test_fit_predict_validates_its_input():
    with pytest.raises(ValueError):
        models.fit_predict("linear", [1, 2, 3, 4], [1, 2, 3], [1])
    with pytest.raises(ValueError):
        models.fit_predict("linear", [1, 2], [1, 2], [1])
