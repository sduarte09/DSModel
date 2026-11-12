"""Machine-learning statistical downscaling of climate-model output to weather stations."""
from . import io, models, pipeline, runner
from .models import METHODS, fit_predict
from .pipeline import DownscaleResult, downscale_station, metrics
from .runner import run_catalog

__version__ = "0.1.0"
__all__ = ["io", "models", "pipeline", "runner", "METHODS", "fit_predict", "DownscaleResult",
           "downscale_station", "metrics", "run_catalog"]
