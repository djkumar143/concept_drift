import json
import logging
from pathlib import Path

from pyspark.ml.classification import LogisticRegressionModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_REGISTRY = PROJECT_ROOT / "model_registry.json"
MODELS_DIR = PROJECT_ROOT / "models"

_cached_model = None
_cached_version = None


def read_model_registry() -> dict:
    with open(MODEL_REGISTRY, "r") as file:
        return json.load(file)


def get_current_model_version() -> str:
    registry = read_model_registry()
    return registry["current_version"]


def load_current_model() -> LogisticRegressionModel:
    global _cached_model
    global _cached_version

    current_version = get_current_model_version()

    if (
        _cached_model is not None
        and _cached_version == current_version
    ):
        return _cached_model

    model_path = MODELS_DIR / current_version

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model '{current_version}' not found at {model_path}"
        )

    _cached_model = LogisticRegressionModel.load(str(model_path))
    _cached_version = current_version

    logger.info(f"Loaded model: {current_version}")

    return _cached_model