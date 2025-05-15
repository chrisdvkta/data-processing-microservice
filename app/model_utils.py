import os
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def get_model_path(model_name: str) -> Path:
    filename = f"{model_name}.pkl"
    return MODEL_DIR / filename


def save_model(model, model_name: str = "default") -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = get_model_path(model_name)
    joblib.dump(model, path)


def load_model(model_name: str = "default"):
    path = get_model_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"No saved model found for '{model_name}'.")
    return joblib.load(path)
