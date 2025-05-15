import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")


def save_model(model) -> None:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No saved model found.")
    return joblib.load(MODEL_PATH)
