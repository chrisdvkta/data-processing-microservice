import pandas as pd

from app.model_utils import load_model


def predict_from_df(df: pd.DataFrame, model_name: str = "default") -> list:
    if "smoker" in df.columns:
        df["smoker"] = df["smoker"].map({"Yes": 1, "No": 0})

    model = load_model(model_name)
    predictions = model.predict(df)

    result = []
    for prediction in predictions.tolist():
        if prediction == 1:
            result.append("At Risk")
        else:
            result.append("Not at Risk")
    return result
