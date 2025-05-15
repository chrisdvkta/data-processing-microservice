import pandas as pd

from app.model_utils import load_model


def predict_from_df(df: pd.DataFrame) -> list:
    if "smoker" in df.columns:
        df["smoker"] = df["smoker"].map({"Yes": 1, "No": 0})

    model = load_model()
    predictions = model.predict(df)

    return predictions.tolist()  # serializable json
