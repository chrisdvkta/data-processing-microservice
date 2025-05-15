import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

from app.model_utils import save_model


def train_model_df(df: pd.DataFrame, model_name: str = "default") -> dict:
    if "risk" not in df.columns:
        raise ValueError("Dataset must have a risk column")

    X = df.drop(columns=["risk"])
    if "smoker" in X.columns:
        X["smoker"] = X["smoker"].map({"Yes": 1, "No": 0})
    y = df["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    save_model(model, model_name)

    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred, normalize=False),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }
