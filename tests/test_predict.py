from fastapi.testclient import TestClient
from app.main import app
from io import BytesIO

client = TestClient(app)


def test_prediction():
    # training for creating a model first.
    training_csv = """age,blood_pressure,cholestrol,smoker,risk
45,120,200,No,Low
52,135,240,Yes,High
60,140,260,No,High
38,115,190,No,Low
"""

    train_file = BytesIO(training_csv.encode("utf-8"))
    response_train = client.post(
        "/train", {"file": ("training_data.csv", train_file, "text/csv")}
    )

    assert response_train.status_code == 200
    # predict without the label column
    csv_data = """
age,blood_pressure,cholesterol,smoker
    45,120,200,No
52,135,240,Yes
60,140,260,No
    """

    file = BytesIO(csv_data.encode("utf-8"))

    response = client.post(
        "/predict", files={"file": ("predict.csv", file, "text/csv")}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "predictions" in json_data
    assert isinstance(json_data["predictions"], list)
