from fastapi.testclient import TestClient
from app.main import app
from io import BytesIO

client = TestClient(app)


def test_prediction():
    training_csv = """age,blood_pressure,cholestrol,smoker,risk
45,120,200,No,Low
52,135,240,Yes,High
60,140,260,No,High
38,115,190,No,Low
"""

    train_file = BytesIO(training_csv.encode("utf-8"))
    response_train = client.post(
        "/train", files={"file": ("train.csv", train_file, "text/csv")}
    )
    assert response_train.status_code == 200

    # without a label
    csv_data = """age,blood_pressure,cholestrol,smoker
45,120,200,No
52,135,240,Yes
60,140,260,No
"""
    predict_file = BytesIO(csv_data.encode("utf-8"))
    response = client.post(
        "/predict", files={"file": ("predict.csv", predict_file, "text/csv")}
    )

    assert response.status_code == 200
    json_data = response.json()
    assert "predictions" in json_data
    assert isinstance(json_data["predictions"], list)


def test_train_and_predict_multiple_models():
    train_csv = """age,blood_pressure,cholesterol,smoker,risk
45,123,200,No,0
52,134,240,Yes,1
60,170,260,No,0
"""

    train_file_heart = BytesIO(train_csv.encode("utf-8"))
    response = client.post(
        "/train?model=heart-risk",
        files={"file": ("heart.csv", train_file_heart, "text/csv")},
    )
    assert response.status_code == 200
    assert "trained" in response.json().get("message", "").lower()

    train_file_lung = BytesIO(train_csv.encode("utf-8"))
    response = client.post(
        "/train?model=lung-risk",
        files={"file": ("lung.csv", train_file_lung, "text/csv")},
    )

    assert response.status_code == 200

    predict_csv = """age,blood_pressure,cholesterol,smoker
50,130,210,No
55,140,250,Yes
"""

    predict_file_heart = BytesIO(predict_csv.encode("utf-8"))
    response = client.post(
        "/predict?model=heart-risk",
        files={"file": ("predict_heart.csv", predict_file_heart, "text/csv")},
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "predictions" in json_data
    assert isinstance(json_data["predictions"], list)

    predict_file_lung = BytesIO(predict_csv.encode("utf-8"))
    response = client.post(
        "/predict?model=lung-risk",
        files={"file": ("predict_lung.csv", predict_file_lung, "text/csv")},
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "predictions" in json_data
    assert isinstance(json_data["predictions"], list)
