from fastapi.testclient import TestClient
from app.main import app
from io import BytesIO

client = TestClient(app)


def test_prediction():
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
