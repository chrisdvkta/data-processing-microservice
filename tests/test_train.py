from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_training():
    with open("tests/sample.csv", "rb") as f:
        response = client.post("/train", files={"file": ("sample.csv", f, "text/csv")})
    assert response.status_code == 200
    assert "message" in response.json()
