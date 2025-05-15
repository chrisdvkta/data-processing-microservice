# 🧠 CSV Health Risk Predictor

A FastAPI microservice that trains and predicts health risk levels from CSV data. Supports multiple model versions, built with Docker, and easily deployable with CI/CD.

---

## 🚀 Features

- Upload CSVs to **train** classification models
- Upload CSVs without labels to **predict** health risks
- Supports **multiple models** via query parameters (`?model=heart-risk`)
- Returns predictions as booleans and human-readable strings
- Built with **FastAPI**, **scikit-learn**, and **pandas**
- Ready for Docker + GitHub Actions + AWS deployment (soon)

---

## 📦 How to Run

### 🖥️ Locally (with Python)

1. **Clone the repository**

2. **Install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the app**

   ```bash
   uvicorn app.main:app --reload
   ```

4. **Open in browser**

   Go to http://localhost:8000/docs for the interactive Swagger UI.

### 🐳 Using Docker

1. **Build the Docker image**

   ```bash
   docker build -t csv-trainer .
   ```

2. **Run the container**

   ```bash
   docker run -p 8000:8000 csv-trainer
   ```

3. **Open in browser**

   Visit http://localhost:8000/docs

---

## 📋 Data Format

### Training Data

#### Sample Training data has been provided in /app/data/training_data.csv

```csv
age,blood_pressure,cholesterol,smoker,risk
45,120,200,No,0
52,135,240,Yes,1
60,140,260,No,1
38,115,190,No,0
```

### Prediction Data

#### Sample Prediction data has been provided in /app/data/testing_data.csv

```csv
age,blood_pressure,cholesterol,smoker
45,120,200,No
52,135,240,Yes
```

**Notes:**

- `smoker` column must contain `"Yes"` or `"No"`
- `risk` must be `0` (not at risk) or `1` (at risk)

---

## 🧪 API Endpoints

| Method | Endpoint   | Description                     |
| ------ | ---------- | ------------------------------- |
| GET    | `/`        | Health check route              |
| POST   | `/train`   | Train a model with labeled data |
| POST   | `/predict` | Predict using trained model     |

### Model Parameter

You can pass an optional model name as a query parameter:

- `/train?model=heart-risk`
- `/predict?model=lung-risk`

Default model name is `"default"`.

### Configuration Notes

- Models are saved in the `app/models/` directory with `.pkl` extension
- If no model is found during prediction, a 404 error is returned
- You can add as many models as you want by changing the query parameter

### Running Tests

```bash
pytest
```

Tests are located in the `tests/` directory and cover full training + prediction flow.

---
