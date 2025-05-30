from io import StringIO
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
import pandas as pd
from app.predict import predict_from_df
from app.train import train_model_df


app = FastAPI()


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Health Risk Predictor API",
        "endpoints": {
            "train": "/train (POST) - Upload a CSV file to train a model",
            "predict": "/predict (POST) - Upload a CSV file to get predictions",
            "docs": "/docs - Swagger UI for interactive API documentation",
        },
        "default_model": "default",
        "note": "You can specify ?model=your_model_name in both /train and /predict",
    }


@app.post("/train")
async def train_model(file: UploadFile = File(...), model: str = Query("default")):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="only csv files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        metrics = train_model_df(df, model_name=model)
        return {"message": "model trained successfully", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Query("default")):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
        predictions = predict_from_df(df, model_name=model)
        return {"message": "Here's the predictions", "predictions": predictions}

    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="no trained model found. train a model first"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
