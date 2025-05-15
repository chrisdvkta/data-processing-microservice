from io import StringIO
from fastapi import FastAPI, File, HTTPException, UploadFile
import pandas as pd
from app.train import train_model_df


app = FastAPI()


@app.get("/")
def read_root():
    return {"hello"}


@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="only csv files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        metrics = train_model_df(df)
        return {"message": "model trained successfully", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
