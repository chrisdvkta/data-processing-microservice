from contextlib import asynccontextmanager
from fastapi import FastAPI
from sklearn.linear_model import LogisticRegression

from app import model_utils
from app.model_utils import load_model, save_model
from app.train import test


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = LogisticRegression()
    model_utils.save_model(model)

    # Load model
    loaded_model = model_utils.load_model()
    print("Model loaded:", loaded_model)

    yield  # This is where FastAPI serves requests


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"hello"}


# @app.get("/hello")
# def save():
#     return test()
