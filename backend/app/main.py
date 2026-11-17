from contextlib import asynccontextmanager

import sklearn
from fastapi import FastAPI, Request

from .predictor import FEATURE_ORDER, MODEL_PATH, Predictor
from .schemas import PredictRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once, when the server starts, before the first request. The model is stored on app.state so every request can reuse it. Loading it inside predict() would unpickle the file again on every request, and the UI sends one request per slider move
    app.state.predictor = Predictor(MODEL_PATH)
    yield

app = FastAPI(title="Genrify Genre API", version="0.1.0", lifespan=lifespan) # lifespan=lifespan: hands FastAPI the startup function above. Without it, the model never loads and predict() fails on the first request

@app.get("/health", tags=["meta"])
def health(request: Request):
    predictor = request.app.state.predictor
    return {"status": "ok", "source": predictor.source, "sklearn_version": sklearn.__version__} # "model", no longer "mock". The version shows which scikit-learn is running, which must match the one that wrote the .pkl

@app.post("/predict", tags=["prediction"])
def predict(payload: PredictRequest, request: Request):
    predictor = request.app.state.predictor
    values = payload.model_dump()
    # FEATURE_ORDER: a dict is not a reliable way to hand the model its columns, so we build the list in the exact training order. float() turns the bool explicit into 1.0 or 0.0
    vector = [float(values[name]) for name in FEATURE_ORDER]
    probabilities = predictor.predict(vector)
    genre = max(probabilities, key=probabilities.get) # .get: max() asks it for each genre's probability and returns the genre with the largest one
    return {
        "genre": genre,
        "confidence": probabilities[genre], # the probability of the winning genre, so it always equals that entry below
        "probabilities": probabilities,
        "source": predictor.source, # .source: "model". The response shape is the same as the mock's, so the frontend needs no change
    }
