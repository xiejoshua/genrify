import joblib
from pathlib import Path

# Path(__file__) is this file. .parent is app/, .parent.parent is backend/, so this points at backend/models/<file>. Built from __file__ instead of "models/..." so it works no matter which folder uvicorn is started from, on your laptop and inside the container
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "genre_classifier.pkl" # the file name is "genre_classifier.pkl"

# the order the model was trained on. scikit-learn matches columns by position, not by name, so a swapped pair still returns plausible-looking wrong answers. Do not reorder
FEATURE_ORDER = (
    "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "time_signature", "duration_ms",
    "popularity", "explicit",
)


class Predictor:
    source = "model" # reported in every response, so a caller can tell a real prediction from the old "mock"

    def __init__(self, path: Path) -> None:
        # joblib.load: reads the .pkl back into Python objects. The notebook wrote it with pickle.dump, and joblib can read that format too. Loading is slow compared to a request, which is why main.py does it once at startup. Only load files you trust: a pickle can run code while it loads. This one is safe because the path is fixed in this file, a model we trained and shipped ourselves, and never comes from a request
        bundle = joblib.load(path)
        self.model = bundle["model"] # the trained LogisticRegression
        # "scaler": the StandardScaler. The model was trained on scaled numbers, so raw input must be scaled the same way first. The other keys are "model", "label_encoder", "feature_cols", "classes"
        self.scaler = bundle["scaler"]
        # classes_: the genre names in the order the model's five outputs come in. LabelEncoder sorts them alphabetically: acoustic, edm, hip-hop, pop, rock
        self.genres = list(bundle["label_encoder"].classes_)
        # the scaler remembers the column names it was trained with, so passing a plain list makes scikit-learn warn on every request. We already control the order, so drop the names
        del self.scaler.feature_names_in_

    def predict(self, vector: list[float]) -> dict[str, float]:
        # transform: apply the training-time scaling. Not fit_transform, which would re-learn the scaling from this one row. The extra [ ] makes it a table with one row, because scikit-learn expects rows of samples
        scaled = self.scaler.transform([vector])
        # predict_proba returns one row per sample, with one probability per genre, so its shape is (1, 5). [0] takes the only row, leaving 5 numbers that add up to 1
        probabilities = self.model.predict_proba(scaled)[0]
        return {genre: float(p) for genre, p in zip(self.genres, probabilities)}
