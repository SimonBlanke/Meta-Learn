# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import joblib
from fastapi import FastAPI
from argparse import ArgumentParser

from .dto import InferenceDTO, PredictionDTO, FeatureTypes

from meta_learn.tabular.regression import MetaLearn


parser = ArgumentParser(description="Meta-Regressor arguments")
parser.add_argument(
    "-path2model",
    type=str,
    default=None,
    help="Path 2 meta-regressor model (.joblib)",
)

app = FastAPI(title="Meta-Learn")


@app.on_event("startup")
def load_model():
    args = parser.parse_args()
    app.model = joblib.load(args.path2model)


@app.post("/feature_types")
def predict():
    return FeatureTypes(feature_types=app.model.feature_types)


@app.post("/predict", response_model=PredictionDTO)
def predict(inference_data: InferenceDTO):
    search_space = inference_data.search_space
    x = inference_data.x
    y = inference_data.y

    x_meta_data = MetaLearn.get_meta_data_X(search_space, x, y)
    prediction = app.model.predict(x_meta_data)

    return PredictionDTO(approx_scores=prediction)
