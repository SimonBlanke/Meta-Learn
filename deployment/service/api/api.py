# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from fastapi import FastAPI
from argparse import ArgumentParser

from .dto import InferenceDTO, PredictionDTO

from .importer import MetaRegressorImporter
from meta_learn.tabular.regression import MetaLearn


parser = ArgumentParser(description="Meta-Regressor arguments")
parser.add_argument(
    "-dataset_type", type=str, default=1, help="Dataset type (e.g. 'tabular')"
)
parser.add_argument(
    "-model_type", type=str, default=1, help="Model type (e.g. 'regression')"
)
parser.add_argument(
    "-model_name",
    type=str,
    default=1,
    help="Model name (e.g. 'decision_tree_classifier')",
)

app = FastAPI(title="Meta-Learn")


@app.on_event("startup")
def load_model():
    args = parser.parse_args()
    MetaRegressor = MetaRegressorImporter(
        args.dataset_type, args.model_type
    ).meta_regressor
    app.model = MetaRegressor().load(args.model_name + ".joblib")


@app.post("/predict", response_model=PredictionDTO)
def predict(inference_data: InferenceDTO):
    search_space = inference_data.search_space
    x = inference_data.x
    y = inference_data.y

    x_meta_data = MetaLearn.get_meta_data_X(search_space, x, y)
    prediction = app.model.predict(x_meta_data)

    return PredictionDTO(approx_scores=prediction)
