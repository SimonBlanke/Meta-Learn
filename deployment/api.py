from fastapi import FastAPI

from .dto import InferenceDTO, PredictionDTO


from meta_learn import MetaRegressor
from meta_learn import MetaLearn


app = FastAPI(title="Meta-Learn")


@app.on_event("startup")
def load_model():
    app.model = MetaRegressor.load("meta_regressor.joblib")


@app.post("/predict", response_model=PredictionDTO)
def predict(inference_data: InferenceDTO):
    search_space = inference_data.search_space
    x = inference_data.x
    y = inference_data.y

    x_meta_data = MetaLearn.get_meta_data_X(search_space, x, y)
    prediction = app.model.predict(x_meta_data)

    return PredictionDTO(approx_scores=prediction)
