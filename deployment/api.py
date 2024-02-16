from fastapi import FastAPI
from pydantic import BaseModel

from meta_learn import MetaRegressor
from meta_learn import MetaLearn


class DataPointDTO(BaseModel):
    search_space: dict
    x: dict
    y: dict


class PredictionDTO(BaseModel):
    approx_scores: list


app = FastAPI(title="Meta-Learn")


@app.on_event("startup")
def load_model():
    app.model = MetaRegressor.load(
        "tabular/classification/DecisionTreeClassifier/meta_regressor.joblib"
    )


@app.post("/predict", response_model=PredictionDTO)
def predict(inference_data: DataPointDTO):
    search_space = inference_data.search_space
    x = inference_data.x
    y = inference_data.y

    x_meta_data = MetaLearn.get_meta_data_X(search_space, x, y)
    prediction = app.model.predict(x_meta_data)

    return PredictionDTO(approx_scores=prediction)
