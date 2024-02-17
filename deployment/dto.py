from pydantic import BaseModel


class InferenceDTO(BaseModel):
    search_space: dict
    x: dict
    y: dict


class PredictionDTO(BaseModel):
    approx_scores: list