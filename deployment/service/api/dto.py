# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from pydantic import BaseModel


class InferenceDTO(BaseModel):
    search_space: dict
    x: dict
    y: dict


class PredictionDTO(BaseModel):
    approx_scores: list


class FeatureTypes(BaseModel):
    feature_types: dict
