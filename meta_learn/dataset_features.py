# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import json
import pandas as pd


from .paths import SyntheticMetaDataPaths


class DatasetFeatures:
    def __init__(self, model_id: str, dataset_id: str) -> None:
        self.dataset_features_path = SyntheticMetaDataPaths.dataset_features(
            model_id, dataset_id
        )

    def load(self) -> pd.core.frame.DataFrame:
        with open(self.dataset_features_path, "r") as f:
            return json.load(f)

    def dump(self, dataset_features: dict):
        with open(self.dataset_features_path, "w") as f:
            json.dump(dataset_features, f)
