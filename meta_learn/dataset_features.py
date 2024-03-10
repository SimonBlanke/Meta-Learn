# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import json
import pandas as pd


from .paths import SyntheticMetaDataPaths


class DatasetFeatures:
    def __init__(
        self,
        dataset_type: str,
        model_type: str,
        base_path: str = None,
    ) -> None:
        self.synth_data_path = SyntheticMetaDataPaths(
            dataset_type, model_type, base_path
        )

    def load(self, model_id, dataset_id) -> pd.core.frame.DataFrame:
        path2csv = self.synth_data_path.dataset_features(model_id, dataset_id)
        with open(path2csv, "r") as f:
            return json.load(f)

    def dump(self, dataset_features: dict, model_id, dataset_id):
        path2csv = self.synth_data_path.dataset_features(model_id, dataset_id)
        with open(path2csv, "w") as f:
            json.dump(dataset_features, f)
