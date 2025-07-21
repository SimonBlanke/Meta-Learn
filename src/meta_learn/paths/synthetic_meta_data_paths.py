# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os

from typing import List

from .paths import Paths


class SyntheticMetaDataPaths(Paths):
    search_data: str = "search_data.csv"
    dataset_features: str = "dataset_features.json"

    def __init__(self, dataset_type, model_type, base_path=None) -> None:
        super().__init__(dataset_type, model_type, base_path)

        self.base_path = self._synthetic_meta_data_base_path()

    @Paths.create_dir
    def _synthetic_meta_data_base_path(self):
        return os.path.join(self.study_type_path, "synthetic_meta_data")

    @Paths.create_dir
    def dataset(self, model_id, dataset_id):
        return os.path.join(self.model(model_id), dataset_id)

    @Paths.create_dir
    def model(self, model_id):
        return os.path.join(self.base_path, model_id)

    def dataset_ids(self, model_id: str) -> List[str]:
        model_path = os.path.join(
            self.base_path,
            model_id,
        )
        return [name for name in os.listdir(model_path)]

    @Paths.create_dir
    def model_dataset(self, model_id, dataset_id):
        return os.path.join(
            self.base_path,
            model_id,
            dataset_id,
        )

    def dataset_features(self, model_id, dataset_id):
        return os.path.join(
            self.model_dataset(
                model_id,
                dataset_id,
            ),
            "dataset_features.json",
        )

    def search_data(self, model_id, dataset_id):
        return os.path.join(
            self.model_dataset(
                model_id,
                dataset_id,
            ),
            "search_data.json",
        )
