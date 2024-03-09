# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os

from typing import List

from .paths import Paths


class SyntheticMetaDataPaths(Paths):
    search_data: str = "search_data.csv"
    dataset_features: str = "dataset_features.json"

    def __init__(self, base_path=None) -> None:
        super().__init__(base_path)

        self.synthetic_meta_data_base_path = os.path.join(
            self.pkg_data, "synthetic_meta_data"
        )

    @Paths.create_dir
    def model_dir(self, dataset_type, model_type):
        return os.path.join(
            self.synthetic_meta_data_base_path, dataset_type, model_type
        )

    @Paths.create_dir
    def dataset(self, model_id, dataset_id):
        return os.path.join(self.model(model_id), dataset_id)

    @Paths.create_dir
    def model(self, model_id):
        return os.path.join(self.synthetic_meta_data_base_path, model_id)

    def dataset_ids(self, dataset_type, model_type, model_id: str) -> List[str]:
        model_path = os.path.join(
            self.synthetic_meta_data_base_path,
            dataset_type,
            model_type,
            model_id,
        )
        return [name for name in os.listdir(model_path)]

    @Paths.create_dir
    def model_dataset(self, dataset_type, model_type, model_id, dataset_id):
        return os.path.join(
            self.synthetic_meta_data_base_path,
            self.model_dir(dataset_type, model_type),
            model_id,
            dataset_id,
        )

    def dataset_features(self, dataset_type, model_type, model_id, dataset_id):
        return os.path.join(
            self.model_dataset(
                dataset_type,
                model_type,
                model_id,
                dataset_id,
            ),
            "dataset_features.json",
        )

    def search_data(self, dataset_type, model_type, model_id, dataset_id):
        return os.path.join(
            self.model_dataset(
                dataset_type,
                model_type,
                model_id,
                dataset_id,
            ),
            "search_data.json",
        )
