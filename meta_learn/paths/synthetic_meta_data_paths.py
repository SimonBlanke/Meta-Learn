# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os

from types import List

from .paths import Paths


class SyntheticMetaDataPaths(Paths):
    search_data: str = "search_data.csv"
    dataset_features: str = "dataset_features.json"

    synthetic_meta_data_base_path: str = os.path.join(
        Paths.pkg_data, "synthetic_meta_data"
    )

    def dataset_ids(cls, model_id: str) -> List[str]:
        model_path = os.path.join(cls.synthetic_meta_data_base_path, model_id)
        return [name for name in os.listdir(model_path)]

    def model_dataset(cls, model_id, dataset_id):
        return os.path.join(cls.synthetic_meta_data_base_path, model_id, dataset_id)

    @Paths.create_dir
    @classmethod
    def dataset_features(cls, model_id, dataset_id):
        return os.path.join(
            cls.model_dataset(model_id, dataset_id), "dataset_features.json"
        )

    @Paths.create_dir
    @classmethod
    def search_data(cls, model_id, dataset_id):
        return os.path.join(cls.model_dataset(model_id, dataset_id), "search_data.json")
