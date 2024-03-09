# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import shutil

from typing import List

from .paths import Paths
from .utils import query_yes_no


class SyntheticMetaDataPaths(Paths):
    search_data: str = "search_data.csv"
    dataset_features: str = "dataset_features.json"

    synthetic_meta_data_base_path: str = os.path.join(
        Paths.pkg_data, "synthetic_meta_data"
    )

    def remove(self, model_id=None, dataset_id=None, always_confirm=False):
        if always_confirm:
            self._remove_confirmed(model_id, dataset_id)
        else:
            question = "Remove synthetic meta data?"
            if query_yes_no(question):
                self._remove_confirmed(model_id, dataset_id)

    def _remove_confirmed(cls, model_id, dataset_id):
        if model_id and dataset_id:
            shutil.rmtree(cls.dataset(model_id, dataset_id))
        elif model_id:
            shutil.rmtree(cls.model(model_id))
        elif not model_id and not dataset_id:
            shutil.rmtree(cls.synthetic_meta_data_base_path)
        else:
            raise ValueError

    def dataset(cls, model_id, dataset_id):
        return os.path.join(cls.model(model_id), dataset_id)

    def model(cls, model_id):
        return os.path.join(cls.synthetic_meta_data_base_path, model_id)

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
