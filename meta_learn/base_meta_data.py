# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import itertools
from typing import Tuple

import pandas as pd


from .search_data import SearchData
from .dataset_features import DatasetFeatures
from .paths import SyntheticMetaDataPaths


class BaseMetaData:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.dataset_feature_generator = None

    def collect(self, X, y, model_id, dataset_id):
        path_d = {
            "dataset_type": self.dataset_type,
            "model_type": self.model_type,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "base_path": self.base_path,
        }

        search_data = SearchData(**path_d)
        dataset_features = DatasetFeatures(**path_d)

        ref_scores = self.dataset_feature_generator.create(X, y)
        dataset_features.dump(ref_scores)

        def decorator(model):
            def wrapper(access):
                parameter = access.para_dict

                result = model(access)

                if isinstance(result, tuple):
                    parameter["score"] = result[0]
                    parameter.update(result[1])
                else:
                    parameter["score"] = result

                search_data.append(parameter)

                return result

            return wrapper

        return decorator

    def get_meta_data_X(self, search_space, X, y):
        keys, values = zip(*search_space.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

        all_parameters = pd.DataFrame(permutations_dicts)

        dataset_features = self.dataset_feature_generator.create(X, y)
        meta_data_test = all_parameters.assign(**dataset_features)

        print("create", len(all_parameters), "samples")
        return meta_data_test

    def get_meta_data(self, model_id: str) -> Tuple[pd.core.frame.DataFrame]:
        meta_data_train_l = []
        for dataset_id in SyntheticMetaDataPaths(self.base_path).dataset_ids(
            self.dataset_type, self.model_type, model_id
        ):
            path_d = {
                "dataset_type": self.dataset_type,
                "model_type": self.model_type,
                "model_id": model_id,
                "dataset_id": dataset_id,
                "base_path": self.base_path,
            }

            search_data_c = SearchData(**path_d)
            dataset_features_c = DatasetFeatures(**path_d)

            dataset_features = dataset_features_c.load()
            search_data = search_data_c.load()

            meta_data_train = search_data.assign(**dataset_features)
            meta_data_train_l.append(meta_data_train)

        meta_data_train = pd.concat(meta_data_train_l, axis=0, ignore_index=True)
        print("Found", len(meta_data_train), "samples")

        meta_data_X = meta_data_train.drop("score", axis=1)
        meta_data_y = meta_data_train["score"]

        return meta_data_X, meta_data_y
