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

        path_d = {
            "dataset_type": self.dataset_type,
            "model_type": self.model_type,
            "base_path": self.base_path,
        }

        self.synth_data_path = SyntheticMetaDataPaths(**path_d)
        self.search_data_m = SearchData(**path_d)
        self.dataset_features_m = DatasetFeatures(**path_d)

    def collect(self, X, y, model_id, dataset_id):
        ref_scores = self.dataset_feature_generator.create(X, y)
        self.dataset_features_m.dump(ref_scores, model_id, dataset_id)

        def decorator(model):
            def wrapper(access):
                parameter = access.para_dict

                result = model(access)

                if isinstance(result, tuple):
                    parameter["score"] = result[0]
                    parameter.update(result[1])
                else:
                    parameter["score"] = result

                self.search_data_m.append(parameter, model_id, dataset_id)

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
        for dataset_id in self.synth_data_path.dataset_ids(model_id):
            dataset_features = self.dataset_features_m.load(model_id, dataset_id)
            search_data = self.search_data_m.load(model_id, dataset_id)

            meta_data_train = search_data.assign(**dataset_features)
            meta_data_train_l.append(meta_data_train)

        meta_data_train = pd.concat(meta_data_train_l, axis=0, ignore_index=True)
        print("Found", len(meta_data_train), "samples")

        meta_data_X = meta_data_train.drop("score", axis=1)
        meta_data_y = meta_data_train["score"]

        return meta_data_X, meta_data_y
