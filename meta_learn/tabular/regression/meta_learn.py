# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import json
import itertools

import pandas as pd

from simple_data_collector import DataCollector


from .default_reference_datasets import dataset_feature_generator


class Paths:
    def __init__(self, path):
        self.path = path

    def get_model_dir(self, model_id):
        return os.path.join(self.path, model_id)

    def get_dataset_dir(self, model_id, dataset_id):
        return os.path.join(self.path, model_id, dataset_id)

    def create_meta_learn_dir(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def create_model_dir(self, model_id):
        path = self.get_model_dir(model_id)
        if not os.path.exists(path):
            os.makedirs(path)

    def create_dataset_dir(self, model_id, dataset_id):
        path = self.get_dataset_dir(model_id, dataset_id)
        if not os.path.exists(path):
            os.makedirs(path)


class MetaLearn:
    def __init__(self, path):
        self.path = Paths(path)

    def collect(self, X, y, model_id, dataset_id):
        print("\n collect")

        self.path.create_dataset_dir(model_id, dataset_id)

        self.dataset_dir = self.path.get_dataset_dir(model_id, dataset_id)

        search_data_path = os.path.join(self.dataset_dir, "search_data.csv")
        self.collector = DataCollector(search_data_path)
        self.dataset_features_path = os.path.join(
            self.dataset_dir, "dataset_features.json"
        )

        ref_scores = dataset_feature_generator(X, y)

        with open(self.dataset_features_path, "w") as f:
            json.dump(ref_scores, f)

        def decorator(model):
            def wrapper(access):
                parameter = access.para_dict

                result = model(access)

                if isinstance(result, tuple):
                    parameter["score"] = result[0]
                    parameter.update(result[1])
                else:
                    parameter["score"] = result

                self.collector.append(parameter)

                return result

            return wrapper

        return decorator

    def get_meta_data_X(self, search_space, X, y):
        print("\n approx_best_para")

        keys, values = zip(*search_space.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

        all_parameters = pd.DataFrame(permutations_dicts)

        ref_scores = dataset_feature_generator(X, y)
        meta_data_test = all_parameters.assign(**ref_scores)

        print("create", len(all_parameters), "samples")
        return meta_data_test

    def get_meta_data(self, model_id):
        print("\n get_meta_data")

        model_dir = self.path.get_model_dir(model_id)
        dataset_id_l = [name for name in os.listdir(model_dir)]

        meta_data_train_l = []
        for dataset_id in dataset_id_l:
            dataset_path = os.path.join(model_dir, dataset_id)

            dataset_features_path = os.path.join(dataset_path, "dataset_features.json")
            search_data_path = os.path.join(dataset_path, "search_data.csv")

            with open(dataset_features_path, "r") as f:
                ref_scores = json.load(f)

            search_data = pd.read_csv(search_data_path)

            meta_data_train = search_data.assign(**ref_scores)
            meta_data_train_l.append(meta_data_train)

        meta_data_train = pd.concat(meta_data_train_l, axis=0, ignore_index=True)
        print("Found", len(meta_data_train), "samples")

        meta_data_X = meta_data_train.drop("score", axis=1)
        meta_data_y = meta_data_train["score"]

        return meta_data_X, meta_data_y
