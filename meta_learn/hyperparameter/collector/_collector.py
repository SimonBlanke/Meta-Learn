# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import inspect
import os
import glob
import pandas as pd
import hashlib

from ...data_wrangler import merge_meta_data

from ._dataset_features import DatasetFeatures
from ._model_features import ModelFeatures


class Collector:
    def __init__(self, search_config, n_jobs=1, meta_data_path=None):
        self.search_config = search_config
        self.n_jobs = n_jobs
        self.meta_data_path = meta_data_path

        self.score_col_name = "mean_test_score"

        self.collector_model = ModelFeatures(self.search_config)

    def extract(self, X, y, _cand_list):
        self.collector_dataset = DatasetFeatures()

        for model_func in self.search_config.keys():
            meta_data = self._get_meta_data(model_func, [X, y], _cand_list)
            path = self._get_file_path(X, y, model_func)

            self._save_toCSV(meta_data, path)

    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()

    def _get_func_str(self, func):
        return inspect.getsource(func)

    def _get_meta_data(self, model_name, data_train, _cand_list):
        X, y = data_train[0], data_train[1]

        md_model = self.collector_model.collect(model_name, X, y, _cand_list)
        md_dataset = self.collector_dataset.collect(model_name, data_train)

        meta_data = merge_meta_data(md_dataset, md_model)

        return meta_data

    def _get_func_metadata(self, paths):
        meta_data_list = []
        for path in paths:
            meta_data = pd.read_csv(path)
            meta_data_list.append(meta_data)

        if len(meta_data_list) > 0:
            meta_data = pd.concat(meta_data_list, ignore_index=True)

            column_names = meta_data.columns
            score_name = [name for name in column_names if self.score_col_name in name]

            para = meta_data.drop(score_name, axis=1)
            score = meta_data[score_name]

            return para, score

    def _get_func_file_paths(self, model_func):
        func_str = self._get_func_str(model_func)

        return self.meta_data_path + (
            "metadata__func_hash="
            + self._get_hash(func_str.encode("utf-8"))
            + "*"
            + "__.csv"
        )

    def _get_file_path(self, X_train, y_train, model_func):
        func_str = self._get_func_str(model_func)
        feature_hash = self._get_hash(X_train)
        label_hash = self._get_hash(y_train)

        return self.meta_data_path + (
            "metadata__func_hash="
            + self._get_hash(func_str.encode("utf-8"))
            + "__feature_hash="
            + feature_hash
            + "__label_hash="
            + label_hash
            + "__.csv"
        )

    def _save_toCSV(self, meta_data_new, path):
        if not os.path.exists(self.meta_data_path):
            os.makedirs(self.meta_data_path)

        # print("meta_data_new", meta_data_new)

        if os.path.exists(path):
            meta_data_old = pd.read_csv(path)
            meta_data = meta_data_old.append(meta_data_new)

            columns = list(meta_data.columns)
            noScore = ["mean_test_score", "cv_default_score"]
            columns_noScore = [c for c in columns if c not in noScore]

            meta_data = meta_data.drop_duplicates(subset=columns_noScore)
        else:
            meta_data = meta_data_new

        meta_data.to_csv(path, index=False)
