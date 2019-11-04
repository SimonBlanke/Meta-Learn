# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import pandas as pd

from ...data_wrangler import merge_meta_data

from ._dataset_features import DatasetFeatures
from ._model_features import ModelFeatures


class Collector:
    def __init__(self):
        self.score_col_name = "mean_test_score"
        self.collector_model = ModelFeatures()

    def extract(self, X, y, _cand_, path):
        self.collector_dataset = DatasetFeatures()

        meta_data = self._get_meta_data([X, y], _cand_)
        self._save_toCSV(meta_data, path)

    def _get_meta_data(self, data_train, _cand_):
        X, y = data_train[0], data_train[1]

        md_model = self.collector_model.collect(X, y, _cand_)
        md_dataset = self.collector_dataset.collect(data_train)

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

        else:
            return None, None

    def _save_toCSV(self, meta_data_new, path):
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
