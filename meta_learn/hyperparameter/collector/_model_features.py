# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd
from importlib import import_module

from sklearn.model_selection import GridSearchCV
from ..label_encoder import label_encoder_dict

from ...data_wrangler import merge_dict, get_default_hyperpara


class ModelFeatures:
    def __init__(self, search_config, cv=2, n_jobs=-1):
        self.search_config = search_config
        self.cv = cv
        self.n_jobs = n_jobs

        self.model_name = None
        self.hyperpara_dict = search_config[list(search_config.keys())[0]]

    def _get_opt_meta_data(self, _cand_list, model_func, X, y):
        results_dict = {}
        para_list = []
        score_list = []

        for _cand_ in _cand_list:
            for key in _cand_._space_.memory.keys():
                pos = np.fromstring(key, dtype=int)
                para = _cand_._space_.pos2para(pos)
                score = _cand_._space_.memory[key]

                if score != 0:
                    para_list.append(para)
                    score_list.append(score)

        results_dict["params"] = para_list
        results_dict["mean_test_score"] = score_list

        return results_dict

    def _get_grid_results(self, X, y, parameters):

        model_grid_search = GridSearchCV(
            self.model, parameters, cv=self.cv, n_jobs=self.n_jobs, verbose=1
        )
        model_grid_search.fit(X, y)

        grid_results = model_grid_search.cv_results_

        return grid_results

    def collect(self, model_func, X, y, _cand_list):
        # self.hyperpara_dict = self._get_hyperpara(model_func)

        results_dict = self._get_opt_meta_data(_cand_list, model_func, X, y)

        para_pd = pd.DataFrame(results_dict["params"])
        md_model = para_pd.reindex(sorted(para_pd.columns), axis=1)

        metric_pd = pd.DataFrame(
            results_dict["mean_test_score"], columns=["mean_test_score"]
        )

        md_model = pd.concat([para_pd, metric_pd], axis=1, ignore_index=False)
        """
        md_model = self._label_enc(md_model) * 1  # to convert False to 0 and True to 1
        """

        return md_model

    def _get_hyperpara(self, model_name):
        return label_encoder_dict[model_name]

    def _label_enc(self, X_train):
        for hyperpara_key in self.hyperpara_dict:
            to_replace = {hyperpara_key: self.hyperpara_dict[hyperpara_key]}
            X_train = X_train.replace(to_replace)
        X_train = X_train.infer_objects()

        return X_train

    def _import_model(self, model):
        sklearn, submod_func = model.rsplit(".", 1)
        module = import_module(sklearn)
        model = getattr(module, submod_func)

        return model
