# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd


class ModelFeatures:
    def __init__(self):
        pass

    def _get_opt_meta_data(self, _cand_, X, y):
        results_dict = {}
        para_list = []
        score_list = []

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

    def collect(self, X, y, _cand_):
        results_dict = self._get_opt_meta_data(_cand_, X, y)

        para_pd = pd.DataFrame(results_dict["params"])
        md_model = para_pd.reindex(sorted(para_pd.columns), axis=1)

        metric_pd = pd.DataFrame(
            results_dict["mean_test_score"], columns=["mean_test_score"]
        )

        md_model = pd.concat([para_pd, metric_pd], axis=1, ignore_index=False)

        return md_model
