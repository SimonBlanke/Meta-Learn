# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import hashlib
import inspect

from sklearn.externals import joblib

from .label_encoder import label_encoder_dict
from ..data_wrangler import find_best_hyperpara


class Predictor:
    def __init__(self, search_config, meta_reg_path):
        self.search_config = search_config
        self.meta_reg_path = meta_reg_path

        self.model_list = list(self.search_config.keys())
        self.model_str = self.model_list[0]

        current_path = os.path.realpath(__file__)
        meta_learn_path, _ = current_path.rsplit("/", 1)

        self.meta_data_path = meta_learn_path + "/meta_data/"
        self.meta_regressor_path = meta_learn_path + "/meta_regressor/"

        func = list(self.search_config.keys())[0]
        self.funsocial_weighttr = inspect.getsource(func)

    def search(self, X_test, model_func):
        # self._load_model(model_func)

        best_para, best_score = self._predict(X_test)
        return best_para, best_score

    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()

    def load_model(self, model_func):
        path = self._get_metaReg_file_path(model_func)
        if os.path.isfile(path):
            self.meta_reg = joblib.load(path)
        else:
            print("File at path", path, "not found")

    def _get_func_str(self, func):
        return inspect.getsource(func)

    def _get_metaReg_file_path(self, model_func):
        func_str = self._get_func_str(model_func)

        return self.meta_regressor_path + (
            "metamodel__func_hash="
            + self._get_hash(func_str.encode("utf-8"))
            + "__.csv"
        )

    def _predict(self, X_test):
        # X_test = self._label_enconding(X_test)
        # print(X_test.info())
        score_pred = self.meta_reg.predict(X_test)
        best_features, best_score = find_best_hyperpara(X_test, score_pred)

        return best_features, best_score

    def _decode_hyperpara_dict(self, para):
        for hyperpara_key in label_encoder_dict[self.model_str]:
            if hyperpara_key in para:
                inv_label_encoder_dict = {
                    v: k
                    for k, v in label_encoder_dict[self.model_str][
                        hyperpara_key
                    ].items()
                }

                encoded_values = para[hyperpara_key]
                para[hyperpara_key] = inv_label_encoder_dict[encoded_values]

        return para

    def _label_enconding(self, X_train):
        for hyperpara_key in self.para:
            to_replace = {hyperpara_key: self.para[hyperpara_key]}
            X_train = X_train.replace(to_replace)
        X_train = X_train.infer_objects()

        return X_train
