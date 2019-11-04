# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os

from sklearn.externals import joblib

from .label_encoder import label_encoder_dict
from ..data_wrangler import find_best_hyperpara


class Predictor:
    def __init__(self):
        pass

    def search(self, X_test):
        best_para, best_score = self._predict(X_test)
        return best_para, best_score

    def load_model(self, path):
        if os.path.isfile(path):
            self.meta_reg = joblib.load(path)
        else:
            print("File at path", path, "not found")

    def _predict(self, X_test):
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
