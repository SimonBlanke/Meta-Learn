# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

from ._recognizer import Recognizer
from ._predictor import Predictor


class MetaRegressor:
    def __init__(self):
        self.meta_reg = None
        self.score_col_name = "mean_test_score"

    def fit(self, X_train, y_train):
        self._train_regressor(X_train, y_train)

    def predict(self, X, y):
        self.recognizer = Recognizer(self.search_config)
        self.predictor = Predictor(self.search_config, self.meta_regressor_path)

        X_test = self.recognizer.get_test_metadata([X, y])

        best_hyperpara_dict, best_score = self.predictor.search(X_test)

        return best_hyperpara_dict, best_score

    def _scale(self, y_train):
        # scale the score -> important for comparison of meta data from datasets in meta regressor training
        scaler = MinMaxScaler()
        y_train = scaler.fit_transform(y_train)
        y_train = pd.DataFrame(y_train, columns=["mean_test_score"])

        return y_train

    def _train_regressor(self, X_train, y_train):
        if self.meta_reg is None:
            n_estimators = int(y_train.shape[0] / 50 + 50)

            self.meta_reg = GradientBoostingRegressor(n_estimators=n_estimators)
            self.meta_reg.fit(X_train, y_train)

    def store_model(self, path):
        joblib.dump(self.meta_reg, path)

    def load_model(self, path):
        self.meta_reg = joblib.load(path)
