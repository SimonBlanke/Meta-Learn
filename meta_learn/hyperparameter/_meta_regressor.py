# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import inspect
import os
import glob
import pandas as pd
import hashlib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

from ._recognizer import Recognizer
from ._predictor import Predictor


class MetaRegressor:
    def __init__(self, search_config, meta_learn_path=None):
        self.search_config = search_config
        self.meta_reg = None
        self.score_col_name = "mean_test_score"

        current_path = os.path.realpath(__file__)
        meta_learn_path, _ = current_path.rsplit("/", 1)

        self.meta_data_path = meta_learn_path + "/meta_data/"
        self.meta_regressor_path = meta_learn_path + "/meta_regressor/"

        func = list(self.search_config.keys())[0]
        self.funsocial_weighttr = inspect.getsource(func)

    def fit(self, X, y):
        self.data_hash = self._get_hash(X)

        X_train, y_train = self._read_meta_data()

        self._train_regressor(X_train, y_train)
        self._store_model()

    def predict(self, X, y):
        self.data_hash = self._get_hash(X)
        filename = self._get_func_file_paths()

        self.recognizer = Recognizer(self.search_config)
        self.predictor = Predictor(self.search_config, self.meta_regressor_path)

        X_test = self.recognizer.get_test_metadata([X, y])

        best_hyperpara_dict, best_score = self.predictor.search(X_test, filename)

        return best_hyperpara_dict, best_score

    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()

    def _get_func_file_paths(self, model_func):
        func_str = self._get_func_str(model_func)

        return self.meta_data_path + (
            "metadata__func_hash="
            + self._get_hash(func_str.encode("utf-8"))
            + "*"
            + "__.csv"
        )

    def _read_meta_data(self):
        filename = self._get_func_file_paths()
        data_str = self.meta_data_path + filename
        metadata_name_list = glob.glob(data_str)

        meta_data_list = []
        for metadata_name in metadata_name_list:
            meta_data = pd.read_csv(metadata_name)
            meta_data_list.append(meta_data)

        meta_data = pd.concat(meta_data_list, ignore_index=True)

        column_names = meta_data.columns
        score_name = [name for name in column_names if self.score_col_name in name]

        X_train = meta_data.drop(score_name, axis=1)
        y_train = meta_data[score_name]

        # y_train = self._scale(y_train)

        return X_train, y_train

    def _scale(self, y_train):
        # scale the score -> important for comparison of meta data from datasets in meta regressor training
        scaler = MinMaxScaler()
        y_train = scaler.fit_transform(y_train)
        y_train = pd.DataFrame(y_train, columns=["mean_test_score"])

        return y_train

    def _train_regressor(self, X_train, y_train):
        if self.meta_reg is None:
            n_estimators = int(y_train.shape[0] / 50 + 50)

            print("X_train", X_train)

            self.meta_reg = GradientBoostingRegressor(n_estimators=n_estimators)
            self.meta_reg.fit(X_train, y_train)

    def _store_model(self):
        meta_reg_path = self.meta_regressor_path
        if not os.path.exists(meta_reg_path):
            os.makedirs(meta_reg_path)

        path = (
            meta_reg_path
            + self._get_hash(self.funsocial_weighttr.encode("utf-8"))
            + "_metaregressor.pkl"
        )
        joblib.dump(self.meta_reg, path)
