# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
from joblib import load, dump
from sklearn.ensemble import GradientBoostingRegressor


class BaseMetaRegressor:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(dir_path, "pretrained_meta_regressors")

    def __init__(self, regressor):
        self.regressor = regressor

        if regressor == "default":
            self.m_reg = GradientBoostingRegressor()
        else:
            self.m_reg = regressor

    def generate_path(self, model):
        path_dir = os.path.abspath(
            os.path.join(self.base_path, self.dataset_type, self.model_type)
        )
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        path_file = os.path.join(path_dir, model)
        return path_file

    def dump(self, objective_function):
        path = self.generate_path(objective_function.__name__)
        dump(self, path)

    def load(self, objective_function):
        path = self.generate_path(objective_function.__name__)
        return load(path)

    def fit(self, X_meta, y_meta, drop_duplicates=True):
        self.feature_types = X_meta.dtypes.astype(str).to_dict()

        if drop_duplicates:
            X_meta = X_meta.drop_duplicates()
            y_meta = y_meta.iloc[X_meta.index]

        self.m_reg.fit(X_meta, y_meta)

    def predict(self, X):
        return self.m_reg.predict(X)
