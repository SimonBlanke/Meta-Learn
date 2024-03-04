# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
from joblib import load, dump
from sklearn.ensemble import GradientBoostingRegressor

from .utils import query_yes_no


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
        dump(self.m_reg, path)

    def load(self, objective_function):
        path = self.generate_path(objective_function.__name__)
        self.m_reg = load(path)

    def remove_confirmed(self, objective_function):
        path = self.generate_path(objective_function.__name__)
        os.remove(path)

    def remove(self, objective_function, always_confirm=False):
        if always_confirm:
            self.remove_confirmed(objective_function)
        else:
            question = "Remove pretrained meta regressor?"
            if query_yes_no(question):
                self.remove_confirmed(objective_function)

    def fit(self, X_meta, y_meta, drop_duplicates=True):
        if drop_duplicates:
            X_meta = X_meta.drop_duplicates()
            y_meta = y_meta.iloc[X_meta.index]

        self.m_reg.fit(X_meta, y_meta)

    def predict(self, X):
        return self.m_reg.predict(X)
