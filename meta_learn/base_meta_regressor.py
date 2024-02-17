# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import pandas as pd

from joblib import load, dump
from sklearn.ensemble import GradientBoostingRegressor
from category_encoders import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted


class BaseMetaRegressor:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(dir_path, "pretrained_meta_regressors")

    def __init__(self, regressor, encoder):
        self.regressor = regressor
        self.encoder = encoder

        if encoder == "default":
            self.enc = OrdinalEncoder()
        else:
            self.enc = encoder

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

    def dump(self, model):
        path = self.generate_path(model)
        dump(self.regressor, path)

    def load(self, model):
        path = self.generate_path(model)
        return load(path)

    def fit(self, X_meta, y_meta, drop_duplicates=True):
        X_numeric = X_meta.apply(
            lambda s: pd.to_numeric(s, errors="coerce").notnull().all()
        )

        if not X_numeric.all():
            print("\n encode")
            self.enc.fit(X_meta)
            X_meta = self.enc.transform(X_meta)
        else:
            self.encoder = False

        if drop_duplicates:
            X_meta = X_meta.drop_duplicates()
            y_meta = y_meta.iloc[X_meta.index]

        self.m_reg.fit(X_meta, y_meta)

    def predict(self, X):
        if self.encoder:
            X = self.enc.transform(X)
            X_pred = self.m_reg.predict(X)
            X_pred = self.enc.inverse_transform(X_pred)
        else:
            X_pred = self.m_reg.predict(X)

        return X_pred
