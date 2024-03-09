# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
from joblib import load, dump
from sklearn.ensemble import GradientBoostingRegressor

from .utils import query_yes_no
from .paths import MetaRegressorPaths


class BaseMetaRegressor:
    def __init__(self, regressor, base_path):
        self.regressor = regressor
        self.base_path = base_path

        if regressor == "default":
            self.m_reg = GradientBoostingRegressor()
        else:
            self.m_reg = regressor

    def _generate_path(self, model):
        path_file = os.path.join(self.path_dir, model)
        return path_file

    def dump(self, model_id):
        dump(
            self.m_reg,
            MetaRegressorPaths(self.base_path).model(
                self.dataset_type, self.model_type, model_id
            ),
        )

    def load(self, model_id):
        self.m_reg = load(
            MetaRegressorPaths(self.base_path).model(
                self.dataset_type, self.model_type, model_id
            )
        )

    def _remove_confirmed(self, model_id):
        os.remove(
            MetaRegressorPaths(self.base_path).model(
                self.dataset_type, self.model_type, model_id
            )
        )

    def remove(self, model_id, always_confirm=False):
        if always_confirm:
            self._remove_confirmed(model_id)
        else:
            question = "Remove pretrained meta regressor?"
            if query_yes_no(question):
                self._remove_confirmed(model_id)

    def fit(self, X_meta, y_meta, drop_duplicates=True):
        if drop_duplicates:
            X_meta = X_meta.drop_duplicates()
            y_meta = y_meta.iloc[X_meta.index]

        self.m_reg.fit(X_meta, y_meta)

    def predict(self, X):
        return self.m_reg.predict(X)
