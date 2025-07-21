# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
from joblib import load, dump
from sklearn.ensemble import GradientBoostingRegressor

from .utils import query_yes_no
from .paths import MetaRegressorPaths


class BaseMetaRegressor:
    """Base class for meta-regressors in the meta-learning framework.

    This class manages the training, persistence, and prediction capabilities
    of meta-regressors that learn to predict model performance based on
    hyperparameters and dataset features. It supports custom regressors or
    defaults to GradientBoostingRegressor.

    Attributes:
        regressor: Either "default" or a scikit-learn compatible regressor instance
        m_reg: The actual regressor instance used for meta-learning
        meta_reg_paths (MetaRegressorPaths): Path manager for saving/loading models
    """

    def __init__(self, regressor, base_path):
        self.regressor = regressor

        self.meta_reg_paths = MetaRegressorPaths(
            self.dataset_type, self.model_type, base_path
        )

        if regressor == "default":
            self.m_reg = GradientBoostingRegressor()
        else:
            self.m_reg = regressor

    def _generate_path(self, model):
        path_file = os.path.join(self.path_dir, model)
        return path_file

    def dump(self, model_id):
        """Save the trained meta-regressor to disk.

        Serializes the meta-regressor using joblib for efficient storage
        and later retrieval.

        Args:
            model_id (str): Unique identifier for the model type
        """

        dump(
            self.m_reg,
            self.meta_reg_paths.model(model_id),
        )

    def load(self, model_id):
        """Load a previously trained meta-regressor from disk.

        Args:
            model_id (str): Unique identifier for the model type
        """

        self.m_reg = load(self.meta_reg_paths.model(model_id))

    def _remove_confirmed(self, model_id):
        os.remove(self.meta_reg_paths.model(model_id))

    def remove(self, model_id, always_confirm=False):
        """Remove a trained meta-regressor with optional confirmation.

        Args:
            model_id (str): Unique identifier for the model type
            always_confirm (bool): If True, skip confirmation prompt
        """

        if always_confirm:
            self._remove_confirmed(model_id)
        else:
            question = "Remove pretrained meta regressor?"
            if query_yes_no(question):
                self._remove_confirmed(model_id)

    def fit(self, X_meta, y_meta, drop_duplicates=True):
        """Train the meta-regressor on meta-data.

        Fits the regressor to learn the relationship between hyperparameters,
        dataset features, and model performance.

        Args:
            X_meta (pd.DataFrame): Features including hyperparameters and dataset characteristics
            y_meta (pd.Series): Target performance scores
            drop_duplicates (bool): Whether to remove duplicate feature rows before training
        """

        if drop_duplicates:
            X_meta = X_meta.drop_duplicates()
            y_meta = y_meta.iloc[X_meta.index]

        self.m_reg.fit(X_meta, y_meta)

    def predict(self, X):
        """Generate performance predictions using the trained meta-regressor.

        Args:
            X (pd.DataFrame): Features for which to predict performance

        Returns:
            np.ndarray: Predicted performance scores
        """

        return self.m_reg.predict(X)
