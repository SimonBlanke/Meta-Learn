# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os

from .paths import Paths


class MetaRegressorPaths(Paths):
    """Path manager specifically for meta-regressor storage.

    This class extends the base Paths class to provide specific path
    generation for trained meta-regressors and their associated data.

    Attributes:
        base_path (str): Root directory for meta-regressor storage
    """

    def __init__(self, dataset_type, model_type, base_path=None) -> None:
        super().__init__(dataset_type, model_type, base_path)

        self.base_path = self._meta_regressors_base_path()

    @Paths.create_dir
    def _meta_regressors_base_path(self):
        return os.path.join(self.study_type_path, "meta_regressors")

    def model(self, model_id):
        """Generate path for a specific meta-regressor file.

        Args:
            model_id (str): Unique identifier for the model

        Returns:
            str: Complete path to the joblib file
        """

        return os.path.join(self.base_path, model_id + ".joblib")

    @Paths.create_dir
    def dataset(self, model_id, dataset_id):
        """Generate path for model-dataset combination.

        Args:
            model_id (str): Unique identifier for the model
            dataset_id (str): Unique identifier for the dataset

        Returns:
            str: Path to the dataset directory
        """

        return os.path.join(self.base_path, model_id, dataset_id)

    def get_objective_function_names(self):
        """Retrieve list of available objective function names.

        Scans the directory structure to find all saved meta-regressors
        and extracts their objective function names.

        Returns:
            list: Names of objective functions with saved meta-regressors
        """

        path_dir = os.path.abspath(
            os.path.join(self.base_path, self.dataset_type, self.model_type)
        )
        paths_l = os.listdir(self.path_dir)
        return [path.split(".joblib")[0] for path in paths_l]
