# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os

from .paths import Paths


class MetaRegressorPaths(Paths):
    def __init__(self, dataset_type, model_type, base_path=None) -> None:
        super().__init__(dataset_type, model_type, base_path)

        self.base_path = self._meta_regressors_base_path()

    @Paths.create_dir
    def _meta_regressors_base_path(self):
        return os.path.join(self.study_type_path, "meta_regressors")

    def model(self, model_id):
        return os.path.join(self.base_path, model_id + ".joblib")

    @Paths.create_dir
    def dataset(self, model_id, dataset_id):
        return os.path.join(self.base_path, model_id, dataset_id)

    def get_objective_function_names(self):
        path_dir = os.path.abspath(
            os.path.join(self.base_path, self.dataset_type, self.model_type)
        )
        paths_l = os.listdir(self.path_dir)
        return [path.split(".joblib")[0] for path in paths_l]
