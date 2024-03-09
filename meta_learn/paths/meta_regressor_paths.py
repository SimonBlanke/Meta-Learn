# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os

from .paths import Paths


class MetaRegressorPaths(Paths):
    def __init__(self, base_path=None) -> None:
        super().__init__(base_path)

        self.meta_regressors_base_path = os.path.join(self.pkg_data, "meta_regressors")

    @Paths.create_dir
    def model_dir(self, dataset_type, model_type):
        return os.path.join(self.meta_regressors_base_path, dataset_type, model_type)

    def model(self, dataset_type, model_type, model_id):
        return os.path.join(self.model_dir(dataset_type, model_type), model_id)

    @Paths.create_dir
    def dataset(self, model_id, dataset_id):
        return os.path.join(self.meta_regressors_base_path, model_id, dataset_id)

    def get_objective_function_names(self):
        path_dir = os.path.abspath(
            os.path.join(self.base_path, self.dataset_type, self.model_type)
        )
        paths_l = os.listdir(self.path_dir)
        return [path.split(".joblib")[0] for path in paths_l]
