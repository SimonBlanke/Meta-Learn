# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os

from .paths import Paths


class MetaRegressorPaths(Paths):
    meta_regressors_path: str = os.path.join(Paths.pkg_data, "meta_regressors")

    @Paths.create_dir
    @classmethod
    def model(cls, model_id):
        return os.path.join(cls.meta_regressors_path, model_id)

    @Paths.create_dir
    @classmethod
    def dataset(cls, model_id, dataset_id):
        return os.path.join(cls.meta_regressors_path, model_id, dataset_id)
