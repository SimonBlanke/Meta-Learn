# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import importlib


class MetaRegressorImporter:
    def __init__(self, dataset_type, model_type) -> None:
        path2module = "meta_learn." + dataset_type + "." + model_type
        self.meta_regressor = getattr(
            importlib.import_module(path2module), "MetaRegressor"
        )
