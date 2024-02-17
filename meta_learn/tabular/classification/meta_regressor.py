# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ...base_meta_regressor import BaseMetaRegressor


class MetaRegressor(BaseMetaRegressor):
    dataset_type = "tabular"
    model_type = "classification"

    def __init__(self, regressor="default", encoder="default"):
        super().__init__(regressor, encoder)
