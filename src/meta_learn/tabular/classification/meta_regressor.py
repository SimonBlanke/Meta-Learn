# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ...base_meta_regressor import BaseMetaRegressor
from .model_and_dataset_type import ModelAndDatasetType


class MetaRegressor(BaseMetaRegressor, ModelAndDatasetType):
    """Meta-regressor for tabular classification tasks.

    Specializes BaseMetaRegressor for predicting classification
    model performance based on hyperparameters and dataset features.
    """

    def __init__(self, regressor="default", base_path=None):
        super().__init__(regressor, base_path)
