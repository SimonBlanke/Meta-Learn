# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .default_reference_models import ref_models
from ..dataset_feature_generator import DatasetFeatureGenerator
from ...base_meta_data import BaseMetaData
from .model_and_dataset_type import ModelAndDatasetType


class MetaData(BaseMetaData, ModelAndDatasetType):
    def __init__(self, base_path):
        super().__init__(base_path)
        self.dataset_feature_generator = DatasetFeatureGenerator(ref_models)
