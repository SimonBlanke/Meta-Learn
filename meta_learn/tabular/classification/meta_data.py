# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .default_reference_models import ref_models
from ..dataset_feature_generator import DatasetFeatureGenerator
from ...base_meta_data import BaseMetaData


class MetaData(BaseMetaData):
    def __init__(self, path):
        super().__init__(path)
        self.dataset_feature_generator = DatasetFeatureGenerator(ref_models)
