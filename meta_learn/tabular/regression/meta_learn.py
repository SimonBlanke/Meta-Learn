# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .default_reference_datasets import ref_models
from ..dataset_feature_generator import DatasetFeatureGenerator
from ...meta_learn_core import MetaLearnCore


class MetaLearn(MetaLearnCore):
    def __init__(self, path):
        super().__init__(path)
        self.dataset_feature_generator = DatasetFeatureGenerator(ref_models)
