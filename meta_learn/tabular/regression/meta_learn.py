# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ...meta_learn_core import MetaLearnCore
from .default_reference_datasets import dataset_feature_generator


class MetaLearn(MetaLearnCore):
    def __init__(self, path):
        super().__init__(path)

        self.dataset_feature_generator = dataset_feature_generator
