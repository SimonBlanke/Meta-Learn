# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import make_classification

from ....synthetic_data_generator import BaseSyntheticDataGenerator
from ..meta_data import MetaData
from .synthetic_data_parameters import dataset_dict
from ..model_and_dataset_type import ModelAndDatasetType


class SyntheticDataGenerator(BaseSyntheticDataGenerator, ModelAndDatasetType):
    def __init__(self, base_path=None) -> None:
        super().__init__(base_path)

        self.meta_data = MetaData(self.base_path)
        self.dataset_dict = dataset_dict

    def generate(self, dataset_para):
        return make_classification(**dataset_para)
