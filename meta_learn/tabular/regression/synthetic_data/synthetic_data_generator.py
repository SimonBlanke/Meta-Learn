# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import make_regression

from ....synthetic_data_generator import BaseSyntheticDataGenerator
from ..meta_learn import MetaLearn
from .synthetic_data_parameters import dataset_dict


class SyntheticDataGenerator(BaseSyntheticDataGenerator):
    def __init__(self, base_path=".") -> None:
        super().__init__(base_path)

        self.meta_learn = MetaLearn(base_path)
        self.dataset_dict = dataset_dict

    def generate(self, dataset_para):
        return make_regression(**dataset_para)
