# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import make_classification

from ...synthetic_data import BaseSyntheticDataGenerator


class SyntheticDataGenerator(BaseSyntheticDataGenerator):
    def __init__(self, base_path=".") -> None:
        super().__init__(base_path)

    def generate(self, dataset_para):
        return make_classification(**dataset_para)
