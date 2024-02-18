# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import make_classification, make_regression

from hyperactive import Hyperactive

from .synthetic_data_parameters import dataset_dict
from ..tabular.regression.meta_learn import MetaLearn


class SyntheticDataGenerator:
    def __init__(self, base_path=".") -> None:
        self.meta_learn = MetaLearn(base_path)
        self.dataset_dict = dataset_dict

    def generate(self, dataset_para):
        return make_classification(**dataset_para)

    def collect(self, objective_function, search_space, model_id, n_iter, n_jobs=1):
        for dataset_id in self.dataset_dict.keys():
            dataset_para = dataset_dict[dataset_id]

            X, y = self.generate(dataset_para)

            objective_function_dec = self.meta_learn.collect(
                X, y, model_id, dataset_id
            )(objective_function)

            pass_through = {
                "X": X,
                "y": y,
            }

            hyper = Hyperactive(verbosity=["progress_bar"])
            hyper.add_search(
                objective_function_dec,
                search_space,
                n_iter=n_iter,
                n_jobs=n_jobs,
                pass_through=pass_through,
            )
            hyper.run()
