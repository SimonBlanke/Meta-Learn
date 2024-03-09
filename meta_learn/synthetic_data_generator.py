# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pandas as pd
from typing import List


from hyperactive import Hyperactive

from .paths import SyntheticMetaDataPaths


class BaseSyntheticDataGenerator:
    def __init__(self) -> None:
        pass

    def collect(self, objective_function, search_space, model_id, n_iter, n_jobs=1):
        for dataset_id, dataset_para in self.dataset_dict.items():
            X, y = self.generate(dataset_para)

            objective_function_dec = self.meta_data.collect(X, y, model_id, dataset_id)(
                objective_function
            )

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