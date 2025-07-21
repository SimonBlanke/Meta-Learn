# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import shutil

from hyperactive import Hyperactive

from .utils import query_yes_no
from .paths import SyntheticMetaDataPaths


class BaseSyntheticDataGenerator:
    def __init__(self, base_path) -> None:
        self.base_path = base_path
        self.synth_meta_data_paths = SyntheticMetaDataPaths(
            self.dataset_type,
            self.model_type,
            base_path,
        )

    def remove(self, model_id=None, dataset_id=None, always_confirm=False):
        if always_confirm:
            self._remove_confirmed(model_id, dataset_id)
        else:
            question = "Remove synthetic meta data?"
            if query_yes_no(question):
                self._remove_confirmed(model_id, dataset_id)

    def _remove_confirmed(self, model_id, dataset_id):
        if model_id and dataset_id:
            shutil.rmtree(self.synth_meta_data_paths.dataset(model_id, dataset_id))
        elif model_id:
            shutil.rmtree(self.synth_meta_data_paths.model(model_id))
        elif not model_id and not dataset_id:
            shutil.rmtree(self.synth_meta_data_paths.base_path)
        else:
            raise ValueError

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
