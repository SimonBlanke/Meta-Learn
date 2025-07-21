# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import shutil

from hyperactive import Hyperactive

from .utils import query_yes_no
from .paths import SyntheticMetaDataPaths


class BaseSyntheticDataGenerator:
    """Base class for generating synthetic training data for meta-learning.

    This class provides functionality to generate synthetic datasets and
    collect performance data across different hyperparameter configurations,
    building a comprehensive meta-dataset for training meta-models.

    Attributes:
        base_path (str): Root directory for data storage
        synth_meta_data_paths (SyntheticMetaDataPaths): Path manager
    """

    def __init__(self, base_path) -> None:
        self.base_path = base_path
        self.synth_meta_data_paths = SyntheticMetaDataPaths(
            self.dataset_type,
            self.model_type,
            base_path,
        )

    def remove(self, model_id=None, dataset_id=None, always_confirm=False):
        """Remove synthetic meta-data with optional confirmation.

        Allows selective removal of data at different granularity levels:
        specific dataset, all datasets for a model, or all data.

        Args:
            model_id (str, optional): Model identifier to remove
            dataset_id (str, optional): Dataset identifier to remove
            always_confirm (bool): If True, skip confirmation prompt
        """

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
        """Collect meta-data by running hyperparameter optimization on synthetic datasets.

        Iterates through predefined synthetic datasets, running hyperparameter
        optimization for each and collecting the results as meta-training data.

        Args:
            objective_function: Function to optimize (takes optimizer access object)
            search_space (dict): Hyperparameter search space definition
            model_id (str): Unique identifier for the model type
            n_iter (int): Number of optimization iterations per dataset
            n_jobs (int): Number of parallel jobs for optimization
        """

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
