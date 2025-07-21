# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pandas as pd

from search_data_collector import CsvSearchData


from .paths import SyntheticMetaDataPaths


class SearchData:
    """Manager for search data persistence in meta-learning.

    This class handles the loading and appending of search data generated
    during hyperparameter optimization experiments. Search data includes
    hyperparameter configurations and their corresponding performance scores.

    Attributes:
        synth_data_path (SyntheticMetaDataPaths): Path manager for data storage
    """

    def __init__(
        self,
        dataset_type: str,
        model_type: str,
        base_path: str = None,
    ) -> None:
        self.synth_data_path = SyntheticMetaDataPaths(
            dataset_type, model_type, base_path
        )

    def load(self, model_id, dataset_id) -> pd.core.frame.DataFrame:
        """Load search data from CSV file.

        Args:
            model_id (str): Unique identifier for the model type
            dataset_id (str): Unique identifier for the dataset

        Returns:
            pd.DataFrame: DataFrame containing search history
        """

        path2csv = self.synth_data_path.search_data(model_id, dataset_id)
        return CsvSearchData(path2csv).load()

    def append(self, search_data: dict, model_id, dataset_id):
        """Append new search result to existing data.

        Args:
            search_data (dict): Dictionary containing hyperparameters and score
            model_id (str): Unique identifier for the model type
            dataset_id (str): Unique identifier for the dataset
        """

        path2csv = self.synth_data_path.search_data(model_id, dataset_id)
        CsvSearchData(path2csv).append(search_data)
