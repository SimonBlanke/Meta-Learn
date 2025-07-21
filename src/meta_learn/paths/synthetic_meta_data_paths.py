# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os

from typing import List

from .paths import Paths


class SyntheticMetaDataPaths(Paths):
    """Path manager for synthetic meta-data storage.

    This class manages the directory structure for storing synthetic
    meta-data including search results and dataset features.

    Attributes:
        base_path (str): Root directory for synthetic meta-data
        search_data (str): Filename for search data
        dataset_features (str): Filename for dataset features
    """

    search_data: str = "search_data.csv"
    dataset_features: str = "dataset_features.json"

    def __init__(self, dataset_type, model_type, base_path=None) -> None:
        super().__init__(dataset_type, model_type, base_path)

        self.base_path = self._synthetic_meta_data_base_path()

    @Paths.create_dir
    def _synthetic_meta_data_base_path(self):
        return os.path.join(self.study_type_path, "synthetic_meta_data")

    @Paths.create_dir
    def dataset(self, model_id, dataset_id):
        """Generate path for specific dataset directory.

        Args:
            model_id (str): Unique identifier for the model
            dataset_id (str): Unique identifier for the dataset

        Returns:
            str: Path to the dataset directory
        """

        return os.path.join(self.model(model_id), dataset_id)

    @Paths.create_dir
    def model(self, model_id):
        """Generate path for specific model directory.

        Args:
            model_id (str): Unique identifier for the model

        Returns:
            str: Path to the model directory
        """

        return os.path.join(self.base_path, model_id)

    def dataset_ids(self, model_id: str) -> List[str]:
        """Retrieve all dataset IDs for a given model.

        Args:
            model_id (str): Unique identifier for the model

        Returns:
            list: Dataset identifiers available for the model
        """

        model_path = os.path.join(
            self.base_path,
            model_id,
        )
        return [name for name in os.listdir(model_path)]

    @Paths.create_dir
    def model_dataset(self, model_id, dataset_id):
        """Generate path for model-dataset combination.

        Args:
            model_id (str): Unique identifier for the model
            dataset_id (str): Unique identifier for the dataset

        Returns:
            str: Path to the model-dataset directory
        """

        return os.path.join(
            self.base_path,
            model_id,
            dataset_id,
        )

    def dataset_features(self, model_id, dataset_id):
        """Generate path for dataset features file.

        Args:
            model_id (str): Unique identifier for the model
            dataset_id (str): Unique identifier for the dataset

        Returns:
            str: Complete path to dataset features JSON file
        """

        return os.path.join(
            self.model_dataset(
                model_id,
                dataset_id,
            ),
            "dataset_features.json",
        )

    def search_data(self, model_id, dataset_id):
        """Generate path for search data file.

        Args:
            model_id (str): Unique identifier for the model
            dataset_id (str): Unique identifier for the dataset

        Returns:
            str: Complete path to search data JSON file
        """

        return os.path.join(
            self.model_dataset(
                model_id,
                dataset_id,
            ),
            "search_data.json",
        )
