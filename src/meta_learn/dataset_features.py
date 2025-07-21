# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import json
import pandas as pd


from .paths import SyntheticMetaDataPaths


class DatasetFeatures:
    """Manager for dataset feature persistence in meta-learning.

    This class handles the loading and saving of dataset features (characteristics
    extracted from datasets) that are used as inputs to meta-models. Features
    are stored as JSON files for easy inspection and manipulation.

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
        """Load dataset features from JSON file.

        Args:
            model_id (str): Unique identifier for the model type
            dataset_id (str): Unique identifier for the dataset

        Returns:
            dict: Dictionary containing dataset features
        """

        path2file = self.synth_data_path.dataset_features(model_id, dataset_id)
        with open(path2file, "r") as f:
            return json.load(f)

    def dump(self, dataset_features: dict, model_id, dataset_id):
        """Save dataset features to JSON file.

        Args:
            dataset_features (dict): Dictionary of feature names to values
            model_id (str): Unique identifier for the model type
            dataset_id (str): Unique identifier for the dataset
        """

        path2file = self.synth_data_path.dataset_features(model_id, dataset_id)
        with open(path2file, "w") as f:
            json.dump(dataset_features, f)
