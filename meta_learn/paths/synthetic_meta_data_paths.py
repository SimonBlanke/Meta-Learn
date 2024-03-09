# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os


from .paths import Paths


class SyntheticMetaDataPaths(Paths):
    search_data: str = "search_data.csv"
    dataset_features: str = "dataset_features.json"

    synthetic_meta_data_path: str = os.path.join(Paths.pkg_data, "synthetic_meta_data")
