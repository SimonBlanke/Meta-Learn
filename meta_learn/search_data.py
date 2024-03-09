# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pandas as pd

from search_data_collector import SearchDataCollector


from .paths import SyntheticMetaDataPaths


class SearchData:
    def __init__(
        self,
        dataset_type: str,
        model_type: str,
        model_id: str,
        dataset_id: str,
        base_path: str = None,
    ) -> None:
        search_data_path = SyntheticMetaDataPaths(base_path).search_data(
            dataset_type, model_type, model_id, dataset_id
        )
        self.collector = SearchDataCollector(search_data_path)

    def load(self) -> pd.core.frame.DataFrame:
        return self.collector.load()

    def append(self, search_data: dict):
        self.collector.append(search_data)
