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
        base_path: str = None,
    ) -> None:
        self.synth_data_path = SyntheticMetaDataPaths(
            dataset_type, model_type, base_path
        )

    def load(self, model_id, dataset_id) -> pd.core.frame.DataFrame:
        path2csv = self.synth_data_path.search_data(model_id, dataset_id)
        return SearchDataCollector(path2csv).load()

    def append(self, search_data: dict, model_id, dataset_id):
        path2csv = self.synth_data_path.search_data(model_id, dataset_id)
        SearchDataCollector(path2csv).append(search_data)
