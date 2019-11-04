import os
import glob

from .collector import Collector
from ._meta_regressor import MetaRegressor
from ._recognizer import Recognizer
from ._predictor import Predictor


class HyperactiveWrapper:
    def __init__(self, search_config):
        self.search_config = search_config

        current_path = os.path.realpath(__file__)
        meta_learn_path, _ = current_path.rsplit("/", 1)

        self.meta_data_path = meta_learn_path + "/meta_data/"
        self.meta_regressor_path = meta_learn_path + "/meta_regressor/"

    def get_func_metadata(self, model_func):
        self.collector = Collector(
            self.search_config, meta_data_path=self.meta_data_path
        )
        paths = glob.glob(self.collector._get_func_file_paths(model_func))

        return self.collector._get_func_metadata(paths)

    def collect(self, X, y, _cand_list):
        self.collector = Collector(
            self.search_config, meta_data_path=self.meta_data_path
        )
        self.collector.extract(X, y, _cand_list)

    def retrain(self, model_func):
        meta_features, target = self.get_func_metadata(model_func)
        self.regressor = MetaRegressor()
        self.regressor.fit(meta_features, target)
        self.regressor.store_model(model_func)

    def search(self, X, y, model_func):
        self.recognizer = Recognizer(self.search_config)
        self.predictor = Predictor(self.search_config, self.meta_regressor_path)

        self.predictor.load_model(model_func)

        X_test = self.recognizer.get_test_metadata([X, y])

        return self.predictor.search(X_test, model_func)
