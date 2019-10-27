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

        # print("\nmeta_learn_path", meta_learn_path)

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

    def train(self):
        self.regressor = MetaRegressor(self.meta_learn_path)
        self.regressor.train_meta_regressor(self.model_list)

    def search(self, X, y):
        self.recognizer = Recognizer(self.search_config)
        self.predictor = Predictor(self.search_config, self.meta_regressor_path)

        X_test = self.recognizer.get_test_metadata([X, y])

        self.best_hyperpara_dict, self.best_score = self.predictor.search(X_test)
