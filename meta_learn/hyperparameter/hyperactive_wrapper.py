import os


from .collector import Collector
from .meta_regressor import MetaRegressor
from .recognizer import Recognizer
from .predictor import Predictor


class HyperactiveWrapper:
    def __init__(self):
        current_path = os.path.realpath(__file__)
        meta_learn_path, _ = current_path.rsplit("/", 1)

        self.meta_data_path = meta_learn_path + "/meta_data/"
        self.meta_regressor_path = meta_learn_path + "/meta_regressor/"

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
