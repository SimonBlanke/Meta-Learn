import os
import glob
import hashlib
import inspect

from .collector import Collector
from ._meta_regressor import MetaRegressor
from ._recognizer import Recognizer
from ._predictor import Predictor


class HyperactiveWrapper:
    def __init__(self, search_config, meta_learn_path_alt=None):
        self.search_config = search_config

        current_path = os.path.realpath(__file__)
        meta_learn_path, _ = current_path.rsplit("/", 1)

        self.meta_data_path = meta_learn_path + "/meta_data/"
        self.meta_regressor_path = meta_learn_path + "/meta_regressor/"

        if not os.path.exists(self.meta_data_path):
            os.makedirs(self.meta_data_path)

        if not os.path.exists(self.meta_regressor_path):
            os.makedirs(self.meta_regressor_path)

    def get_func_metadata(self, _cand_):
        self.collector = Collector()
        paths = glob.glob(self._get_func_file_paths(_cand_.func_))
        if len(paths) > 0:
            return self.collector._get_func_metadata(paths)
        else:
            return None, None

    def collect(self, X, y, _cand_):
        self.collector = Collector()
        path = self._get_file_path(X, y, _cand_.func_)
        self.collector.extract(X, y, _cand_, path)

    def retrain(self, _cand_):
        path = self._get_metaReg_file_path(_cand_.func_)
        meta_features, target = self.get_func_metadata(_cand_)

        if meta_features is None or target is None:
            return
        self.regressor = MetaRegressor()
        self.regressor.fit(meta_features, target)
        self.regressor.store_model(path)

    def search(self, X, y, _cand_):
        path = self._get_metaReg_file_path(_cand_.func_)

        if not os.path.exists(path):
            return None, None

        self.recognizer = Recognizer(_cand_)
        self.predictor = Predictor()

        self.predictor.load_model(path)

        X_test = self.recognizer.get_test_metadata([X, y])

        return self.predictor.search(X_test)

    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()

    def _get_func_str(self, func):
        return inspect.getsource(func)

    def _get_metaReg_file_path(self, model_func):
        func_str = self._get_func_str(model_func)

        return self.meta_regressor_path + (
            "metamodel__func_hash="
            + self._get_hash(func_str.encode("utf-8"))
            + "__.csv"
        )

    def _get_func_file_paths(self, model_func):
        func_str = self._get_func_str(model_func)
        self.func_path = self._get_hash(func_str.encode("utf-8")) + "/"

        directory = self.meta_data_path + self.func_path
        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory + ("metadata" + "*" + "__.csv")

    def _get_file_path(self, X_train, y_train, model_func):
        func_str = self._get_func_str(model_func)
        feature_hash = self._get_hash(X_train)
        label_hash = self._get_hash(y_train)

        self.func_path = self._get_hash(func_str.encode("utf-8")) + "/"

        directory = self.meta_data_path + self.func_path
        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory + (
            "metadata"
            + "__feature_hash="
            + feature_hash
            + "__label_hash="
            + label_hash
            + "__.csv"
        )
