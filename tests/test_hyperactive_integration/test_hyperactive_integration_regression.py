# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import joblib

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer


from meta_learn.tabular.regression import SyntheticDataGenerator
from meta_learn.tabular.regression.meta_learn import MetaLearn
from meta_learn.tabular.regression import MetaRegressor


dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

data = load_breast_cancer()
X_inf, y_inf = data.data, data.target


def dtc_function(opt):
    X = opt.pass_through["X"]
    y = opt.pass_through["y"]

    dtc_model = DecisionTreeRegressor(
        max_depth=opt["max_depth"],
        min_samples_split=opt["min_samples_split"],
        min_samples_leaf=opt["min_samples_leaf"],
    )
    scores = cross_val_score(dtc_model, X, y, cv=3)
    return scores.mean()


search_space = {
    "max_depth": list(range(2, 100)),
    "min_samples_split": list(range(2, 100)),
    "min_samples_leaf": list(range(1, 100)),
}
dataset_dict = {
    "test_dataset_0": {
        "n_samples": 300,
        "n_features": 10,
        "n_informative": 4,
        "class_sep": 0.9,
        "n_classes": 3,
        "flip_y": 0.3,
    },
    "test_dataset_1": {
        "n_samples": 500,
        "n_features": 12,
        "n_informative": 8,
        "n_redundant": 3,
        "class_sep": 1.0,
        "n_clusters_per_class": 3,
        "flip_y": 0.1,
    },
    "test_dataset_2": {
        "n_samples": 1000,
        "n_features": 14,
        "n_informative": 5,
        "n_redundant": 4,
        "n_repeated": 2,
        "class_sep": 0.5,
        "n_classes": 4,
        "flip_y": 0.1,
    },
}


def test_hyperactive_integration_classification():
    model_id = "test_dtc"

    synth_data = SyntheticDataGenerator(dir_path)
    synth_data.dataset_dict = dataset_dict
    synth_data.collect(dtc_function, search_space, model_id, n_iter=10)

    meta_learn = MetaLearn(dir_path)
    meta_X, meta_y = meta_learn.get_meta_data(model_id)
    meta_data_X_test = meta_learn.get_meta_data_X(search_space, X_inf, y_inf)

    gbr = MetaRegressor()
    gbr.fit(meta_X, meta_y)

    joblib.dump(gbr, "meta_regressor.joblib")
    gbr_ = joblib.load("meta_regressor.joblib")
    os.remove("meta_regressor.joblib")

    gbr_.predict(meta_data_X_test)

    meta_learn.remove()
