# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import joblib

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer


from meta_learn.tabular.classification.synthetic_data import SyntheticDataGenerator
from meta_learn.tabular.classification.meta_data import MetaData
from meta_learn.tabular.classification import MetaRegressor


dir_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "_tmp_test_data")
)

data = load_breast_cancer()
X_inference, y_inference = data.data, data.target


def dtc_function(opt):
    X = opt.pass_through["X"]
    y = opt.pass_through["y"]

    dtc_model = DecisionTreeClassifier(
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
        "n_informative": 6,
        "class_sep": 0.9,
        "n_classes": 3,
        "flip_y": 0.3,
    },
    "test_dataset_1": {
        "n_samples": 1000,
        "n_features": 12,
        "n_informative": 7,
        "n_redundant": 3,
        "class_sep": 1.0,
        "n_clusters_per_class": 3,
        "flip_y": 0.1,
    },
    "test_dataset_2": {
        "n_samples": 3000,
        "n_features": 18,
        "n_informative": 12,
        "n_redundant": 1,
        "n_repeated": 1,
        "class_sep": 2,
        "flip_y": 0.1,
    },
}


def test_integration():
    model_id = "test_dtc"

    synth_data = SyntheticDataGenerator(base_path=dir_path)
    synth_data.dataset_dict = dataset_dict
    synth_data.collect(dtc_function, search_space, model_id, n_iter=10)

    meta_data = MetaData(base_path=dir_path)
    meta_X, meta_y = meta_data.get_training_data(model_id)
    meta_data_X_test = meta_data.get_inference_data(
        search_space, X_inference, y_inference
    )

    gbr = MetaRegressor(base_path=dir_path)
    gbr.fit(meta_X, meta_y)
    gbr.dump(model_id)

    gbr_load = MetaRegressor(base_path=dir_path)
    gbr_load.load(model_id)
    gbr_load.predict(meta_data_X_test)
    # gbr_load.remove(model_id, always_confirm=True)

    # synth_data.remove(always_confirm=True)
