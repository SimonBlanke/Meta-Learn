# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import make_sparse_uncorrelated, make_friedman1
from sklearn.datasets import load_diabetes, fetch_california_housing


def ref_dataset_1(model, para):
    data = load_diabetes()
    X, y = data.data, data.target
    return model(para)


def ref_dataset_2(model, para):
    X, y = make_sparse_uncorrelated(n_samples=500, n_features=12)
    return model(para)


def ref_dataset_3(model, para):
    data = fetch_california_housing()
    X, y = data.data, data.target
    return model(para)


ref_datasets = {
    "dataset1": ref_dataset_1,
    "dataset2": ref_dataset_2,
    "dataset3": ref_dataset_3,
}


def model_feature_generator(model, para):
    ref_scores = {}

    for model_name in ref_datasets.keys():
        ref_scores[model_name] = ref_datasets[model_name](model, para)
    print("ref_scores", ref_scores)

    return ref_scores
