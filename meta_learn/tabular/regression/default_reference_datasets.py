# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


def reg_ref_model(X, y):
    reg = LinearRegression()
    reg.fit(X, y)
    return reg.score(X, y)


def dtr_ref_model(X, y):
    dtr = DecisionTreeRegressor()
    return cross_val_score(dtr, X, y, cv=5).mean()


def gbr_ref_model(X, y):
    gbr = GradientBoostingRegressor()
    return cross_val_score(gbr, X, y, cv=5).mean()


ref_models = {
    "reg": reg_ref_model,
    "dtr": dtr_ref_model,
    "gbr": gbr_ref_model,
}


def dataset_feature_generator(X, y):
    ref_scores = {}

    print("Generate dataset features...", end="\r")
    for model_name in ref_models.keys():
        ref_scores[model_name] = ref_models[model_name](X, y)

    ref_scores["n_samples"] = X.shape[0]
    ref_scores["n_features"] = X.shape[1]

    print("Dataset features generated    ")

    # print("\n ref_scores \n", ref_scores, "\n")

    return ref_scores
