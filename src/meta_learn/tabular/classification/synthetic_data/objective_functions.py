# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


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
