# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


def svm_ref_model(X, y):
    svr = SVC()
    return cross_val_score(svr, X, y, cv=5).mean()


def dtr_ref_model_short(X, y):
    dtr = DecisionTreeClassifier(max_depth=3)
    return cross_val_score(dtr, X, y, cv=5).mean()


def dtr_ref_model(X, y):
    dtr = DecisionTreeClassifier(max_depth=None)
    return cross_val_score(dtr, X, y, cv=5).mean()


def gbr_ref_model_25(X, y):
    gbr = GradientBoostingClassifier(n_estimators=25, max_depth=3)
    return cross_val_score(gbr, X, y, cv=5).mean()


def gbr_ref_model_100(X, y):
    gbr = GradientBoostingClassifier(n_estimators=100, max_depth=3)
    return cross_val_score(gbr, X, y, cv=5).mean()


ref_models = {
    "svm": svm_ref_model,
    "dtr5": dtr_ref_model_short,
    "dtr": dtr_ref_model,
    "gbr25": gbr_ref_model_25,
    "gbr100": gbr_ref_model_100,
}
