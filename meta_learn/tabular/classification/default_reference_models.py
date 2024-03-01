# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


def reg_ref_model(X, y):
    reg = LogisticRegression()
    reg.fit(X, y)
    return reg.score(X, y)


def dtr_ref_model(X, y):
    dtr = DecisionTreeClassifier(max_depth=None)
    return cross_val_score(dtr, X, y, cv=5).mean()


def gbr_ref_model(X, y):
    gbr = GradientBoostingClassifier(n_estimators=100, max_depth=3)
    return cross_val_score(gbr, X, y, cv=5).mean()


ref_models = {
    "reg": reg_ref_model,
    "dtr": dtr_ref_model,
    "gbr": gbr_ref_model,
}
