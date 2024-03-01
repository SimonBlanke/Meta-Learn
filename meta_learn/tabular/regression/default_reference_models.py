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
    dtr = DecisionTreeRegressor(max_depth=None)
    return cross_val_score(dtr, X, y, cv=5).mean()


def gbr_ref_model(X, y):
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3)
    return cross_val_score(gbr, X, y, cv=5).mean()


ref_models = {
    "reg": reg_ref_model,
    "dtr": dtr_ref_model,
    "gbr": gbr_ref_model,
}
