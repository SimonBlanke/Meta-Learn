# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from meta_learn.tabular.classification import MetaRegressor


def test_meta_regressor():
    meta_reg = MetaRegressor()
    print(meta_reg.get_objective_function_names())
