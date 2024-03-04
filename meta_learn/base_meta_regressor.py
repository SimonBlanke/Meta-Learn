# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os, sys
from joblib import load, dump
from sklearn.ensemble import GradientBoostingRegressor


def query_yes_no(question, default="no"):
    # ref: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


class BaseMetaRegressor:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(dir_path, "pretrained_meta_regressors")

    def __init__(self, regressor):
        self.regressor = regressor

        if regressor == "default":
            self.m_reg = GradientBoostingRegressor()
        else:
            self.m_reg = regressor

    def generate_path(self, model):
        path_dir = os.path.abspath(
            os.path.join(self.base_path, self.dataset_type, self.model_type)
        )
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        path_file = os.path.join(path_dir, model)
        return path_file

    def dump(self, objective_function):
        path = self.generate_path(objective_function.__name__)
        dump(self.m_reg, path)

    def load(self, objective_function):
        path = self.generate_path(objective_function.__name__)
        self.m_reg = load(path)

    def remove_confirmed(self, objective_function):
        path = self.generate_path(objective_function.__name__)
        os.remove(path)

    def remove(self, objective_function, always_confirm=False):
        if always_confirm:
            self.remove_confirmed(objective_function)
        else:
            question = "Remove pretrained meta regressor?"
            if query_yes_no(question):
                self.remove_confirmed(objective_function)

    def fit(self, X_meta, y_meta, drop_duplicates=True):
        if drop_duplicates:
            X_meta = X_meta.drop_duplicates()
            y_meta = y_meta.iloc[X_meta.index]

        self.m_reg.fit(X_meta, y_meta)

    def predict(self, X):
        return self.m_reg.predict(X)
