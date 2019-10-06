# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import pandas as pd

from importlib import import_module
from sklearn.model_selection import cross_val_score


class DatasetFeatures:
    def __init__(self, cv=2, n_jobs=-1):
        pass

    def collect(self, model_name, data_train):
        self.X_train = data_train[0]
        self.y_train = data_train[1]

        # List of functions to get the different features of the dataset
        func_list = [self.get_number_of_instances, self.get_number_of_features]

        features_from_dataset = {}
        for func in func_list:
            name, value = func()
            features_from_dataset[name] = value

        features_from_dataset = pd.DataFrame(features_from_dataset, index=[0])
        features_from_dataset = features_from_dataset.reindex(
            sorted(features_from_dataset.columns), axis=1
        )

        return features_from_dataset

    #################################################################################
    # Dataset feature-functions

    def get_number_of_instances(self):
        return "N_rows", int(self.X_train.shape[0])

    def get_number_of_features(self):
        return "N_columns", int(self.X_train.shape[1])
