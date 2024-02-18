# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os


class Paths:
    base_path_name = "data_meta_learn"

    def __init__(self, path):
        self.base_path_name = os.path.join(path, self.base_path_name)

    def get_model_dir(self, model_id):
        return os.path.join(self.base_path_name, model_id)

    def get_dataset_dir(self, model_id, dataset_id):
        return os.path.join(self.base_path_name, model_id, dataset_id)

    def create_meta_learn_dir(self):
        if not os.path.exists(self.base_path_name):
            os.makedirs(self.base_path_name)

    def create_model_dir(self, model_id):
        path = self.get_model_dir(model_id)
        if not os.path.exists(path):
            os.makedirs(path)

    def create_dataset_dir(self, model_id, dataset_id):
        path = self.get_dataset_dir(model_id, dataset_id)
        if not os.path.exists(path):
            os.makedirs(path)
