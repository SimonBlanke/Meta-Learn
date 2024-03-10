# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os


class Paths:
    base_path: str = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    )

    package_data: str = "package_data"

    def __init__(self, dataset_type, model_type, base_path) -> None:
        if base_path:
            self.base_path = base_path
        pkg_data: str = os.path.join(self.base_path, self.package_data)
        self.study_type_path: str = self._study_type_path(
            pkg_data, dataset_type, model_type
        )

    @staticmethod
    def create_dir(method):
        def wrapper(*args, **kwargs):
            path = method(*args, **kwargs)
            if not os.path.exists(path):
                os.makedirs(path)
            return path

        return wrapper

    @create_dir
    def _study_type_path(self, pkg_data_path, dataset_type, model_type):
        return os.path.join(pkg_data_path, dataset_type, model_type)
