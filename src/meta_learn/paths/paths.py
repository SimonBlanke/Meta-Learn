# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os


class Paths:
    """Base class for path management in the meta-learning framework.

    This class establishes the directory structure for storing meta-learning
    artifacts and provides utilities for creating directories on demand.

    Attributes:
        base_path (str): Root directory for all meta-learning data
        package_data (str): Subdirectory name for package data storage
        study_type_path (str): Path specific to dataset/model type combination
    """

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
        """Decorator that ensures directory creation for path-returning methods.

        This decorator wraps methods that return file paths, automatically
        creating the directory structure if it doesn't exist.

        Args:
            method: Method that returns a file or directory path

        Returns:
            function: Wrapped method that creates directories as needed
        """

        def wrapper(*args, **kwargs):
            path = method(*args, **kwargs)
            if not os.path.exists(path):
                os.makedirs(path)
            return path

        return wrapper

    @create_dir
    def _study_type_path(self, pkg_data_path, dataset_type, model_type):
        return os.path.join(pkg_data_path, dataset_type, model_type)
