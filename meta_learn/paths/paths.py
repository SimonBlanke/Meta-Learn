# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os


class Paths:
    base_path: str = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    )

    package_data: str = "package_data"

    def __init__(self, base_path: str) -> None:
        if base_path:
            self.base_path = base_path
        self.pkg_data: str = os.path.join(self.base_path, self.package_data)

    @staticmethod
    def create_dir(method):
        def wrapper(*args, **kwargs):
            path = method(*args, **kwargs)
            if not os.path.exists(path):
                os.makedirs(path)
            return path

        return wrapper
