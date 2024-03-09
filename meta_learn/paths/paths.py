# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os


class Paths:
    dir_path: str = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    package_data: str = "package_data"

    pkg_data: str = os.path.join(dir_path, package_data)

    @staticmethod
    def create_dir(method):
        def wrapper(*args, **kwargs):
            path = method(*args, **kwargs)
            if not os.path.exists(path):
                os.makedirs(path)
            return path

        return wrapper
