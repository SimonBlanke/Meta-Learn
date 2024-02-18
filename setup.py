import os
import re

from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


def find_version(*filepath):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, *filepath)) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


setup(
    name="meta-learn",
    version=find_version("meta_learn/__init__.py"),
    author="Simon Blanke",
    author_email="simon.blanke@yahoo.com",
    license="MIT",
    description="Collection and utilization of metadata from machine learning models and problems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["machine learning", "deep learning", "optimization", "data-science"],
    url="https://github.com/SimonBlanke/Meta-Learn",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
    ],
    install_requires=open("./requirements.txt").readlines(),
    python_requires=">=3.5",
    include_package_data=True,
)
