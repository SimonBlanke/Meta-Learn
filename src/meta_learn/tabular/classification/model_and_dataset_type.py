# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class ModelAndDatasetType:
    """Mixin class defining dataset and model type constants.

    This class serves as a mixin to provide consistent dataset_type
    and model_type attributes across related classes in the framework.

    Attributes:
        dataset_type (str): Type of dataset (e.g., "tabular")
        model_type (str): Type of model task (e.g., "classification", "regression")
    """

    dataset_type = "tabular"
    model_type = "classification"
