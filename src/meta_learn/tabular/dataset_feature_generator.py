# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class DatasetFeatureGenerator:
    """Generator for extracting meta-features from datasets.

    This class computes dataset characteristics that serve as features
    for meta-models. It uses reference models to capture dataset complexity
    and adds basic statistical properties.

    Attributes:
        ref_models (dict): Dictionary of reference model functions
    """

    def __init__(self, ref_models) -> None:
        self.ref_models = ref_models

    def create(self, X, y):
        """Extract meta-features from a dataset.

        Computes various dataset characteristics including performance scores
        from reference models and basic dataset statistics. These features
        help meta-models understand dataset properties.

        Args:
            X: Feature matrix of the dataset
            y: Target values of the dataset

        Returns:
            dict: Dictionary containing computed meta-features including:
                - Reference model scores (one per configured model)
                - n_samples: Number of samples in the dataset
                - n_features: Number of features in the dataset
        """

        ref_scores = {}

        print("Generate dataset features...", end="\r")
        for model_name in self.ref_models.keys():
            ref_scores[model_name] = self.ref_models[model_name](X, y)

        ref_scores["n_samples"] = X.shape[0]
        ref_scores["n_features"] = X.shape[1]

        print("Dataset features generated    ")

        # print("\n ref_scores \n", ref_scores, "\n")

        return ref_scores
