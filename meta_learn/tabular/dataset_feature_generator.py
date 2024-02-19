# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class DatasetFeatureGenerator:
    def __init__(self, ref_models) -> None:
        self.ref_models = ref_models

    def create(self, X, y):
        ref_scores = {}

        print("Generate dataset features...", end="\r")
        for model_name in self.ref_models.keys():
            ref_scores[model_name] = self.ref_models[model_name](X, y)

        ref_scores["n_samples"] = X.shape[0]
        ref_scores["n_features"] = X.shape[1]

        print("Dataset features generated    ")

        # print("\n ref_scores \n", ref_scores, "\n")

        return ref_scores
