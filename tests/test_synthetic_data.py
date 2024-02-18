# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from meta_learn.synthetic_data import SyntheticDataGenerator


def test_synthetic_data():
    model_id = "test_dtc"

    synth_data = SyntheticDataGenerator()
    synth_data.collect(model_id, n_iter=10)
