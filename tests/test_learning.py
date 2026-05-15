import numpy as np
import pandas as pd

from ndfu import UnimodalLearner


def _toy_frame():
    return pd.DataFrame(
        {
            "x": [-2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 0.0, 0.2],
            "scores": [
                [1, 1, 1, 2, 2, 3],
                [1, 1, 1, 1, 2, 2],
                [1, 1, 2, 2, 2, 3],
                [3, 4, 4, 5, 5, 5],
                [4, 4, 4, 5, 5, 5],
                [3, 4, 5, 5, 5, 5],
                [1, 1, 1, 5, 5, 5],
                [1, 1, 5, 5, 5, 5],
            ],
        }
    )


def test_unimodal_learner_adds_annotation_columns():
    learner = UnimodalLearner(_toy_frame(), feature_cols=["x"], scale=range(1, 6))

    assert {"HIST", "DFU", "binary_target", "kplus_target"}.issubset(learner.train.columns)
    assert learner.train.loc[6, "DFU"] > 0
    assert learner.train.loc[6, "kplus_target"] == "k+1"
    assert learner.train.loc[0, "binary_target"] == "toxic"
    assert learner.train.loc[3, "binary_target"] == "civil"


def test_unimodal_learner_threshold_changes_kplus_support():
    learner = UnimodalLearner(_toy_frame(), feature_cols=["x"], scale=range(1, 6))

    low_threshold = learner.label_with_threshold(learner.train, 0.0)
    high_threshold = learner.label_with_threshold(learner.train, 1.0)

    assert (low_threshold == "k+1").sum() > 0
    assert (high_threshold == "k+1").sum() == 0


def test_unimodal_learner_fits_models_and_predicts_binary_from_kplus():
    learner = UnimodalLearner(_toy_frame(), feature_cols=["x"], scale=range(1, 6))

    binary_model = learner.fit_binary_baseline()
    clean_model = learner.fit_unimodal_only_baseline()
    kplus_model = learner.fit_kplus_model()

    assert set(binary_model.classes_) == {"civil", "toxic"}
    assert set(clean_model.classes_) == {"civil", "toxic"}
    assert set(kplus_model.classes_) == {"civil", "k+1", "toxic"}

    predictions = learner.binary_predictions_from_kplus(learner.train)

    assert predictions.shape == (len(learner.train),)
    assert set(np.unique(predictions)).issubset({"civil", "toxic"})
