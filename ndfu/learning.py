"""Learning helpers built around nDFU.

The :class:`UnimodalLearner` class packages the K+1 learning pattern used in
the accompanying notebooks: ordinary labels are used for unimodal examples,
while high-nDFU examples are assigned to an additional class.
"""

from collections.abc import Callable, Sequence
from typing import Optional

import numpy as np

from . import SCALE10, dfu, pdf


class UnimodalLearner:
    """Prepare and train binary, reduced, and K+1 models from annotations.

    Parameters
    ----------
    train, dev, test : pandas.DataFrame
        Data splits. Each split must contain ``scores_col``. Model-fitting
        methods also require either ``feature_cols`` or a ``bert`` column.
    feature_cols : sequence of str, optional
        Numeric feature columns used for model training. If omitted, a ``bert``
        column containing vector embeddings is used for compatibility with the
        original application notebook.
    scores_col : str, default="scores"
        Column containing per-item ordinal annotations.
    scale : sequence, default=SCALE10
        Ordered annotation scale passed to :func:`ndfu.pdf`.
    threshold : float, default=0.0
        Items with ``DFU > threshold`` are assigned to ``kplus_label``.
    ordinary_label_func : callable, optional
        Function mapping a sequence of scores to an ordinary class label. If
        omitted, scores below the middle of ``scale`` vote for
        ``positive_label``; otherwise they vote for ``negative_label``.
    estimator_factory : callable, optional
        Factory returning an sklearn-compatible classifier. The default is
        ``LogisticRegression(max_iter=1000, random_state=random_state)``.
    """

    def __init__(
        self,
        train,
        dev=None,
        test=None,
        *,
        feature_cols: Optional[Sequence[str]] = None,
        scores_col: str = "scores",
        scale: Sequence = SCALE10,
        threshold: float = 0.0,
        ordinary_label_func: Optional[Callable] = None,
        low_score_threshold=None,
        positive_label: str = "toxic",
        negative_label: str = "civil",
        kplus_label: str = "k+1",
        estimator_factory: Optional[Callable] = None,
        random_state: int = 2046,
    ):
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.scores_col = scores_col
        self.scale = list(scale)
        self.threshold = threshold
        self.ordinary_label_func = ordinary_label_func
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.kplus_label = kplus_label
        self.estimator_factory = estimator_factory
        self.random_state = random_state

        if low_score_threshold is None:
            low_score_threshold = self.scale[len(self.scale) // 2]
        self.low_score_threshold = low_score_threshold

        self.train = self.annotate(train.copy())
        self.dev = self.annotate(dev.copy()) if dev is not None else None
        self.test = self.annotate(test.copy()) if test is not None else None

    def ordinary_label(self, scores):
        """Return the ordinary, non-K+1 label for an annotation list."""

        if self.ordinary_label_func is not None:
            return self.ordinary_label_func(scores)

        positive_share = np.mean([score < self.low_score_threshold for score in scores])
        return self.positive_label if positive_share >= 0.5 else self.negative_label

    def make_kplus_label(self, score, scores, threshold=None):
        """Return ``kplus_label`` for high-nDFU items, otherwise ordinary label."""

        if threshold is None:
            threshold = self.threshold
        if score > threshold:
            return self.kplus_label
        return self.ordinary_label(scores)

    def annotate(self, frame, threshold=None):
        """Add ``HIST``, ``DFU``, ``binary_target``, and ``kplus_target`` columns."""

        if self.scores_col not in frame.columns:
            raise ValueError(f"missing required scores column: {self.scores_col}")

        frame["HIST"] = frame[self.scores_col].apply(lambda scores: pdf(scores, self.scale))
        frame["DFU"] = frame.HIST.apply(dfu)
        frame["binary_target"] = frame[self.scores_col].apply(self.ordinary_label)
        frame["kplus_target"] = frame.apply(
            lambda row: self.make_kplus_label(row.DFU, row[self.scores_col], threshold),
            axis=1,
        )
        return frame

    def label_with_threshold(self, frame, threshold):
        """Return K+1 labels for ``frame`` using a custom nDFU threshold."""

        return frame.apply(
            lambda row: self.make_kplus_label(row.DFU, row[self.scores_col], threshold),
            axis=1,
        )

    def features(self, frame):
        """Return model features from explicit columns or a BERT-vector column."""

        if self.feature_cols is not None:
            return frame[self.feature_cols].to_numpy()
        if "bert" not in frame.columns:
            raise ValueError("feature_cols must be provided unless frame contains a 'bert' column")
        return np.concatenate(frame.bert.to_numpy()).reshape(frame.shape[0], -1)

    def _new_estimator(self):
        if self.estimator_factory is not None:
            return self.estimator_factory()

        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise ImportError(
                "Model fitting requires scikit-learn. Install scikit-learn or pass "
                "an estimator_factory."
            ) from exc

        return LogisticRegression(max_iter=1000, random_state=self.random_state)

    def fit_binary_baseline(self):
        """Train a binary model on all training examples."""

        self.binary_model = self._new_estimator()
        self.binary_model.fit(self.features(self.train), self.train.binary_target)
        return self.binary_model

    def fit_unimodal_only_baseline(self):
        """Train a binary model after removing high-nDFU training examples."""

        clean_train = self.train[self.train.DFU == 0]
        if clean_train.empty:
            raise ValueError("no zero-nDFU training examples available")
        self.clean_train_size_ = len(clean_train)
        self.removed_train_size_ = len(self.train) - len(clean_train)
        self.clean_binary_model = self._new_estimator()
        self.clean_binary_model.fit(self.features(clean_train), clean_train.binary_target)
        return self.clean_binary_model

    def fit_kplus_model(self):
        """Train a model on ordinary labels plus the K+1 class."""

        self.kplus_model = self._new_estimator()
        self.kplus_model.fit(self.features(self.train), self.train.kplus_target)
        return self.kplus_model

    def binary_predictions_from_kplus(self, frame, ordinary_labels=None):
        """Predict ordinary labels from a K+1 model by ignoring K+1 probability."""

        if ordinary_labels is None:
            ordinary_labels = [self.negative_label, self.positive_label]

        probabilities = self.kplus_model.predict_proba(self.features(frame))
        classes = list(self.kplus_model.classes_)
        ordinary_indices = [classes.index(label) for label in ordinary_labels]
        ordinary_probabilities = probabilities[:, ordinary_indices]
        ordinary_labels = np.asarray(ordinary_labels)
        return ordinary_labels[np.argmax(ordinary_probabilities, axis=1)]
