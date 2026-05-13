"""Normalized Distance from Unimodality (nDFU).

The public API intentionally stays small: use ``pdf`` for ordinal rating
frequencies, ``cpdf`` for cumulative frequencies, ``to_hist`` for binned
numeric data, and ``dfu`` for the distance from unimodality.
"""

from collections import Counter

import numpy as np

SCALE10 = list(range(1, 11))


def _as_non_empty_1d_array(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_histogram(hist):
    hist = _as_non_empty_1d_array(hist, "histogram")
    if np.any(hist < 0):
        raise ValueError("histogram values must be non-negative")
    if hist.max() == 0:
        raise ValueError("histogram must contain at least one positive value")
    return hist


def dfu(input_data, histogram_input=True, normalised=True):
    """Return the Distance from Unimodality score.

    Parameters
    ----------
    input_data : sequence of numbers
        A histogram/relative-frequency vector by default. If
        ``histogram_input`` is false, raw scores are first converted to a
        normalized histogram with :func:`to_hist`.
    histogram_input : bool, default=True
        Whether ``input_data`` is already a histogram.
    normalised : bool, default=True
        Whether to divide the raw DFU score by the maximum histogram value.

    Returns
    -------
    float
        The raw or normalized distance from unimodality.
    """

    hist = input_data if histogram_input else to_hist(input_data)
    hist = _validate_histogram(hist)

    max_value = hist.max()
    pos_max = int(np.flatnonzero(hist == max_value)[0])

    max_diff = 0.0
    for i in range(pos_max, len(hist) - 1):
        max_diff = max(max_diff, hist[i + 1] - hist[i])
    for i in range(pos_max, 0, -1):
        max_diff = max(max_diff, hist[i - 1] - hist[i])

    if normalised:
        return float(max_diff / max_value)
    return float(max_diff)


def to_hist(scores, bins_num=3, normed=True):
    """Create a histogram from numeric scores.

    Parameters
    ----------
    scores : sequence of numbers
        Raw scores to bin.
    bins_num : int, default=3
        Number of bins to create.
    normed : bool, default=True
        Whether to return relative frequencies instead of counts.
    """

    scores = _as_non_empty_1d_array(scores, "scores")
    counts, _ = np.histogram(a=scores, bins=bins_num)
    if not normed:
        return counts
    total = counts.sum()
    if total == 0:
        raise ValueError("histogram cannot be normalized because it is empty")
    return counts / total


def pdf(scores, scale=SCALE10):
    """Return relative frequencies of ordinal ratings over ``scale``."""

    scores = list(scores)
    scale = list(scale)
    if not scores:
        raise ValueError("scores must not be empty")
    if not scale:
        raise ValueError("scale must not be empty")

    freqs = Counter(scores)
    return np.array([freqs[s] / len(scores) for s in scale], dtype=float)


def cpdf(scores, scale=SCALE10):
    """Return cumulative relative frequencies of ordinal ratings."""

    return np.cumsum(pdf(scores, scale))


__all__ = ["SCALE10", "cpdf", "dfu", "pdf", "to_hist"]
