import numpy as np
import pytest

from ndfu import cpdf, dfu, pdf, to_hist


def test_dfu_accepts_lists_and_arrays():
    values = [0.2, 0.6, 0.2]

    assert dfu(values) == pytest.approx(0.0)
    assert dfu(np.array(values)) == pytest.approx(0.0)


def test_dfu_detects_non_unimodal_histogram():
    assert dfu([0.5, 0.0, 0.5]) == pytest.approx(1.0)
    assert dfu([0.5, 0.0, 0.5], normalised=False) == pytest.approx(0.5)


def test_dfu_can_bin_raw_scores():
    assert dfu([1, 1, 5, 5], histogram_input=False, normalised=False) > 0


def test_pdf_and_cpdf():
    scores = [1, 1, 2, 5, 5, 5]

    np.testing.assert_allclose(pdf(scores, range(1, 6)), [2 / 6, 1 / 6, 0, 0, 3 / 6])
    np.testing.assert_allclose(cpdf(scores, range(1, 6)), [2 / 6, 3 / 6, 3 / 6, 3 / 6, 1])


def test_to_hist_counts_and_relative_frequencies():
    scores = [1, 1, 2, 5, 5, 5]

    counts = to_hist(scores, bins_num=3, normed=False)
    assert counts.sum() == len(scores)
    np.testing.assert_allclose(to_hist(scores, bins_num=3).sum(), 1.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: dfu([]),
        lambda: dfu([0, 0, 0]),
        lambda: dfu([-1, 2, 3]),
        lambda: pdf([], range(1, 6)),
        lambda: pdf([1, 2, 3], []),
        lambda: to_hist([]),
    ],
)
def test_invalid_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()
