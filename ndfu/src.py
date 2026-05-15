"""Backward-compatible imports for older examples.

Prefer importing directly from ``ndfu`` in new code.
"""

from . import SCALE10, UnimodalLearner, cpdf, dfu, pdf, to_hist

__all__ = ["SCALE10", "UnimodalLearner", "cpdf", "dfu", "pdf", "to_hist"]
