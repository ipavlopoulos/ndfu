"""Compatibility wrapper for the old top-level ``src`` package.

Prefer importing directly from ``ndfu`` in new code.
"""

from ndfu import SCALE10, UnimodalLearner, cpdf, dfu, pdf, to_hist

__all__ = ["SCALE10", "UnimodalLearner", "cpdf", "dfu", "pdf", "to_hist"]
