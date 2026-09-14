"""Putting a series onto the timestamps another one uses.

``block_average`` here is the mean of the samples falling in each target
interval. The ``block_average`` in :mod:`fluxmethods.detrending` is a different
method on a different step -- it removes a period's mean in place.
"""

from . import commonly_used
from .commonly_used import block_average, fft_resample, linear, nearest

__all__ = ["nearest", "linear", "fft_resample", "block_average", "commonly_used"]
