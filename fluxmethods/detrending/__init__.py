"""Removing the part of a series that is not turbulence.

``block_average`` here removes the period's mean and leaves the samples where
they are. The ``block_average`` in :mod:`fluxmethods.resampling` is a different
method on a different step -- the mean of the samples in each target interval.
"""

from . import commonly_used
from .commonly_used import block_average, linear_detrend

__all__ = ["block_average", "linear_detrend", "commonly_used"]
