"""Finding the delay between the wind and a scalar, and applying it.

``commons`` holds the sample/second arithmetic the others share; ``maximisation``
searches for the covariance peak, ``prescribed`` reads a delay from a table, and
``fixed`` applies a constant one.
"""

from . import commons, fixed, maximisation, prescribed

__all__ = ["commons", "fixed", "maximisation", "prescribed"]
