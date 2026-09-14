"""Finding the delay between the wind and a scalar, and applying it.

``commons`` holds the sample/second arithmetic the others share; ``maximisation``
searches for the covariance peak, ``prescribed`` reads a delay from a table, and
``fixed`` applies a constant one.

These want a ``time`` coordinate in **float seconds** -- ``commons.acq_freq``
derives the rate by differencing it, so a ``datetime64`` coordinate differences to
nanoseconds and gives 1e-8 Hz with no complaint. Pass ``acq_freq=`` explicitly if
your data is timestamped.
"""

from . import commons, fixed, maximisation, prescribed
from .fixed import fix_time_lag
from .maximisation import time_lag, time_lag_w_default
from .prescribed import prescribed_time_lag

__all__ = ["time_lag", "time_lag_w_default", "fix_time_lag", "prescribed_time_lag",
           "commons", "fixed", "maximisation", "prescribed"]
