"""fluxmethods: eddy-covariance methods, published as functions and nothing else.

There is no program here. No pipeline, no configuration, no registry, no I/O --
only the estimators, each one a plain function over arrays, importable on its
own. What to do with the numbers is the caller's business.

Each module is named for the work it implements, so a citation and an import are
the same string::

    from fluxmethods.wilczak_et_al_2001 import double_rotation
    from fluxmethods.mauder_et_al_2013 import mauder2013

The methods are re-exported here for convenience; the modules are the contract.
"""

from .mauder_et_al_2013 import mauder2013
from .resampling import block_average, fft_resample, linear, nearest
from .vickers_et_al_1997 import spike_detection_vickers97
from .wilczak_et_al_2001 import double_rotation, planarfit, triple_rotation

__version__ = "0.1.0"

__all__ = [
    "block_average",
    "double_rotation",
    "fft_resample",
    "linear",
    "mauder2013",
    "nearest",
    "planarfit",
    "spike_detection_vickers97",
    "triple_rotation",
]
