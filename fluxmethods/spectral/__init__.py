"""The parts of the spectral chain that are estimators rather than facade.

``lpfc`` is the wind/stability look-up correction; ``sensor_table`` reads a sensor
geometry table and stamps what it read. The analytic transfer functions, the
fitting models and the in-situ chain stay in the reference implementation: they
are registered inside its facade module and resolve units and variable names,
which is policy rather than method.
"""

from . import lpfc, sensor_table
from .lpfc import load_lpfc_lut_table, lookup_lpfc_cf, lpfc_lut
from .sensor_table import load_sensor_geometry, sensor_geometry, stamp_manifest

__all__ = ["lpfc_lut", "lookup_lpfc_cf", "load_lpfc_lut_table",
           "sensor_geometry", "load_sensor_geometry", "stamp_manifest",
           "lpfc", "sensor_table"]
