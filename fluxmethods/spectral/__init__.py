"""The parts of the spectral chain that are estimators rather than facade.

``lpfc`` is the wind/stability look-up correction; ``sensor_table`` reads a
sensor geometry table and stamps what it read. The fitting models and the
transfer-function chain stay in the reference implementation: they resolve units
and variable names, which is policy rather than method.
"""

from . import lpfc, sensor_table

__all__ = ["lpfc", "sensor_table"]
