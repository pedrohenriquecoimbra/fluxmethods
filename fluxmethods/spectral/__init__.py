"""The parts of the spectral chain that are estimators rather than facade.

``analytic`` computes the loss from the instrument's geometry and a model
cospectrum, needing no measured spectrum; ``eddypro`` is EddyPro's own band-pass
correction, transcribed; ``lpfc`` is the wind/stability look-up factor; and
``sensor_table`` reads a sensor geometry table and stamps what it read.

What is **not** here is the half of the chain that fits a transfer function to a
site's own measured spectra (``requires_spectra=True`` in the reference
implementation's registry). Those reach into its units layer, its variable-name
resolution and a registry of fitting models -- policy rather than method.
"""

from . import analytic, eddypro, lpfc, sensor_table
from .analytic import analytic_tube, block_average_highpass, sonic_response
from .eddypro import (bpcf_anemometric_fluxes, bpcf_moncrieff_97, bpcf_momentum)
from .lpfc import load_lpfc_lut_table, lookup_lpfc_cf, lpfc_lut
from .sensor_table import load_sensor_geometry, sensor_geometry, stamp_manifest

__all__ = [
    "block_average_highpass", "sonic_response", "analytic_tube",
    "bpcf_moncrieff_97", "bpcf_anemometric_fluxes", "bpcf_momentum",
    "lpfc_lut", "lookup_lpfc_cf", "load_lpfc_lut_table",
    "sensor_geometry", "load_sensor_geometry", "stamp_manifest",
    "analytic", "eddypro", "lpfc", "sensor_table",
]
