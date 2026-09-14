"""The spectral corrections: what the instrument and the averaging lost.

Two halves, split the way the reference implementation's registry splits them,
on ``requires_spectra``:

* :mod:`.analytic` derives the loss from the instrument's geometry and a model
  cospectrum, and needs no measured spectrum. :mod:`.eddypro` is EddyPro's own
  band-pass correction, transcribed, and belongs to the same half.
* :mod:`.measured` fits a transfer function to the spectra a site actually
  measured. It carries the machinery that needs -- :mod:`.commons` for the
  integration and the fitting, :mod:`.fitting_models` for the models themselves.

:mod:`.lpfc` is the wind/stability look-up factor, and :mod:`.sensor_table`
reads the instrument geometry the analytic half asks for. That table is *data*:
``instruments/`` beside this package holds one file per model, and without it the
analytic corrections have no path length to work from.
"""

from . import (analytic, commons, eddypro, fitting_models, lpfc, measured,
               sensor_table)
from .analytic import analytic_tube, block_average_highpass, sonic_response
from .commons import (correction_factor_bracketed,
                      correction_factor_from_transfer_function,
                      fit_cutoff_frequency)
from .eddypro import bpcf_anemometric_fluxes, bpcf_moncrieff_97, bpcf_momentum
from .fitting_models import (COSPECTRAL_MODELS, TRANSFER_FUNCTION_MODELS,
                             lorentzian, transfer_function_generator)
from .lpfc import load_lpfc_lut_table, lookup_lpfc_cf, lpfc_lut
from .measured import (cutoff_lut, fully_analytical, generic_experimental,
                       ibrom_et_al_2007)
from .sensor_table import load_sensor_geometry, sensor_geometry, stamp_manifest

__all__ = [
    # analytic -- no measured spectrum needed
    "block_average_highpass", "sonic_response", "analytic_tube",
    "bpcf_moncrieff_97", "bpcf_anemometric_fluxes", "bpcf_momentum",
    "lpfc_lut", "lookup_lpfc_cf", "load_lpfc_lut_table",
    # fitted to measured spectra
    "ibrom_et_al_2007", "generic_experimental", "fully_analytical", "cutoff_lut",
    "correction_factor_from_transfer_function", "correction_factor_bracketed",
    "fit_cutoff_frequency", "transfer_function_generator", "lorentzian",
    "TRANSFER_FUNCTION_MODELS", "COSPECTRAL_MODELS",
    # the instrument table the analytic half reads
    "sensor_geometry", "load_sensor_geometry", "stamp_manifest",
    "analytic", "commons", "eddypro", "fitting_models", "lpfc", "measured",
    "sensor_table",
]
