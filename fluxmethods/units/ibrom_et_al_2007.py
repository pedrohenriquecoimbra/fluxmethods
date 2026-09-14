"""EddyPro's route to a dry mixing ratio, sample by sample, in the analyser's cell.

For a closed-path analyser EddyPro does not apply Webb afterwards. It converts
each raw sample to a mixing ratio per mole of *dry* air first, and then there is
nothing left to correct: a dry mixing ratio is conserved under the expansion and
the dilution that the correction exists to undo. Ibrom et al. (2007) is the paper
that argues for doing it this way rather than applying WPL to a closed-path
signal, and this is the transcription of it:

    r = d_gas,i * v_cell,i / (1 - chi_h2o,i)

with, per sample ``i``,

    d_gas,i    the gas molar density the analyser reports  [mmol m-3]
    v_cell,i   the molar volume of the air in the cell, R T_cell / P_cell
    chi_h2o,i  the water-vapour mole fraction in the cell

``d_gas v_cell`` is the mole fraction of the sample; dividing by ``1 - chi_h2o``
re-expresses it per mole of dry air. Sample by sample, not on the means: it is
precisely the *fluctuations* of temperature, pressure and humidity that carry the
density signal, so a conversion applied to the block means would remove nothing.

Cell, not ambient
-----------------
An LI-7200 measures the air *inside its cell*, which has been drawn down a tube,
warmed by the instrument and dropped in pressure by the pump. Its ``CO2_CONC`` is
a cell molar density and its ``CO2`` a cell mole fraction --- neither is an
ambient quantity, and this is why the sample declares ``col_18`` (``cell_t``) and
``col_21`` (``int_p``) at all. So ``v_cell`` here is built from the cell
temperature and the cell pressure, never from ``air_temperature`` and
``air_pressure``, and the refusal below treats their absence as fatal rather than
substituting the ambient ones.

The distinction is not a matter of degree. Applying an *ambient* density
correction to a cell quantity is wrong in kind: the fluctuations it would correct
for are the ones the cell has already damped or imposed, so the correction would
be describing air that is not the air the number came from.

Exact or refuse
---------------
EddyPro applies this only when the same instrument measured H2O, that H2O can be
expressed as a mole fraction, and fast cell temperature and cell pressure are
both present. Each is load-bearing: without the co-located H2O there is no
``chi_h2o`` to divide by, and without the cell state ``v_cell`` is not the
sample's molar volume. Any of them missing and the gas is left alone with a
warning, in the same spirit as
:func:`~..spectral.main.block_average_highpass` refusing a detrending it does not
describe. Half a conversion is not a smaller conversion; it is a different and
unstated quantity.

References
----------
Ibrom, A., Dellwik, E., Larsen, S. E., and Pilegaard, K. (2007). On the use of
the Webb-Pearman-Leuning theory for closed-path eddy correlation measurements.
Tellus B, 59, 937-946.
(``ibrom2007b`` --- not the low-pass filtering paper of the same year.)
"""

# built-in modules
import logging

# 3rd party modules
import xarray as xr

# project modules
from ..core import constants, provenance
from ..core.measure_type import measure_type_of
from ..core.micrometeorology import add_gas_mass_densities

logger = logging.getLogger(__name__)

#: The name this routine is registered under, stamped onto every series it
#: converts. Defined here, where the stamp is written, because ``main`` reads
#: those stamps back to learn which gases were converted -- the two have to agree
#: on one string, and they are in different modules.
METHOD_NAME = 'molardensity_to_drymixingratio'

#: Names a cell temperature may arrive under. ``cell_t`` and ``int_t`` are
#: EddyPro's own tokens (the sample's ``col_18`` and ``col_19``/``col_20``); the
#: rest are the spellings other readers use. A configuration may name one
#: explicitly and that always wins.
CELL_TEMPERATURE_NAMES = ('cell_t', 'cell_temperature', 'int_t', 'int_t_1',
                          'int_t_2', 't_cell')

#: Likewise for the cell pressure. ``int_p`` is EddyPro's (``col_21``).
CELL_PRESSURE_NAMES = ('int_p', 'cell_p', 'cell_pressure', 'p_cell')

#: The unit a converted gas is left in. ``mmol/mol`` falls out of
#: ``[mmol m-3] * [m3 mol-1]`` on its own, so nothing is rescaled by hand.
MIXING_RATIO_UNITS = 'mmol/mol'


def _is_quantified(da):
    """Whether a variable is carrying its units rather than declaring them."""
    return hasattr(getattr(da, 'data', None), 'units')


def _qty(ds, name):
    """``ds[name]`` as a unit-aware variable, whichever state the dataset is in.

    The high-frequency chain runs on a dequantified dataset --- units live in the
    ``units`` attribute, not on the array --- while a caller working on an
    averaged period has them on the array. This conversion is the one correction
    whose arithmetic mixes three different dimensions (a molar density, a
    temperature and a pressure), so it cannot be done on bare magnitudes without
    silently depending on which units each happened to arrive in. It quantifies
    what it needs and hands the result back in whatever state it found.
    """
    da = ds[name]
    return da if _is_quantified(da) else da.pint.quantify()


def _instrument(ds, name):
    """The instrument a variable declares, if it declares one."""
    attrs = ds[name].attrs
    return attrs.get('instrument') or (attrs.get('Attr') or {}).get('instrument')


def _find(ds, candidates, instrument=None, declared=None):
    """A variable from ``candidates``, preferring one on the same instrument.

    An explicitly configured name wins outright and is *not* second-guessed: if a
    run names a variable that is not there, that is a refusal, not a cue to go
    looking for another.
    """
    if declared:
        return declared if declared in ds else None
    present = [n for n in candidates if n in ds]
    if not present:
        return None
    if instrument:
        same = [n for n in present if _instrument(ds, n) == instrument]
        if same:
            return same[0]
    return present[0]


def _refresh_density(ds, gas, attrs, converted, quantified):
    """Rebuild ``rho_<gas>`` from the converted variable, or drop it.

    Micrometeorology took that mean from the *cell* density, before this ran, so
    it now describes a quantity the dataset no longer holds. It is rebuilt by the
    same derivation rather than by a second copy of it here --- one implementation
    of "concentration to density", following the ``measure_type`` this method just
    rewrote. If the molar volumes it needs are absent the stale value is dropped
    anyway: a density correction that finds nothing refuses, which is right, while
    one that finds the old number would scale itself by the wrong air.
    """
    name = f'rho_{gas}'
    ds = ds.drop_vars(name, errors='ignore')
    needed = ('air_molar_volume', 'dry_air_molar_volume')
    if not all(v in ds for v in needed):
        return ds
    view = xr.Dataset({gas: converted, **{v: _qty(ds, v) for v in needed}})
    view[gas].attrs = attrs
    view = add_gas_mass_densities(view)
    if name not in view:
        return ds
    rho = view[name]
    return ds.assign(**{name: rho if quantified else rho.pint.dequantify()})


def _cell_mole_fraction_h2o(ds, h2o, cell_molar_volume):
    """The water-vapour mole fraction *in the cell*, from however H2O is reported.

    Deliberately built here rather than read from ``h2o_mole_fraction``:
    micrometeorology derives that one without asking whether the analyser is
    open- or closed-path, so for a closed-path column it is a cell mole fraction
    under an ambient name. The frame of reference is the whole point of this
    method, so it says which one it means.

    Returns ``None`` when H2O cannot be expressed as a mole fraction, which is one
    of EddyPro's three conditions.
    """
    kind = measure_type_of(ds, h2o)
    water = _qty(ds, h2o)
    if kind == 'mole_fraction':
        return water
    if kind == 'molar_density':
        return water * cell_molar_volume
    if kind == 'mixing_ratio':
        # Per mole of dry air -> per mole of moist air.
        water = water.pint.to('dimensionless')
        return water / (1 + water)
    return None


def ibrom_et_al_2007(ds, select=None, cell_temperature=None, cell_pressure=None,
                     h2o='h2o', **kwargs):
    """Convert each selected gas from a cell molar density to a dry mixing ratio.

    Rewrites the variable in place and, with it, its ``measure_type`` and
    ``units``. That rewriting is the entire interface to the rest of the pipeline:
    the flux conversion, the mean density and whether a density correction is owed
    are all read off ``measure_type`` through one registry, so once this step has
    run, every consequence follows without any other step being told a conversion
    happened. There is deliberately no "converted" flag to check.

    A gas that is not a molar density is passed over rather than refused --- there
    is nothing to convert, and saying so as a refusal would be noise.
    """
    select = select or [v for v in ('co2', 'h2o') if v in ds]
    if isinstance(select, str):
        select = [s.strip() for s in select.split(',') if s.strip()]

    for gas in list(select):
        if gas not in ds:
            logger.warning("ibrom_et_al_2007: %r is not in the dataset; skipping.", gas)
            continue
        if measure_type_of(ds, gas) != 'molar_density':
            logger.debug(
                "ibrom_et_al_2007: %s is not a molar density; nothing to convert.", gas)
            continue

        instrument = _instrument(ds, gas)

        # Condition 1: the same instrument measured H2O. Water from a different
        # analyser is water in a different cell, sampled through a different tube
        # with a different lag -- it does not describe this sample.
        if h2o not in ds:
            logger.warning(
                "ibrom_et_al_2007: no %r in the dataset, so the cell water-vapour "
                "mole fraction the conversion divides by cannot be formed; %s is "
                "left as a molar density.", h2o, gas)
            continue
        if instrument and _instrument(ds, h2o) not in (None, instrument):
            logger.warning(
                "ibrom_et_al_2007: %s is measured by %r but %s by %r. EddyPro "
                "applies this conversion only within one instrument, because the "
                "cell state and the lag are the instrument's own; %s is left as a "
                "molar density.", gas, instrument, h2o, _instrument(ds, h2o), gas)
            continue

        # Condition 3 (checked before 2, which needs the cell volume): fast cell
        # temperature and cell pressure. The ambient ones are not a fallback --
        # see the module docstring.
        t_cell = _find(ds, CELL_TEMPERATURE_NAMES, instrument, cell_temperature)
        p_cell = _find(ds, CELL_PRESSURE_NAMES, instrument, cell_pressure)
        if t_cell is None or p_cell is None:
            logger.warning(
                "ibrom_et_al_2007: %s needs a fast cell temperature and cell "
                "pressure (found %r and %r). The ambient ones describe different "
                "air and are not substituted, so %s is left as a molar density "
                "and the density correction remains owed on it.",
                gas, t_cell, p_cell, gas)
            continue

        # v_cell = R T_cell / P_cell, from the cell's own state.
        cell_molar_volume = (constants.R * _qty(ds, t_cell).pint.to('K')
                             / _qty(ds, p_cell).pint.to('Pa'))

        # Condition 2: that H2O is convertible to a mole fraction.
        chi_h2o = _cell_mole_fraction_h2o(ds, h2o, cell_molar_volume)
        if chi_h2o is None:
            logger.warning(
                "ibrom_et_al_2007: %s is reported in a form that cannot be "
                "expressed as a mole fraction, so the dilution factor cannot be "
                "formed; %s is left as a molar density.", h2o, gas)
            continue

        quantified = _is_quantified(ds[gas])
        attrs = dict(ds[gas].attrs)
        converted = (_qty(ds, gas) * cell_molar_volume
                     / (1 - chi_h2o.pint.to('dimensionless')))
        converted = converted.pint.to(MIXING_RATIO_UNITS)

        # The rewrite that makes the rest of the pipeline follow.
        was = attrs.get('measure_type') or measure_type_of(ds, gas)
        attrs['measure_type'] = 'mixing_ratio'
        attrs['units'] = MIXING_RATIO_UNITS
        if isinstance(attrs.get('Attr'), dict):
            attrs['Attr'] = {**attrs['Attr'], 'measure_type': 'mixing_ratio'}

        # Provenance, on the variable rather than only on the dataset. After the
        # rewrite above a converted series is indistinguishable from one reported as
        # a dry mixing ratio to begin with, and the density correction's decision to
        # stand aside has no visible reason. Recording the measure it came from, and
        # what changed it, is what lets a reader tell the two routes apart in the
        # output -- and what makes "no WPL here" an answer rather than an absence.
        attrs.setdefault('measure_type_reported', was)
        provenance.record_applied(attrs, 'conversion', METHOD_NAME,
                                  on=[gas], measure_from=was,
                                  measure_to='mixing_ratio')

        ds = _refresh_density(ds, gas, attrs, converted, quantified)
        if not quantified:
            converted = converted.pint.dequantify()
        converted.attrs = attrs
        ds = ds.assign(**{gas: converted})

        logger.info(
            "ibrom_et_al_2007: %s converted to a dry mixing ratio in the cell "
            "(%s, %s); it is no longer owed a density correction.",
            gas, t_cell, p_cell)
    return ds
