"""Flagging samples that do not belong, within each averaging period."""

from . import mauder_et_al_2013, vickers_et_al_1997
from .mauder_et_al_2013 import mauder2013
from .vickers_et_al_1997 import linear_interpolate_spikes, spike_detection_vickers97

__all__ = ["mauder2013", "spike_detection_vickers97", "linear_interpolate_spikes",
           "mauder_et_al_2013", "vickers_et_al_1997"]
