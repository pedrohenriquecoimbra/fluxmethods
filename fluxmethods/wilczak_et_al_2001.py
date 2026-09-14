"""This module implements the double, triple rotation and planar fit corrections for wind components
as described by Wilczak et al. (2001). The corrections are applied to the u, v, and w components of wind data.
It provides functions for both double and triple rotation corrections, allowing for the correction of wind data
to account for the tilt of the anemometer and the alignment of the mean wind vector.
"""

# built-in modules
import re
import os
import warnings
import logging
from functools import reduce

# 3rd party modules
import numpy as np
import pandas as pd
import scipy as sp

# project modules

logger = logging.getLogger('ep.corrections.axis_rotation.wilczak_et_al_2001')

def double_rotation(u, v, w, verbosity=0):
    """
    Perform a double rotation correction on the input wind components u, v, w.
    This function applies the first rotation to align the mean wind vector with the x-axis,
    and the second rotation to align the mean vertical wind component with the z-axis.

    Parameters:
        u (array-like): The u component of the wind.
        v (array-like): The v component of the wind.
        w (array-like): The w component of the wind.
        verbosity (int): Level of verbosity for debug output.

    Returns:
        tuple: Corrected u, v, w components and the angles of the first (theta) and second (phi) rotations.
    """
    # Ensure inputs are numpy arrays
    u = np.asarray(u)
    v = np.asarray(v)
    w = np.asarray(w)

    # first rotation: align the x-axis with the mean horizontal wind. atan2 (not
    # atan) is required so u aligns to +|wind| for every wind direction; atan
    # collapses the quadrant and flips u to -|wind| whenever mean(u) < 0.
    theta = np.arctan2(np.nanmean(v), np.nanmean(u))
    u1 = u * np.cos(theta) + v * np.sin(theta)
    v1 = -u * np.sin(theta) + v * np.cos(theta)
    w1 = w

    # second rotation: tilt into the mean streamline so mean(w) -> 0.
    phi = np.arctan2(np.nanmean(w1), np.nanmean(u1))
    u2 = u1 * np.cos(phi) + w1 * np.sin(phi)
    v2 = v1
    w2 = -u1 * np.sin(phi) + w1 * np.cos(phi)
    
    if verbosity > 0:
        print(f"Double rotation angles: theta={theta}, phi={phi}")
    
    # Return the corrected components and angles as a named tuple-like object
    return type('var_', (object,), 
                {"u": u2, 
                 "v": v2, 
                 "w": w2, 
                 "theta": theta,
                 "phi": phi,
                 "meta": {}})


def triple_rotation(u, v, w, verbosity=0):
    r"""
    Perform a triple rotation correction on the input wind components u, v, w.
    This function applies the double rotation method followed by a third rotation
    to correct for the tilt of the anemometer.

    The third-rotation angle follows Wilczak et al. (2001, eq. 3):
    tan(2*psi) = 2<v'w'> / (<v'^2> - <w'^2>), i.e. psi is HALF the arctangent --
    dropping the 1/2 doubles the angle. atan2 keeps the quadrant when the variance
    difference is negative. EddyPro additionally *skips* the third rotation
    when \|psi\| > 10 deg; that guard is EddyPro's own, not part of Wilczak et
    al. (2001), so it is deliberately NOT transcribed here -- the cost of that
    formulation difference is measured, per window class, by
    ``tests/test_eddypro_variants.py`` (``ep_rot_triple_guarded``).

    Parameters:
        u (array-like): The u component of the wind.
        v (array-like): The v component of the wind.
        w (array-like): The w component of the wind.
        verbosity (int): Level of verbosity for debug output.

    Returns:
        tuple: Corrected u, v, w components and the angle of the third rotation (psi).
    """
    #first and second rotations
    dr = double_rotation(u, v, w, verbosity)
    u2, v2, w2, theta, phi = dr.u, dr.v, dr.w, dr.theta, dr.phi

    #third rotation: psi = 0.5 * atan2(2<v'w'>, <v'^2> - <w'^2>) (Wilczak 2001)
    psi = 0.5 * np.arctan2(2 * np.nanmean(v2 * w2),
                           np.nanmean(v2**2) - np.nanmean(w2**2))
    u3 = u2
    v3 = v2 * np.cos(psi) + w2 * np.sin(psi)
    w3 = -v2 * np.sin(psi) + w2 * np.cos(psi)
    
    if verbosity > 0:
        print(f"First rotation angles: theta={theta}, phi={phi}, psi={psi}")
    if verbosity > 0:
        print(f"Mean w after triple rotation: {np.nanmean(w3)}")

    # Return the corrected components and angles as a named tuple-like object
    return type('var_', (object,), 
                {"u": u3, 
                 "v": v3, 
                 "w": w3, 
                 "theta": theta, 
                 "phi": phi, 
                 "psi": psi,
                 "meta": {}})


def findB(u, v, w):
    """Least-squares plane w = b0 + b1 u + b2 v (Wilczak et al. 2001).

    The normal-equation matrix is built from *sums over the samples*, not the
    scalar means: with the means, every row is a multiple of ``[1, u, v]`` and
    the matrix is rank-1 (exactly singular).

    Module-level (rather than nested in :func:`planarfit`) so the fit itself can
    be validated in isolation: EddyPro's planar fit applies this same plane fit
    to a *population* of window-mean winds per wind sector, while
    :func:`planarfit` applies it to the samples of a single window, and the
    ground-truth tests need to score the two formulations separately.
    """
    mask = np.isfinite(u) & np.isfinite(v) & np.isfinite(w)
    u, v, w = u[mask], v[mask], w[mask]
    n = float(u.size)
    su, sv, sw = float(u.sum()), float(v.sum()), float(w.sum())
    suv, suw, svw = float((u * v).sum()), float((u * w).sum()), float((v * w).sum())
    su2, sv2 = float((u * u).sum()), float((v * v).sum())

    H = np.matrix([[n, su, sv], [su, su2, suv], [sv, suv, sv2]])
    g = np.matrix([sw, suw, svw]).T
    x = sp.linalg.solve(H, g)
    return x[0, 0], x[1, 0], x[2, 0]


def planarfit(u, v, w, verbosity=0, b0=None, b1=None, b2=None):
    """Perform a planar fit correction on the input wind components u, v, w.
    This function calculates the mean wind vector and applies a rotation to align it with the z-axis.

    Parameters:
        u (array-like): The u component of the wind.
        v (array-like): The v component of the wind.
        w (array-like): The w component of the wind.
        verbosity (int): Level of verbosity for debug output.
        b0, b1, b2 (float, optional): Precomputed coefficients of the plane
            ``w = b0 + b1*u + b2*v``. Wilczak et al. (2001) define the plane
            over a long-term population of window-mean winds, then rotate every
            window with that one plane; only the final alignment of u with the
            window's mean wind stays per-window. Passing the coefficients (all
            three, e.g. from :func:`findB` on the population, carried by the
            config block like a lag table) applies those semantics; leaving
            them out fits the plane on this window's own samples (unchanged
            per-window behaviour).

    Returns:
        tuple: Corrected u, v, w components after planar fit correction.
    """

    # Ensure inputs are numpy arrays
    u = np.asarray(u)
    v = np.asarray(v)
    w = np.asarray(w)

    meanU = np.nanmean(u)
    meanV = np.nanmean(v)
    meanW = np.nanmean(w)

    given = [c is not None for c in (b0, b1, b2)]
    if all(given):
        b0, b1, b2 = float(b0), float(b1), float(b2)
    elif any(given):
        raise ValueError(
            "planarfit takes either all of b0, b1, b2 (a precomputed plane) "
            "or none of them (fit on this window's samples)")
    else:
        b0, b1, b2 = findB(u, v, w)

    Deno = np.sqrt(1 + b1 ** 2 + b2 ** 2)
    p31 = -b1 / Deno
    p32 = -b2 / Deno
    p33 = 1 / Deno

    cosγ = p33 / np.sqrt(p32**2+p33**2)
    sinγ = -p32 / np.sqrt(p32**2 + p33**2)
    cosβ = np.sqrt(p32**2 + p33**2)
    sinβ = p31

    R2 = np.matrix([[1, 0, 0],
                    [0, cosγ, -sinγ],
                    [0, sinγ, cosγ]])
    R3 = np.matrix([[cosβ, 0, sinβ],
                    [0, 1, 0],
                    [-sinβ, 0, cosβ]])

    A0 = R3.T * R2.T * [[meanU], [meanV], [meanW]]

    α = np.arctan2(A0[1].tolist()[0][0],
                   A0[0].tolist()[0][0])

    R1 = np.matrix([[np.cos(α), -np.sin(α), 0],
                    [np.sin(α), np.cos(α), 0],
                    [0, 0, 1]])

    A1 = R1.T * ((R3.T * R2.T) * np.matrix([u, v, w - b0]))

    U1 = np.array(A1[0])[0]
    V1 = np.array(A1[1])[0]
    W1 = np.array(A1[2])[0]

    if verbosity > 0:
        print(f"Planar fit angles: α={α}, β={np.arctan2(sinβ, cosβ)}, γ={np.arctan2(sinγ, cosγ)}")
        print(f"Mean w after planar fit: {np.nanmean(W1)}")

    if type(u) == pd.Series:
        U1 = pd.Series(U1)
    if type(v) == pd.Series:
        V1 = pd.Series(V1)
    if type(w) == pd.Series:
        W1 = pd.Series(W1)

    # Return the corrected components as a named tuple-like object
    return type('var_', (object,), 
                {"u": U1, 
                 "v": V1, 
                 "w": W1,
                 "meta": {}})
