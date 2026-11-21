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


#: EddyPro's limit on the third rotation, in degrees (``ThirdRotation``).
EDDYPRO_ROLL_LIMIT_DEG = 10.0


def _warn_where_eddypro_differs(psi, cross, spread):
    """Log a warning when EddyPro's third rotation would not be this one.

    EddyPro takes ``roll = 0.5*atan(cross/spread)`` and applies it only when
    ``|roll| < 10 deg``; with ``spread == 0`` it does not rotate. Here
    ``psi = 0.5*atan2(cross, spread)`` and is always applied. The two agree when
    ``spread > 0`` and ``|psi| < 10 deg``, and only then.
    """
    psi_deg = float(np.degrees(psi))
    if not np.isfinite(psi_deg) or abs(psi_deg) < EDDYPRO_ROLL_LIMIT_DEG:
        return
    if spread < 0:
        roll = float(np.degrees(0.5 * np.arctan(cross / spread)))
        eddypro = (f"<v'^2> < <w'^2>, so EddyPro's half atan (not atan2) gives "
                   f"{roll:.2f} deg instead, which it would ")
        eddypro += ("apply" if abs(roll) < EDDYPRO_ROLL_LIMIT_DEG else
                    "not apply either, leaving the period double-rotated")
    else:
        eddypro = (f"EddyPro rotates a third time only when |psi| < "
                   f"{EDDYPRO_ROLL_LIMIT_DEG:g} deg and would leave this period "
                   f"double-rotated")
    logger.warning(
        "triple_rotation: third-rotation angle psi = %.2f deg, applied (Wilczak "
        "et al. 2001 set no limit); %s.", psi_deg, eddypro)


def triple_rotation(u, v, w, verbosity=0):
    r"""
    Perform a triple rotation correction on the input wind components u, v, w.
    This function applies the double rotation method followed by a third rotation
    to correct for the tilt of the anemometer.

    The third-rotation angle follows Wilczak et al. (2001, eq. 3):
    tan(2*psi) = 2<v'w'> / (<v'^2> - <w'^2>), i.e. psi is HALF the arctangent --
    dropping the 1/2 doubles the angle. atan2 keeps the quadrant when the variance
    difference is negative. EddyPro (7.0.9, ``ThirdRotation`` in
    ``tilt_correction.f90``) differs in two ways: it rotates a third time only
    when \|psi\| < 10 deg, and it takes the half *atan* of the ratio, not atan2,
    so where <v'^2> < <w'^2> its angle is this one less 90 deg. Neither is in
    Wilczak et al. (2001), so neither is transcribed here: the rotation is
    applied whatever the angle. A period where EddyPro would have done otherwise
    (\|psi\| >= 10 deg, which every period with <v'^2> < <w'^2> is) is logged as a
    warning that says what EddyPro would have done. The cost of the guard is
    measured, per window class, by onefluxpro's
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
    cross = 2 * np.nanmean(v2 * w2)
    spread = np.nanmean(v2**2) - np.nanmean(w2**2)
    psi = 0.5 * np.arctan2(cross, spread)
    _warn_where_eddypro_differs(psi, cross, spread)
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

    This is how the plane :func:`planarfit` takes is obtained: Wilczak et al.
    (2001, sect. 3) fit it to a *population* of window-mean winds (``ū_j, v̄_j,
    w̄_j`` over many runs), as EddyPro does per wind sector, and then apply that
    one plane to every window. Handed one window's samples instead, the slope of
    w on u is about ``cov(u', w') / var(u')``, so the "plane" is the momentum
    flux itself and rotating into it removes most of ``u'w'``.

    Fewer than three finite points, or points that do not span a plane (all on
    one line), leave the plane undetermined: the answer is NaN for all three
    coefficients rather than a ``LinAlgError``, the same way a rotation of a
    period with no finite wind is NaN.
    """
    u, v, w = (np.asarray(a, dtype=float).ravel() for a in (u, v, w))
    mask = np.isfinite(u) & np.isfinite(v) & np.isfinite(w)
    u, v, w = u[mask], v[mask], w[mask]
    if u.size < 3:
        return np.nan, np.nan, np.nan
    n = float(u.size)
    su, sv, sw = float(u.sum()), float(v.sum()), float(w.sum())
    suv, suw, svw = float((u * v).sum()), float((u * w).sum()), float((v * w).sum())
    su2, sv2 = float((u * u).sum()), float((v * v).sum())

    H = np.matrix([[n, su, sv], [su, su2, suv], [sv, suv, sv2]])
    g = np.matrix([sw, suw, svw]).T
    try:
        x = sp.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    return x[0, 0], x[1, 0], x[2, 0]


def planarfit(u, v, w, verbosity=0, b0=None, b1=None, b2=None):
    """Perform a planar fit correction on the input wind components u, v, w.
    This function calculates the mean wind vector and applies a rotation to align it with the z-axis.

    Parameters:
        u (array-like): The u component of the wind.
        v (array-like): The v component of the wind.
        w (array-like): The w component of the wind.
        verbosity (int): Level of verbosity for debug output.
        b0, b1, b2 (float): Coefficients of the plane ``w = b0 + b1*u + b2*v``,
            all three, required. Wilczak et al. (2001, sect. 3) fit the plane by
            least squares over a long-term population of window-mean winds and
            rotate every window with that one plane; only the final alignment of
            u with the window's mean wind stays per window. Obtain them with
            :func:`findB` on the window means of a long unrotated record and
            carry them in the step's config block (``[[axis_rotation]] b0 = ..``).
            One plane for every wind direction: per-sector planes (EddyPro's
            sector fit, GEddySoft's tilt file) are not read here.

    Raises:
        ValueError: when the plane is missing, partial or not finite. There is
            no per-window fallback: fitting the plane on one window's own
            samples makes w uncorrelated with u by construction and strips most
            of the momentum flux (about 75-90 % on synthetic and FR-Gri data).

    Returns:
        tuple: Corrected u, v, w components after planar fit correction.
    """

    # One series per component: the rotation below is built on a 3 x N matrix,
    # and a period arrives as (date, time). The shape goes back on the way out.
    shape = np.shape(u)
    u = np.asarray(u).ravel()
    v = np.asarray(v).ravel()
    w = np.asarray(w).ravel()

    meanU = np.nanmean(u)
    meanV = np.nanmean(v)
    meanW = np.nanmean(w)

    given = [c is not None for c in (b0, b1, b2)]
    if not all(given):
        raise ValueError(
            "planarfit needs the plane w = b0 + b1*u + b2*v, all three of b0, b1, "
            f"b2 (got {'only some' if any(given) else 'none'} of them). Wilczak et "
            "al. (2001) fit it once over a population of window-mean winds -- "
            "findB on the window means of a long unrotated record -- and apply "
            "it to every window; set b0, b1, b2 in [Corrections][[axis_rotation]]. "
            "Refusing rather than fitting the plane on this window's own samples, "
            "which removes most of the momentum flux.")
    b0, b1, b2 = float(b0), float(b1), float(b2)
    if not np.all(np.isfinite([b0, b1, b2])):
        raise ValueError(
            f"planarfit: the plane (b0={b0}, b1={b1}, b2={b2}) is not finite; a "
            "population too small to fit (findB returns NaN) gives no plane.")

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

    U1 = np.array(A1[0])[0].reshape(shape)
    V1 = np.array(A1[1])[0].reshape(shape)
    W1 = np.array(A1[2])[0].reshape(shape)

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
