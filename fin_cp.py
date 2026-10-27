"""Mach-dependent thin-fin CP, adapted to this repository's FinSet geometry.

OpenRocket technical documentation v13.05, section 3.2.2, equations 3.30-3.36:

This locates a FIN SET's CP, not the combined vehicle CP. It does not supply
supersonic normal-force slopes, body CP, or a whole-vehicle stability model.
"""

import math

from geometry import FinSet


def _fin_cp_fraction(mach: float, aspect_ratio: float) -> float:
    """CP distance aft of MAC leading edge, divided by MAC length.

    Quarter chord through Mach 0.5; Eq. 3.36 quintic up to Mach 2;
    Eq. 3.35 correlation thereafter. Positive denominator is required, but
    is not by itself a validation of the correlation for a given fin shape.
    """
    if not math.isfinite(mach) or mach < 0:
        raise ValueError("Mach must be finite and nonnegative.")
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError("Aspect ratio must be positive and finite.")
    if mach <= 0.5:
        return 0.25

    def supersonic_fraction(m):
        z = aspect_ratio * math.sqrt(m * m - 1.0)
        denominator = 2.0 * z - 1.0
        if denominator <= 0:
            raise ValueError("Supersonic CP correlation outside supported range.")
        return (z - 0.67) / denominator

    if mach >= 2.0:
        return supersonic_fraction(mach)

    beta2 = math.sqrt(3.0)
    z2 = aspect_ratio * beta2
    if 2.0 * z2 - 1.0 <= 0:
        raise ValueError("Invalid aspect ratio for interpolation.")
    f2 = (z2 - 0.67) / (2.0 * z2 - 1.0)
    fp2 = 0.34 * aspect_ratio * 2.0 / beta2 / (2.0 * z2 - 1.0)**2
    t = (mach - 0.5) / 1.5
    D = f2 - 0.25
    E = 1.5 * fp2
    a2 = 10.0 * D - 6.0 * E
    a3 = -20.0 * D + 14.0 * E
    a4 = 15.0 * D - 11.0 * E
    a5 = -4.0 * D + 3.0 * E
    return 0.25 + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5


def fin_cp_location(fin_set: FinSet, mach: float) -> float:
    """Return axial fin-set CP in meters aft of the vehicle nose tip.

    FinSet already stores the absolute root leading-edge position, so no
    separate root_le input is needed. MAC geometry is exact for trapezoidal
    fins (including rectangular and triangular limiting cases).
    """
    root = fin_set.fin_root_chord
    tip = fin_set.fin_tip_chord
    mac = (2.0 / 3.0) * (root**2 + root*tip + tip**2) / (root + tip)
    x_mac_le = (fin_set.fin_leading_edge_sweep_distance
                * (root + 2.0*tip) / (3.0 * (root + tip)))
    fraction = _fin_cp_fraction(mach, fin_set.fin_aspect_ratio)
    return fin_set.fin_leading_edge_axial_position + x_mac_le + fraction * mac
