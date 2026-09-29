"""Drag and centre of pressure.

The CD table is a generic slender-rocket PLACEHOLDER, not ATLAS data. Drag is the
largest single driver of predicted apogee, so replace it with a RASAero II export
(it handles supersonic properly and gives booster and sustainer separately).

CP uses Barrowman: nose plus fins, with body interference. It is a small-angle
linear theory, which is exactly where 1 DOF lives, so evaluating it at zero angle
of attack costs nothing in validity. Supersonic CP from this method is approximate
and is flagged as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

CD_MAX_MACH = 6.5
CD_SOURCE = "PLACEHOLDER generic curve - replace with RASAero II"

_MACH = (0.0, 0.5, 0.8, 0.95, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5)
_CD_STACK = (0.45, 0.45, 0.48, 0.62, 0.70, 0.65, 0.56, 0.48, 0.42, 0.38, 0.34, 0.32, 0.30)

NOSE_CP_FACTOR = {
    "conical": 0.666,
    "ogive": 0.466,
    "tangent_ogive": 0.466,
    "von_karman": 0.500,
    "parabolic": 0.500,
    "elliptical": 0.333,
}


def drag_coefficient(mach: float, *, stage2_alone: bool, powered: bool) -> float:
    """CD for the mated stack or the sustainer alone, power on or off."""
    cd = _interpolate(abs(mach))
    if stage2_alone:
        cd *= 0.92          # shorter body, less skin friction
    if not powered:
        cd += 0.06          # base drag with the nozzle not flowing
    return cd


def _interpolate(mach: float) -> float:
    if mach <= _MACH[0]:
        return _CD_STACK[0]
    if mach >= _MACH[-1]:
        return _CD_STACK[-1]
    for i in range(len(_MACH) - 1):
        if _MACH[i] <= mach <= _MACH[i + 1]:
            frac = (mach - _MACH[i]) / (_MACH[i + 1] - _MACH[i])
            return _CD_STACK[i] + frac * (_CD_STACK[i + 1] - _CD_STACK[i])
    return _CD_STACK[-1]


def drag_force_n(
    velocity_m_s: float,
    density_kg_m3: float,
    reference_area_m2: float,
    mach: float,
    *,
    stage2_alone: bool,
    powered: bool,
    cd_multiplier: float = 1.0,
) -> float:
    """Drag magnitude. The caller applies the sign."""
    q = 0.5 * density_kg_m3 * velocity_m_s**2
    cd = drag_coefficient(mach, stage2_alone=stage2_alone, powered=powered)
    return q * reference_area_m2 * cd * cd_multiplier


@dataclass(frozen=True)
class Cp:
    station_from_nose_m: float
    cn_alpha: float
    approximate: bool


def _compressibility(mach: float) -> tuple[float, bool]:
    """Prandtl-Glauert, clamped through the transonic singularity."""
    m = abs(mach)
    if m < 0.85:
        return 1.0 / math.sqrt(1.0 - m * m), False
    if m <= 1.2:
        return 1.0 / math.sqrt(1.0 - 0.85 * 0.85), True
    return 1.0 / math.sqrt(m * m - 1.0), True


def centre_of_pressure(
    *,
    body_diameter_m: float,
    nose_length_m: float,
    nose_shape: str,
    fins,
    fin_root_le_from_nose_m: float,
    mach: float = 0.0,
) -> Cp:
    """Barrowman CP, measured from the nose tip."""
    radius = body_diameter_m / 2.0

    cn_nose = 2.0
    cp_nose = NOSE_CP_FACTOR.get(nose_shape, 0.466) * nose_length_m

    span, cr, ct = fins.semi_span_m, fins.root_chord_m, fins.tip_chord_m
    if span <= 0.0 or cr + ct <= 0.0:
        return Cp(cp_nose, cn_nose, True)

    sweep = fins.mid_chord_sweep_m
    cn_fins = 4.0 * fins.count * (span / body_diameter_m) ** 2 / (
        1.0 + math.sqrt(1.0 + (2.0 * sweep / (cr + ct)) ** 2)
    )
    cn_fins *= 1.0 + radius / (span + radius)     # body interference
    factor, approximate = _compressibility(mach)
    cn_fins *= factor

    cp_fins = (
        fin_root_le_from_nose_m
        + sweep * (cr + 2.0 * ct) / (3.0 * (cr + ct))
        + (1.0 / 6.0) * ((cr + ct) - cr * ct / (cr + ct))
    )

    cn_total = cn_nose + cn_fins
    return Cp((cn_nose * cp_nose + cn_fins * cp_fins) / cn_total, cn_total, approximate)
