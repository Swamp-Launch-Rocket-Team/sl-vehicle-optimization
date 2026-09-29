"""US Standard Atmosphere 1976 and altitude-varying gravity.

Analytic, so there is no dependency and no version drift. Valid to 86 km; above
that returns vacuum, which is safe because density there is ~7e-6 kg/m^3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G0 = 9.80665
R_EARTH = 6371000.0
MU_EARTH = 3.986004418e14
R_AIR = 287.05287
GAMMA_AIR = 1.4
CEILING_M = 86000.0

# (base geopotential altitude m, base temperature K, lapse K/m, base pressure Pa)
_LAYERS = (
    (0.0, 288.15, -0.0065, 101325.0),
    (11000.0, 216.65, 0.0, 22632.06),
    (20000.0, 216.65, 0.001, 5474.889),
    (32000.0, 228.65, 0.0028, 868.0187),
    (47000.0, 270.65, 0.0, 110.9063),
    (51000.0, 270.65, -0.0028, 66.93887),
    (71000.0, 214.65, -0.002, 3.956420),
    (84852.0, 186.946, 0.0, 0.3734),
)


@dataclass(frozen=True)
class Air:
    pressure_pa: float
    density_kg_m3: float
    temperature_k: float
    sound_speed_m_s: float

    def mach(self, velocity_m_s: float) -> float:
        """Mach number, or 0.0 in vacuum where it is undefined."""
        if self.sound_speed_m_s <= 0.0:
            return 0.0
        return abs(velocity_m_s) / self.sound_speed_m_s


def air_at(altitude_m: float, temperature_offset_k: float = 0.0) -> Air:
    """Atmospheric state at a geometric altitude."""
    if altitude_m >= CEILING_M:
        return Air(0.0, 0.0, 186.946 + temperature_offset_k, 0.0)

    h = max(altitude_m, 0.0)
    h = R_EARTH * h / (R_EARTH + h)  # geometric -> geopotential

    base_h, base_t, lapse, base_p = _LAYERS[0]
    for layer in _LAYERS:
        if h >= layer[0]:
            base_h, base_t, lapse, base_p = layer
        else:
            break

    if lapse == 0.0:
        temperature = base_t
        pressure = base_p * math.exp(-G0 * (h - base_h) / (R_AIR * base_t))
    else:
        temperature = base_t + lapse * (h - base_h)
        pressure = base_p * (temperature / base_t) ** (-G0 / (R_AIR * lapse))

    temperature += temperature_offset_k
    return Air(
        pressure_pa=pressure,
        density_kg_m3=pressure / (R_AIR * temperature),
        temperature_k=temperature,
        sound_speed_m_s=math.sqrt(GAMMA_AIR * R_AIR * temperature),
    )


def gravity(altitude_m: float) -> float:
    """Inverse-square gravity. 3% lower at 100 km, so worth modelling."""
    return MU_EARTH / (R_EARTH + max(altitude_m, 0.0)) ** 2
