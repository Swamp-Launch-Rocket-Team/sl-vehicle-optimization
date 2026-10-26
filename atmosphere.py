"""Interchangeable constant and ISA atmosphere models (SI units)."""

from dataclasses import dataclass
from math import exp, isfinite, sqrt

G = 9.80665
R_AIR = 287.05287


@dataclass(frozen=True)
class Atmosphere:
    temperature: float  # K
    pressure: float  # Pa (Kraken instead expects bar)
    density: float  # kg/m^3
    speed_of_sound: float  # m/s


def isa(altitude: float) -> Atmosphere:
    """Dry standard atmosphere, geometric altitude 0 to ~86 km MSL.

    Uses geopotential layer boundaries and hydrostatic pressure integration.
    Layer table: NASA/TM-2005-213659. No weather or humidity corrections.
    """
    if not isfinite(altitude) or altitude < 0:
        raise ValueError("ISA altitude must be finite and nonnegative (meters MSL).")
    height = 6356766.0 * altitude / (6356766.0 + altitude)
    boundaries = (0., 11000., 20000., 32000., 47000., 51000., 71000., 84852.)
    lapse_rates = (-0.0065, 0., 0.001, 0.0028, 0., -0.0028, -0.002)
    if height > boundaries[-1]:
        raise ValueError("Altitude exceeds the supported ISA layers (~86 km MSL).")
    temperature, pressure = 288.15, 101325.0
    for base, top, lapse in zip(boundaries, boundaries[1:], lapse_rates):
        delta = min(height, top) - base
        next_temperature = temperature + lapse * delta
        if lapse == 0:
            pressure *= exp(-G * delta / (R_AIR * temperature))
        else:
            pressure *= (temperature / next_temperature) ** (G / (R_AIR * lapse))
        temperature = next_temperature
        if height <= top:
            break
    return Atmosphere(temperature, pressure, pressure / (R_AIR * temperature),
                      sqrt(1.4 * R_AIR * temperature))


class ConstantAtmosphere:
    """Hold dry-air conditions fixed throughout a flight.

    Density and sound speed are derived from pressure [Pa] and temperature [K]
    for consistency. Defaults represent standard sea-level conditions.
    """

    def __init__(self, *, temperature: float = 288.15,
                 pressure: float = 101325.0) -> None:
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive (K).")
        if not isfinite(pressure) or pressure <= 0:
            raise ValueError("Pressure must be finite and positive (Pa).")
        self.conditions = Atmosphere(
            temperature=temperature,
            pressure=pressure,
            density=pressure / (R_AIR * temperature),
            speed_of_sound=sqrt(1.4 * R_AIR * temperature),
        )

    def __call__(self, altitude: float) -> Atmosphere:
        """Accept meters MSL like ISA; hold conditions independent of altitude."""
        if not isfinite(altitude) or altitude < 0:
            raise ValueError("Altitude must be finite and nonnegative (meters MSL).")
        return self.conditions
