"""Kraken adapter. kraken.py is imported, never modified.

Why an adapter is needed: kraken.py sets exit pressure equal to ambient pressure,
i.e. it assumes a perfectly expanded nozzle at every altitude. That implies an
11.3 in exit diameter on a 4 in motor at 30 km and over-predicts thrust ~13% high
up -- which an apogee-maximising optimizer will happily exploit. So we call Kraken
once at a design altitude, freeze the nozzle, and compute off-design thrust as

    F = mdot * u_e + (p_e - p_a) * A_e

Derived, not chosen: throat area scales with propellant mass (to hold 38 bar), so
the two stages cannot share a nozzle -- the grain is a core burner, so burn area
scales with length and a common throat would drive chamber pressure to hundreds of
bar. Exit area is additionally capped by the casing bore.

Inherited assumptions, all unvalidated while Kraken is being built: burn time is
fixed at 9.78 s (self-consistent -- constant web on a radially burning grain), and
mass flow is constant. A real progressive core burner would put peak thrust, peak
acceleration and peak dynamic pressure later and higher than modelled.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kraken import Kraken  # noqa: E402

BAR = 1.0e5


def max_propellant_mass(max_motor_length_m: float) -> float:
    """Largest propellant mass fitting the length cap, per kraken.py's own build-up."""
    grain = max_motor_length_m - Kraken.nozzle_closure_length - Kraken.forward_closure_length
    return max(grain, 0.0) * Kraken.p_density * Kraken.cross_sectional_area


def bore_exit_area_m2() -> float:
    return math.pi * (Kraken.propellant_OD / 2.0) ** 2


def _c_star() -> float:
    g = Kraken.p_gamma
    ideal = math.sqrt(g * Kraken.p_specific_gas_constant * Kraken.p_flame_temperature) / (
        g * math.sqrt((2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0)))
    )
    return Kraken.c_star_efficency * ideal


def _expansion_ratio(pe_bar: float) -> float:
    g, pc = Kraken.p_gamma, Kraken.chamber_pressure
    return (
        (2.0 / (g + 1.0)) ** (1.0 / (g - 1.0))
        * (pc / pe_bar) ** (1.0 / g)
        / math.sqrt((g + 1.0) / (g - 1.0) * (1.0 - (pe_bar / pc) ** ((g - 1.0) / g)))
    )


def _thrust_coefficient(pe_bar: float) -> float:
    g, pc = Kraken.p_gamma, Kraken.chamber_pressure
    ideal = (
        2.0 * g**2 / (g - 1.0)
        * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0))
        * (1.0 - (pe_bar / pc) ** ((g - 1.0) / g))
    ) ** 0.5
    return Kraken.c_thrust_efficency * ideal


def _pressure_for_ratio(target: float) -> float:
    """Invert the expansion ratio by bisection; it is monotonic in exit pressure."""
    lo, hi = 1e-6, Kraken.chamber_pressure * 0.99
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _expansion_ratio(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Motor:
    propellant_mass_kg: float
    burn_time_s: float
    mass_flow_kg_s: float
    dry_mass_kg: float
    wet_mass_kg: float
    length_m: float
    cg_from_aft_m: float
    throat_area_m2: float
    exit_area_m2: float
    expansion_ratio: float
    exit_pressure_pa: float
    exhaust_velocity_m_s: float
    bore_limited: bool

    def thrust(self, ambient_pressure_pa: float) -> float:
        momentum = self.mass_flow_kg_s * self.exhaust_velocity_m_s
        pressure = (self.exit_pressure_pa - ambient_pressure_pa) * self.exit_area_m2
        return max(momentum + pressure, 0.0)

    def mass(self, t_since_ignition_s: float) -> float:
        burned = self.mass_flow_kg_s * min(max(t_since_ignition_s, 0.0), self.burn_time_s)
        return self.wet_mass_kg - burned

    def exit_pressure_ratio(self, ambient_pressure_pa: float) -> float:
        """p_e / p_a. Below ~0.4 the nozzle flow separates."""
        if ambient_pressure_pa <= 0.0:
            return float("inf")
        return self.exit_pressure_pa / ambient_pressure_pa


def build_motor(propellant_mass_kg: float, design_altitude_m: float = 0.0) -> Motor:
    from .atmosphere import air_at

    if propellant_mass_kg <= 0.0:
        raise ValueError("propellant mass must be positive")

    raw = Kraken(propellant_mass_kg)
    dry_mass, wet_mass, cg, length = raw.get_mass_properties()

    burn_time = Kraken.burn_time
    mass_flow = propellant_mass_kg / burn_time
    c_star = _c_star()
    throat_area = mass_flow * c_star / (Kraken.chamber_pressure * BAR)

    design_bar = max(air_at(design_altitude_m).pressure_pa, 1.0) / BAR
    wanted = _expansion_ratio(design_bar) * throat_area
    cap = bore_exit_area_m2()
    bore_limited = wanted > cap
    exit_area = min(wanted, cap)
    ratio = exit_area / throat_area
    if bore_limited:
        # Less expanded than asked for, so exit pressure is higher than the
        # design altitude implies. Recover it from the achievable ratio.
        design_bar = _pressure_for_ratio(ratio)

    return Motor(
        propellant_mass_kg=propellant_mass_kg,
        burn_time_s=burn_time,
        mass_flow_kg_s=mass_flow,
        dry_mass_kg=dry_mass,
        wet_mass_kg=wet_mass,
        length_m=length,
        cg_from_aft_m=cg,
        throat_area_m2=throat_area,
        exit_area_m2=exit_area,
        expansion_ratio=ratio,
        exit_pressure_pa=design_bar * BAR,
        exhaust_velocity_m_s=_thrust_coefficient(design_bar) * c_star,
        bore_limited=bore_limited,
    )
