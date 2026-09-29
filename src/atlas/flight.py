"""Vertical 1 DOF flight simulation.

State is (altitude, velocity). Fixed-step RK4, phase by phase, with the apogee
crossing interpolated so the result does not depend on landing on a step boundary.

Phases: burn_1 -> coast_stack -> coast_s2 -> burn_2 -> coast_final

1 DOF has no gravity turn, no wind and no angle of attack, so apogee is
OVER-predicted. It verifies nothing about attitude, weathercocking, tip-off,
pitch-back at separation, roll, or control authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .aero import drag_force_n
from .atmosphere import G0, air_at, gravity
from .rocket import Rocket

MAX_FLIGHT_TIME_S = 2000.0


@dataclass
class Sample:
    t: float
    altitude_m: float
    velocity_m_s: float
    mach: float
    dynamic_pressure_pa: float
    phase: str


@dataclass
class Flight:
    samples: list[Sample] = field(default_factory=list)
    apogee_m: float = 0.0
    apogee_time_s: float = 0.0
    rail_exit_velocity_m_s: float = 0.0
    max_mach: float = 0.0
    max_dynamic_pressure_pa: float = 0.0
    max_acceleration_g: float = 0.0
    separation_altitude_m: float = 0.0
    ignition_altitude_m: float = 0.0
    ignition_pressure_pa: float = 0.0
    burnout_1_altitude_m: float = 0.0
    burnout_2_altitude_m: float = 0.0
    cd_extrapolated: bool = False

    def max_mach_in(self, phases: tuple[str, ...]) -> float:
        values = [s.mach for s in self.samples if s.phase in phases]
        return max(values) if values else 0.0

    def peak_q_sample(self) -> Sample | None:
        return max(self.samples, key=lambda s: s.dynamic_pressure_pa) if self.samples else None


def simulate(
    rocket: Rocket,
    *,
    temperature_offset_k: float = 0.0,
    cd_multiplier: float = 1.0,
    dt: float = 0.01,
    record_every: int = 10,
) -> Flight:
    """Fly the rocket and return the trajectory."""
    from .aero import CD_MAX_MACH

    config = rocket.config
    flight = Flight()
    s1, s2 = rocket.stage_1, rocket.stage_2
    area = rocket.reference_area_m2

    altitude = config.launch_elevation_m
    ground = altitude
    velocity = 0.0
    t = 0.0
    rail_cleared = False
    step_index = 0

    phases = (
        ("burn_1", s1.motor.burn_time_s, True, False),
        ("coast_stack", config.t_separation_s, False, False),
        ("coast_s2", rocket.design.t_ignite_s, False, True),
        ("burn_2", s2.motor.burn_time_s, True, True),
        ("coast_final", MAX_FLIGHT_TIME_S, False, True),
    )

    for phase, duration, powered, stage2_alone in phases:
        elapsed = 0.0
        while elapsed < duration - 1e-12:
            step = min(dt, duration - elapsed)

            if phase == "burn_1":
                mass = s1.mass_at(elapsed) + s2.mass_at(None)
            elif phase == "coast_stack":
                mass = s1.mass_at(s1.motor.burn_time_s) + s2.mass_at(None)
            elif phase == "coast_s2":
                mass = s2.mass_at(None)
            elif phase == "burn_2":
                mass = s2.mass_at(elapsed)
            else:
                mass = s2.mass_at(s2.motor.burn_time_s)

            def acceleration(h: float, v: float) -> float:
                air = air_at(h, temperature_offset_k)
                if phase == "burn_1":
                    thrust = s1.motor.thrust(air.pressure_pa)
                elif phase == "burn_2":
                    thrust = s2.motor.thrust(air.pressure_pa)
                else:
                    thrust = 0.0
                drag = drag_force_n(
                    v, air.density_kg_m3, area, air.mach(v),
                    stage2_alone=stage2_alone, powered=powered,
                    cd_multiplier=cd_multiplier,
                )
                if v < 0.0:
                    drag = -drag
                return (thrust - drag) / mass - gravity(h)

            a1 = acceleration(altitude, velocity)
            a2 = acceleration(altitude + 0.5 * step * velocity, velocity + 0.5 * step * a1)
            a3 = acceleration(
                altitude + 0.5 * step * (velocity + 0.5 * step * a1), velocity + 0.5 * step * a2
            )
            a4 = acceleration(
                altitude + step * (velocity + 0.5 * step * a2), velocity + step * a3
            )

            new_velocity = velocity + step / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
            new_altitude = altitude + step * (velocity + step / 6.0 * (a1 + a2 + a3))

            # On the rail the vehicle cannot sink into the pad; a thrust-to-weight
            # below 1 shows up as a failed rail exit check instead.
            if not rail_cleared and new_velocity < 0.0:
                new_velocity, new_altitude = 0.0, altitude

            air = air_at(altitude, temperature_offset_k)
            mach = air.mach(velocity)
            q = 0.5 * air.density_kg_m3 * velocity**2

            flight.max_mach = max(flight.max_mach, mach)
            flight.max_dynamic_pressure_pa = max(flight.max_dynamic_pressure_pa, q)
            flight.max_acceleration_g = max(
                flight.max_acceleration_g, abs(a1 + gravity(altitude)) / G0
            )
            if mach > CD_MAX_MACH:
                flight.cd_extrapolated = True

            if step_index % record_every == 0:
                flight.samples.append(Sample(t, altitude, velocity, mach, q, phase))
            step_index += 1

            if not rail_cleared and (new_altitude - ground) >= config.rail_length_m:
                rail_cleared = True
                flight.rail_exit_velocity_m_s = new_velocity

            if phase == "coast_final" and new_velocity <= 0.0 < velocity:
                frac = velocity / (velocity - new_velocity)
                flight.apogee_m = altitude + frac * (new_altitude - altitude)
                flight.apogee_time_s = t + frac * step
                return flight

            altitude, velocity = new_altitude, new_velocity
            t += step
            elapsed += step

        if phase == "burn_1":
            flight.burnout_1_altitude_m = altitude
        elif phase == "coast_stack":
            flight.separation_altitude_m = altitude
        elif phase == "coast_s2":
            flight.ignition_altitude_m = altitude
            flight.ignition_pressure_pa = air_at(altitude, temperature_offset_k).pressure_pa
        elif phase == "burn_2":
            flight.burnout_2_altitude_m = altitude

    flight.apogee_m = altitude
    flight.apogee_time_s = t
    return flight
