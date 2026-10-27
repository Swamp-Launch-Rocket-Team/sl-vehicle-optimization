"""Simple x-z point-mass flight from liftoff to first apogee (SI units).

Thrust is vertical; wind is horizontal; drag opposes air-relative velocity.
Only zero-AoA drag is used: no lift, attitude dynamics, rail, or separation kick.
This is a force approximation, not a prediction of attitude/AoA in crosswind.
The default Cd=0.5 is a PLACEHOLDER, not a geometry-dependent drag model.
"""

from dataclasses import dataclass
from math import isfinite, pi
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from vehicle import Vehicle
from atmosphere import Atmosphere, isa

G = 9.80665


class FlightError(RuntimeError):
    """The supplied candidate could not complete the requested ascent."""


@dataclass(frozen=True)
class FlightState:
    time: float
    x: float
    z: float  # height above launch site
    vx: float
    vz: float
    airspeed: float
    mach: float
    mass: float
    reference_diameter: float
    upper_stage_only: bool


def constant_drag(vehicle: Vehicle, state: FlightState, ambient: Atmosphere) -> float:
    """Placeholder; replace with drag_model(vehicle, state, ambient) -> Cd."""
    return 0.5


@dataclass
class Trajectory:
    time: np.ndarray
    x: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vz: np.ndarray
    airspeed: np.ndarray
    mach: np.ndarray
    mass: np.ndarray
    thrust: np.ndarray
    cd: np.ndarray
    density: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    upper_stage_only: np.ndarray

    @property
    def apogee(self) -> float:
        """Height above launch site, meters."""
        return float(self.z[-1])

    def save_csv(self, path: str) -> None:
        names = tuple(self.__dataclass_fields__)
        np.savetxt(path, np.column_stack([getattr(self, name) for name in names]),
                   delimiter=",", header=",".join(names), comments="")


def simulate(
    vehicle: Vehicle,
    *,
    initial_conditions=(0.0, 0.0, 0.0, 0.0),
    launch_altitude: float = 0.0,
    wind_x: float = 0.0,
    atmosphere_model: Callable[[float], Atmosphere] = isa,
    drag_model: Callable[[Vehicle, FlightState, Atmosphere], float] = constant_drag,
    max_time: float = 600.0,
    max_step: float = 0.2,
    rtol: float = 1e-7,
    atol: float = 1e-8,
) -> Trajectory:
    """Run the supplied Vehicle from t=0 to its first downward vz=0 crossing.

    initial_conditions = (x, z, vx, vz), meters and m/s; z is above launch site.
    atmosphere_model accepts altitude [m MSL] and returns an Atmosphere.
    wind_x is a constant signed speed in m/s (+x). Motor ignition times and
    separation_time use the vehicle's flight clock, unchanged by this function.
    Separation drops booster mass/drag area/thrust without changing velocity.
    Starts on the pad when all initial velocities and height are zero; waits
    until thrust exceeds weight. No launch rail or attitude dynamics is modeled.
    Raises if apogee is not reached by max_time. Output times are adaptive and
    include propulsion/configuration events; event samples use post-event data.
    """
    y = np.asarray(initial_conditions, dtype=float)
    if y.shape != (4,) or not np.all(np.isfinite(y)) or y[1] < 0 or y[3] < 0:
        raise ValueError("Supply finite (x, z, vx, vz), with z >= 0 and vz >= 0.")
    for value in (max_time, max_step, rtol, atol):
        if not isfinite(value) or value <= 0:
            raise ValueError("Time limits and integration tolerances must be positive.")
    if not isfinite(wind_x) or not isfinite(launch_altitude) or launch_altitude < 0:
        raise ValueError("Wind must be finite and launch altitude nonnegative.")

    # Geometry is fixed during each run. Integrate shell masses once, rather
    # than repeating the shell quadrature on every ODE evaluation. A fresh call
    # rebuilds these totals, so optimization geometry changes are respected.
    configurations = {}
    for upper_only in (False, True):
        stages = (vehicle.upper_stage,) if upper_only else (
            vehicle.upper_stage, vehicle.booster_stage,
        )
        components = [vehicle.nose_cone]
        for stage in stages:
            components.extend((*stage.body_sections, *stage.fin_sets, *stage.fixed_components))
        structural_mass = sum(component.mass for component in components)
        motors = tuple(stage.motor for stage in stages)
        diameter = (vehicle.upper_stage_reference_diameter if upper_only
                    else vehicle.stacked_reference_diameter)
        configurations[upper_only] = structural_mass, motors, diameter

    def evaluate(t, values, upper_only, burning=None):
        x, z, vx, vz = values
        ambient = atmosphere_model(launch_altitude + z)
        structural_mass, motors, diameter = configurations[upper_only]
        mass = structural_mass
        thrust = 0.0
        for motor in motors:
            elapsed = t - motor.ignition_time
            mass += motor.motor.dry_mass + float(motor.motor.remaining_propellant_mass(elapsed))
            if burning is None or motor in burning:
                # At an interval's right endpoint, use its left-limit thrust.
                if burning is not None:
                    elapsed = np.clip(elapsed, 0., np.nextafter(motor.motor.burn_time, 0.))
                thrust += float(motor.motor.thrust(elapsed, ambient.pressure))
        if not isfinite(mass) or mass <= 0:
            raise ValueError("Flight mass must be finite and positive.")
        airspeed = float(np.hypot(vx - wind_x, vz))
        state = FlightState(float(t), float(x), float(z), float(vx), float(vz),
                            airspeed, airspeed / ambient.speed_of_sound, float(mass),
                            diameter, upper_only)
        cd = float(drag_model(vehicle, state, ambient))
        if not isfinite(cd) or cd < 0:
            raise ValueError("Drag model must return a finite nonnegative Cd.")
        return state, ambient, thrust, cd

    def apogee_event(t, values):
        return values[3]

    apogee_event.terminal = True
    apogee_event.direction = -1

    # Integrate separately across ignition, burnout, and separation. Holding
    # each interval's discrete configuration fixed prevents RK stages from
    # averaging across a discontinuity at an interval endpoint.
    boundaries = {0., max_time}
    for motor in (vehicle.upper_stage.motor, vehicle.booster_stage.motor):
        boundaries.update((motor.ignition_time, motor.ignition_time + motor.motor.burn_time))
        for knot in getattr(motor.motor, "times", ()):
            boundaries.add(motor.ignition_time + float(knot))
    boundaries.add(vehicle.separation_time)
    boundaries = sorted(t for t in boundaries if 0 <= t <= max_time)
    times, states = [], []
    on_pad = y[1] == 0 and y[2] == 0 and y[3] == 0
    if y[3] == 0 and not on_pad:
        raise FlightError("An airborne initial state must have positive vertical velocity.")
    reached_apogee = False
    for start, end in zip(boundaries, boundaries[1:]):
        midpoint = (start + end) / 2
        upper_only = midpoint >= vehicle.separation_time
        burning = tuple(m for m in configurations[upper_only][1]
                        if m.ignition_time <= midpoint < m.ignition_time + m.motor.burn_time)

        if on_pad:
            def net_lift(t):
                state, _, thrust, _ = evaluate(t, y, upper_only, burning)
                return thrust - state.mass * G

            if not times:
                times.append(start)
                states.append(y.copy())
            if net_lift(start) > 0:
                liftoff = start
            elif net_lift(end) > 0:
                liftoff = brentq(net_lift, start, end, xtol=1e-12)
            else:
                times.append(end)
                states.append(y.copy())
                continue
            if liftoff > times[-1]:
                times.append(liftoff)
                states.append(y.copy())
            start = liftoff
            on_pad = False

        def derivative(t, values):
            state, ambient, thrust, cd = evaluate(t, values, upper_only, burning)
            drag_factor = (0.5 * ambient.density * state.airspeed * cd
                           * pi * state.reference_diameter**2 / 4 / state.mass)
            return (state.vx, state.vz, -drag_factor * (state.vx - wind_x),
                    thrust / state.mass - G - drag_factor * state.vz)

        solution = solve_ivp(derivative, (start, end), y, events=apogee_event,
                             max_step=max_step, rtol=rtol, atol=atol)
        if not solution.success:
            raise FlightError(solution.message)
        skip = 1 if times else 0
        times.extend(solution.t[skip:])
        states.extend(solution.y.T[skip:])
        y = solution.y[:, -1]
        if solution.t_events[0].size:
            reached_apogee = True
            break
    if not reached_apogee:
        raise FlightError("No liftoff or apogee before max_time; check vehicle inputs.")

    profiles = {name: [] for name in Trajectory.__dataclass_fields__}
    for t, values in zip(times, states):
        state, ambient, thrust, cd = evaluate(t, values, t >= vehicle.separation_time)
        row = (t, state.x, state.z, state.vx, state.vz, state.airspeed, state.mach,
               state.mass, thrust, cd, ambient.density, ambient.pressure,
               ambient.temperature, state.upper_stage_only)
        for name, value in zip(profiles, row):
            profiles[name].append(value)
    return Trajectory(**{name: np.asarray(values) for name, values in profiles.items()})


if __name__ == "__main__":
    from example_vehicle import build_example_vehicle

    trajectory = simulate(build_example_vehicle(), wind_x=5.0)
    print("Using placeholder constant Cd = 0.5 (not a design-quality drag prediction).")
    print(f"Apogee: {trajectory.apogee:.1f} m above launch site")
    print(f"Time to apogee: {trajectory.time[-1]:.2f} s")
    print(f"Maximum sampled airspeed: {trajectory.airspeed.max():.1f} m/s")
    print(f"Maximum sampled Mach: {trajectory.mach.max():.3f}")
    print(f"Horizontal displacement at apogee: {trajectory.x[-1]:.1f} m")
    trajectory.save_csv("trajectory.csv")
    print("Saved trajectory.csv")
