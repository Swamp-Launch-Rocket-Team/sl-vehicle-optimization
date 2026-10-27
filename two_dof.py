"""Simple x-z point-mass flight from liftoff to first apogee (SI units).

Zero AoA aligns the body and thrust with air-relative velocity; drag acts backward.
Wind is horizontal. An optional launch rail fixes pitch until release.
No lift, attitude dynamics, or separation kick is modeled.
At zero airspeed in free flight, the body direction defaults to vertical.
The default Cd=0.5 is a PLACEHOLDER, not a geometry-dependent drag model.
"""

from dataclasses import dataclass
from math import atan2, cos, isfinite, pi, sin
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
class LaunchRail:
    """Straight rail in local-horizontal (x, z) coordinates, in SI units.

    Cant is radians from vertical toward +x. The origin is the vehicle's
    initial reference-point position; length is its travel to rail release.
    """

    length: float
    cant: float = 0.0
    origin_x: float = 0.0
    origin_z: float = 0.0

    def __post_init__(self):
        if not all(isfinite(value) for value in (
            self.length, self.cant, self.origin_x, self.origin_z,
        )):
            raise ValueError("Launch rail parameters must be finite.")
        if self.length <= 0:
            raise ValueError("Launch rail length must be positive.")
        if not -pi / 2 < self.cant < pi / 2:
            raise ValueError("Launch rail cant must point above the horizontal.")
        if self.origin_z < 0:
            raise ValueError("Launch rail origin must be above or at launch-site height.")

    @property
    def direction(self) -> tuple[float, float]:
        """Unit direction (x, z) from the rail base toward release."""
        return sin(self.cant), cos(self.cant)

    @property
    def pitch(self) -> float:
        """Body pitch in radians, measured from +x toward +z."""
        return pi / 2 - self.cant

    def distance_along(self, x: float, z: float) -> float:
        """Signed projection of a position along the rail, in meters."""
        dx, dz = self.direction
        return (x - self.origin_x) * dx + (z - self.origin_z) * dz

    def contains(self, x: float, z: float, *, tolerance: float = 1e-8) -> bool:
        """Whether a reference point lies on the rail before release.

        The release endpoint is excluded. Integration should use a terminal
        distance-along event to transition permanently into free flight.
        """
        if not isfinite(tolerance) or tolerance < 0:
            raise ValueError("Rail position tolerance must be finite and nonnegative.")
        if not isfinite(x) or not isfinite(z):
            return False
        dx, dz = self.direction
        cross_track = (x - self.origin_x) * dz - (z - self.origin_z) * dx
        return (abs(cross_track) <= tolerance
                and -tolerance <= self.distance_along(x, z) < self.length)


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
    pitch: np.ndarray  # radians from local +x toward +z
    on_rail: np.ndarray  # false at and after the rail-exit sample

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
    launch_rail: LaunchRail | None = None,
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
    A supplied launch_rail constrains initially on-rail positions and velocities
    until release. At rest at its base, waits for positive acceleration along
    the rail. Without a rail, stationary ground starts wait for thrust > weight.
    Raises if apogee is not reached by max_time. Output times are adaptive and
    include propulsion/configuration events; event samples use post-event data.
    """
    state_vector = np.asarray(initial_conditions, dtype=float)
    if state_vector.shape != (4,) or not np.all(np.isfinite(state_vector)) or state_vector[1] < 0 or state_vector[3] < 0:
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

    def rail_exit_event(t, values):
        return launch_rail.distance_along(values[0], values[1]) - launch_rail.length

    rail_exit_event.terminal = True
    rail_exit_event.direction = 1

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
    on_rail = False
    if launch_rail is not None:
        on_rail = launch_rail.contains(state_vector[0], state_vector[1])
    rail_exit_time = 0.0
    if on_rail:
        rail_exit_time = float("inf")
        rail_dx, rail_dz = launch_rail.direction
        if abs(state_vector[2] * rail_dz - state_vector[3] * rail_dx) > 1e-8:
            raise ValueError("Initial on-rail velocity must be parallel to the rail.")
        on_pad = (abs(launch_rail.distance_along(state_vector[0], state_vector[1])) <= 1e-8
                  and state_vector[2] == 0 and state_vector[3] == 0)
    else:
        on_pad = state_vector[1] == 0 and state_vector[2] == 0 and state_vector[3] == 0
        if launch_rail is not None and on_pad:
            raise ValueError("A stationary launch must start at the supplied rail base.")
    if state_vector[3] == 0 and not on_pad:
        raise FlightError("An airborne initial state must have positive vertical velocity.")
    reached_apogee = False
    for start, end in zip(boundaries, boundaries[1:]):
        midpoint = (start + end) / 2
        upper_only = midpoint >= vehicle.separation_time
        burning = tuple(m for m in configurations[upper_only][1]
                        if m.ignition_time <= midpoint < m.ignition_time + m.motor.burn_time)

        def derivative(t, values):
            state, ambient, thrust, cd = evaluate(t, values, upper_only, burning)
            if on_rail:
                pitch = launch_rail.pitch
            elif state.airspeed > 0:
                pitch = atan2(state.vz, state.vx - wind_x)
            else:
                pitch = pi / 2
            drag = (0.5 * ambient.density * state.airspeed**2 * cd
                    * pi * state.reference_diameter**2 / 4)
            axial_acceleration = (thrust - drag) / state.mass
            if on_rail:
                # Rail reaction cancels the component of gravity normal to it.
                along_acceleration = axial_acceleration - G * rail_dz
                return (state.vx, state.vz, along_acceleration * rail_dx,
                        along_acceleration * rail_dz)
            return (state.vx, state.vz, axial_acceleration * cos(pitch),
                    axial_acceleration * sin(pitch) - G)

        if on_pad:
            def net_lift(t):
                if on_rail:
                    acceleration = derivative(t, state_vector)
                    return acceleration[2] * rail_dx + acceleration[3] * rail_dz
                state, _, thrust, _ = evaluate(t, state_vector, upper_only, burning)
                return thrust - state.mass * G

            if not times:
                times.append(start)
                states.append(state_vector.copy())
            if net_lift(start) > 0:
                liftoff = start
            elif net_lift(end) > 0:
                liftoff = brentq(net_lift, start, end, xtol=1e-12)
            else:
                times.append(end)
                states.append(state_vector.copy())
                continue
            if liftoff > times[-1]:
                times.append(liftoff)
                states.append(state_vector.copy())
            start = liftoff
            on_pad = False

        while start < end:
            events = [apogee_event]
            if on_rail:
                events.append(rail_exit_event)
            solution = solve_ivp(derivative, (start, end), state_vector, events=events,
                                 max_step=max_step, rtol=rtol, atol=atol)
            if not solution.success:
                raise FlightError(solution.message)
            skip = 1 if times else 0
            times.extend(solution.t[skip:])
            states.extend(solution.y.T[skip:])
            state_vector = solution.y[:, -1]
            if solution.t_events[0].size:
                if on_rail:
                    raise FlightError("Vehicle stopped ascending before leaving the launch rail.")
                reached_apogee = True
                break
            if on_rail and solution.t_events[1].size:
                on_rail = False
                start = float(solution.t[-1])
                rail_exit_time = start
            else:
                break
        if reached_apogee:
            break
    if not reached_apogee:
        raise FlightError("No liftoff or apogee before max_time; check vehicle inputs.")

    profiles = {name: [] for name in Trajectory.__dataclass_fields__}
    for t, values in zip(times, states):
        state, ambient, thrust, cd = evaluate(t, values, t >= vehicle.separation_time)
        sample_on_rail = t < rail_exit_time
        if sample_on_rail:
            pitch = launch_rail.pitch
        elif state.airspeed > 0:
            pitch = atan2(state.vz, state.vx - wind_x)
        else:
            pitch = pi / 2
        row = (t, state.x, state.z, state.vx, state.vz, state.airspeed, state.mach,
               state.mass, thrust, cd, ambient.density, ambient.pressure,
               ambient.temperature, state.upper_stage_only, pitch, sample_on_rail)
        for name, value in zip(profiles, row):
            profiles[name].append(value)
    return Trajectory(**{name: np.asarray(values) for name, values in profiles.items()})


if __name__ == "__main__":
    from example_vehicle import build_example_vehicle

    trajectory = simulate(
        build_example_vehicle(),
        launch_rail=LaunchRail(length=12.0 * 0.3048, cant=5.0 * pi / 180.0),
        wind_x=5.0,
    )
    print("Using placeholder constant Cd = 0.5 (not a design-quality drag prediction).")
    print(f"Apogee: {trajectory.apogee:.1f} m above launch site")
    print(f"Time to apogee: {trajectory.time[-1]:.2f} s")
    print(f"Maximum sampled airspeed: {trajectory.airspeed.max():.1f} m/s")
    print(f"Maximum sampled Mach: {trajectory.mach.max():.3f}")
    print(f"Horizontal displacement at apogee: {trajectory.x[-1]:.1f} m")
    trajectory.save_csv("trajectory.csv")
    print("Saved trajectory.csv")
