"""Geometry-based component masses in kilograms and axial positions in meters.

All positions use the vehicle nose tip as their origin. Shell thickness is
measured radially; fin thickness is measured normal to the fin surface.
"""

from math import isfinite, pi, sqrt

import numpy as np

from geometry import AxisymmetricSection, FinSet
from kraken import Kraken
from thrust_curve import ThrustCurve


def _require_positive(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _require_nonnegative(value: float, name: str) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")


class ShellMassComponent:
    """Uniform-density material between an external profile and its inner radius.

    Inner radius is max(external radius - wall thickness, 0), leaving a solid
    tip where a pointed profile is narrower than the wall thickness. End caps,
    bulkheads, and internal fittings must be added as separate components.
    """

    def __init__(
        self,
        *,
        section: AxisymmetricSection,
        wall_thickness: float,
        material_density: float,
        forward_axial_position: float,
    ) -> None:
        _require_positive(wall_thickness, "Wall thickness")
        _require_positive(material_density, "Material density")
        _require_nonnegative(forward_axial_position, "Forward axial position")
        self.section = section
        self.wall_thickness = wall_thickness
        self.material_density = material_density
        self.forward_axial_position = forward_axial_position

    def _volume_and_axial_volume_moment(self) -> tuple[float, float]:
        """Integrate material area and its local first moment with Simpson's rule."""
        interval_count = 256
        interval_length = self.section.length / interval_count
        weighted_area_sum = 0.0
        weighted_area_moment_sum = 0.0
        for sample_index in range(interval_count + 1):
            axial_distance = (
                self.section.length if sample_index == interval_count
                else sample_index * interval_length
            )
            external_area = self.section.cross_sectional_area(axial_distance)
            external_radius = sqrt(external_area / pi)
            inner_radius = max(external_radius - self.wall_thickness, 0.0)
            material_area = external_area - pi * inner_radius**2
            weight = 1 if sample_index in (0, interval_count) else (
                4 if sample_index % 2 else 2
            )
            weighted_area_sum += weight * material_area
            weighted_area_moment_sum += weight * material_area * axial_distance
        return (
            weighted_area_sum * interval_length / 3,
            weighted_area_moment_sum * interval_length / 3,
        )

    @property
    def mass(self) -> float:
        material_volume, _ = self._volume_and_axial_volume_moment()
        return self.material_density * material_volume

    @property
    def center_of_mass(self) -> float:
        material_volume, axial_volume_moment = self._volume_and_axial_volume_moment()
        if material_volume <= 0:
            raise ValueError("Shell center of mass requires positive material volume.")
        return self.forward_axial_position + axial_volume_moment / material_volume


class GeometryOnlyComponent:
    """Exposed motor casing geometry; its mass is already in the motor."""

    def __init__(self, *, section: AxisymmetricSection):
        self.section = section
        self.forward_axial_position = 0.0

    @property
    def mass(self):
        return 0.0

    @property
    def center_of_mass(self):
        return self.forward_axial_position + self.section.length / 2


class FinMassComponent:
    """All fins in one set, with uniform thickness and material density."""

    def __init__(
        self, *, fin_set: FinSet, fin_thickness: float, material_density: float,
    ) -> None:
        _require_positive(fin_thickness, "Fin thickness")
        _require_positive(material_density, "Material density")
        self.fin_set = fin_set
        self.fin_thickness = fin_thickness
        self.material_density = material_density

    @property
    def mass(self) -> float:
        return (
            self.fin_set.number_of_fins * self.fin_set.fin_planform_area
            * self.fin_thickness * self.material_density
        )

    @property
    def center_of_mass(self) -> float:
        root_chord = self.fin_set.fin_root_chord
        tip_chord = self.fin_set.fin_tip_chord
        chord_sum = root_chord + tip_chord
        return (
            self.fin_set.fin_leading_edge_axial_position
            + self.fin_set.fin_leading_edge_sweep_distance
            * (root_chord + 2 * tip_chord) / (3 * chord_sum)
            + (root_chord**2 + root_chord * tip_chord + tip_chord**2)
            / (3 * chord_sum)
        )


class FixedMassComponent:
    """A supplied mass and axial center of mass, for equipment or other hardware."""

    def __init__(self, *, mass: float, center_of_mass: float) -> None:
        _require_nonnegative(mass, "Mass")
        _require_nonnegative(center_of_mass, "Center of mass")
        self.mass = mass
        self.center_of_mass = center_of_mass


def _as_time_vector(time_vector) -> np.ndarray:
    flight_times = np.asarray(time_vector, dtype=float)
    if flight_times.ndim != 1 or not np.all(np.isfinite(flight_times)):
        raise ValueError("Flight times must be a one-dimensional vector of finite values.")
    return flight_times


class MotorMassComponent:
    """Motor hardware and propellant at a fixed motor midpoint.

    Time and ignition time are in seconds on the same flight clock. Burnout
    leaves dry hardware attached; stage separation is handled by the vehicle.
    """

    def __init__(
        self, *, motor: Kraken | ThrustCurve, forward_axial_position: float, ignition_time: float,
    ) -> None:
        _require_nonnegative(motor.p_mass, "Initial propellant mass")
        _require_positive(motor.dry_mass, "Motor dry mass")
        _require_positive(motor.motor_length, "Motor length")
        _require_positive(motor.burn_time, "Motor burn time")
        _require_nonnegative(forward_axial_position, "Motor forward axial position")
        _require_nonnegative(ignition_time, "Ignition time")
        self.motor = motor
        self.forward_axial_position = forward_axial_position
        self.ignition_time = ignition_time

    @property
    def center_of_mass(self) -> float:
        return self.forward_axial_position + self.motor.motor_length / 2

    def remaining_propellant_mass(self, time_vector) -> np.ndarray:
        flight_times = _as_time_vector(time_vector)
        return self.motor.remaining_propellant_mass(flight_times - self.ignition_time)

    def total_mass(self, time_vector) -> np.ndarray:
        return self.motor.dry_mass + self.remaining_propellant_mass(time_vector)


class MassModel:
    """Combine structural components and motors, evaluating geometry on demand.

    Supply a time vector when motors are present; both outputs then have the
    same shape as that vector. Static-only models also support scalar results
    without a time argument. Hardware remains attached after motor burnout.
    """

    def __init__(
        self,
        *,
        components: tuple[ShellMassComponent | GeometryOnlyComponent | FinMassComponent | FixedMassComponent, ...],
        motors: tuple[MotorMassComponent, ...] = (),
    ) -> None:
        self.components = tuple(components)
        self.motors = tuple(motors)

    def _mass_and_axial_mass_moment(self, time_vector):
        if time_vector is None and self.motors:
            raise ValueError("Supply a time vector when the mass model contains motors.")
        total_mass = 0.0
        axial_mass_moment = 0.0
        for component in self.components:
            component_mass = component.mass
            _require_nonnegative(component_mass, "Component mass")
            total_mass += component_mass
            if component_mass > 0:
                axial_mass_moment += component_mass * component.center_of_mass
        if time_vector is not None:
            flight_times = _as_time_vector(time_vector)
            total_mass = np.full_like(flight_times, total_mass)
            axial_mass_moment = np.full_like(flight_times, axial_mass_moment)
            for motor in self.motors:
                motor_mass = motor.total_mass(flight_times)
                total_mass += motor_mass
                axial_mass_moment += motor_mass * motor.center_of_mass
        if not np.all(np.isfinite(total_mass) & (total_mass > 0)):
            raise ValueError("Total mass must be finite and positive.")
        return total_mass, axial_mass_moment

    def total_mass(self, time_vector=None) -> float | np.ndarray:
        total_mass, _ = self._mass_and_axial_mass_moment(time_vector)
        return total_mass

    def center_of_mass(self, time_vector=None) -> float | np.ndarray:
        total_mass, axial_mass_moment = self._mass_and_axial_mass_moment(time_vector)
        return axial_mass_moment / total_mass
