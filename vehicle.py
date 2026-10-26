"""Shared component ownership and flight configurations for a two-stage vehicle."""

from math import isfinite

import numpy as np

from mass_model import (
    FinMassComponent, FixedMassComponent, MassModel, MotorMassComponent,
    ShellMassComponent,
)
from stability_barrowman import StabilityBarrowman


class VehicleStage:
    """Stage components, with shell sections ordered from forward to aft.

    Include the transition as the first booster body section. Motor and fin
    positions are measured from the vehicle nose tip, including after separation.
    """

    def __init__(
        self,
        *,
        body_sections: tuple[ShellMassComponent, ...],
        fin_sets: tuple[FinMassComponent, ...],
        motor: MotorMassComponent,
        fixed_components: tuple[FixedMassComponent, ...] = (),
    ) -> None:
        self.body_sections = tuple(body_sections)
        self.fin_sets = tuple(fin_sets)
        self.motor = motor
        self.fixed_components = tuple(fixed_components)


class Vehicle:
    """Feed shared geometry to mass and stability models for each flight sample.

    The booster and its transition are removed at and after separation_time.
    Shell positions are derived from their ordered lengths on every evaluation.
    Fin and motor positions remain explicit inputs measured from the nose tip.
    Results describe the continuing upper vehicle, not the detached booster.
    """

    def __init__(
        self,
        *,
        nose_cone: ShellMassComponent,
        upper_stage: VehicleStage,
        booster_stage: VehicleStage,
        separation_time: float,
        stacked_reference_diameter: float,
        upper_stage_reference_diameter: float,
    ) -> None:
        if not isfinite(separation_time) or separation_time < 0:
            raise ValueError("Separation time must be finite and nonnegative.")
        for diameter in (stacked_reference_diameter, upper_stage_reference_diameter):
            if not isfinite(diameter) or diameter <= 0:
                raise ValueError("Reference diameters must be finite and positive.")
        self.nose_cone = nose_cone
        self.upper_stage = upper_stage
        self.booster_stage = booster_stage
        self.separation_time = separation_time
        self.stacked_reference_diameter = stacked_reference_diameter
        self.upper_stage_reference_diameter = upper_stage_reference_diameter

    def _build_models(self, include_booster: bool):
        # Rebuild the component lists so geometry edits and replacements are
        # reflected by both models without maintaining independent geometry.
        stages = (self.upper_stage, self.booster_stage) if include_booster else (
            self.upper_stage,
        )
        self.nose_cone.forward_axial_position = 0.0
        forward_axial_position = self.nose_cone.section.length
        shell_components = []
        fin_components = []
        fixed_components = []
        motors = []
        for stage in stages:
            for shell in stage.body_sections:
                shell.forward_axial_position = forward_axial_position
                forward_axial_position += shell.section.length
                shell_components.append(shell)
            fin_components.extend(stage.fin_sets)
            fixed_components.extend(stage.fixed_components)
            motors.append(stage.motor)
        reference_diameter = (
            self.stacked_reference_diameter if include_booster
            else self.upper_stage_reference_diameter
        )
        mass_model = MassModel(
            components=(self.nose_cone, *shell_components, *fin_components, *fixed_components),
            motors=tuple(motors),
        )
        stability_model = StabilityBarrowman(
            reference_diameter=reference_diameter,
            nose_cone=self.nose_cone.section,
            body_sections=tuple(shell.section for shell in shell_components),
            fin_sets=tuple(component.fin_set for component in fin_components),
        )
        return mass_model, stability_model

    def _flight_times(self, time_vector) -> np.ndarray:
        flight_times = np.asarray(time_vector, dtype=float)
        if flight_times.ndim != 1 or not np.all(np.isfinite(flight_times)):
            raise ValueError("Flight times must be a one-dimensional vector of finite values.")
        return flight_times

    def _mass_result(self, time_vector, calculation: str) -> np.ndarray:
        flight_times = self._flight_times(time_vector)
        result = np.empty_like(flight_times)
        before_separation = flight_times < self.separation_time
        for include_booster, mask in ((True, before_separation), (False, ~before_separation)):
            if np.any(mask):
                mass_model, _ = self._build_models(include_booster)
                result[mask] = getattr(mass_model, calculation)(flight_times[mask])
        return result

    def total_mass(self, time_vector) -> np.ndarray:
        return self._mass_result(time_vector, "total_mass")

    def center_of_mass(self, time_vector) -> np.ndarray:
        return self._mass_result(time_vector, "center_of_mass")

    def _stability_result(self, time_vector, mach_number_vector, calculation: str) -> np.ndarray:
        flight_times = self._flight_times(time_vector)
        mach_numbers = np.asarray(mach_number_vector, dtype=float)
        if mach_numbers.shape != flight_times.shape:
            raise ValueError("Mach numbers and flight times must have matching vector shapes.")
        if not np.all(np.isfinite(mach_numbers) & (mach_numbers >= 0) & (mach_numbers < 1)):
            raise ValueError("Stability requires finite Mach numbers satisfying 0 <= Mach < 1.")
        result = np.empty_like(flight_times)
        before_separation = flight_times < self.separation_time
        for include_booster, mask in ((True, before_separation), (False, ~before_separation)):
            if np.any(mask):
                _, stability_model = self._build_models(include_booster)
                stability_model.set_mach_number(mach_numbers[mask])
                result[mask] = getattr(stability_model, calculation)()
        return result

    def normal_force_coefficient_derivative(self, time_vector, mach_number_vector) -> np.ndarray:
        return self._stability_result(
            time_vector, mach_number_vector, "normal_force_coefficient_derivative",
        )

    def center_of_pressure(self, time_vector, mach_number_vector) -> np.ndarray:
        return self._stability_result(time_vector, mach_number_vector, "center_of_pressure")

    def stability_margin(self, time_vector, mach_number_vector) -> np.ndarray:
        """Return (CP - CG) / configuration reference diameter, in calibers."""
        flight_times = self._flight_times(time_vector)
        center_of_pressure = self.center_of_pressure(flight_times, mach_number_vector)
        center_of_mass = self.center_of_mass(flight_times)
        reference_diameter = np.where(
            flight_times < self.separation_time,
            self.stacked_reference_diameter,
            self.upper_stage_reference_diameter,
        )
        return (center_of_pressure - center_of_mass) / reference_diameter
