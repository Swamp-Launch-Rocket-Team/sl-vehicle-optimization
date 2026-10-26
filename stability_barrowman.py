"""Barrowman subsonic stability calculations using shared vehicle geometry."""

from math import isfinite, pi

import numpy as np

from geometry import AxisymmetricSection, FinSet


class StabilityBarrowman:
    """Own one vehicle configuration, including all attached fin sets.

    Body sections are consecutive and ordered aft from the nose base. All fin
    positions share the nose-tip origin and all forces share one reference area.
    Set flight Mach numbers with ``set_mach_number`` before evaluation.
    Each fin set uses the combined OpenRocket body-interference approximation;
    aerodynamic interactions between fin sets are not modeled.
    Use separate configurations for the stacked vehicle and separated stages.
    """

    def __init__(
        self,
        *,
        reference_diameter: float,
        nose_cone: AxisymmetricSection,
        body_sections: tuple[AxisymmetricSection, ...],
        fin_sets: tuple[FinSet, ...],
    ) -> None:
        self.reference_diameter = reference_diameter
        self.nose_cone = nose_cone
        self.body_sections = tuple(body_sections)
        self.fin_sets = tuple(fin_sets)
        if not isfinite(self.reference_diameter) or self.reference_diameter <= 0:
            raise ValueError("Reference diameter must be finite and positive.")

    def set_mach_number(self, mach_number_vector) -> None:
        """Store a Mach vector and compute beta; stability remains subsonic."""
        mach_number = np.array(mach_number_vector, dtype=float, copy=True)
        if mach_number.ndim != 1:
            raise ValueError("Mach numbers must be a one-dimensional vector.")
        if not np.all(np.isfinite(mach_number) & (mach_number >= 0)):
            raise ValueError("Mach numbers must be finite and nonnegative.")
        compressibility_factor = np.sqrt(np.where(
            mach_number < 1,
            1 - mach_number**2,
            mach_number**2 - 1,
        ))
        self.mach_number = mach_number
        self.compressibility_factor = compressibility_factor

    @property
    def reference_cross_sectional_area(self) -> float:
        return pi * (self.reference_diameter / 2)**2

    @property
    def nose_cone_length(self) -> float:
        return self.nose_cone.length

    @property
    def nose_cone_base_area(self) -> float:
        return self.nose_cone.aft_cross_sectional_area

    @property
    def nose_cone_base_diameter(self) -> float:
        return 2 * (self.nose_cone_base_area / pi)**0.5

    @property
    def nose_tip_cross_sectional_area(self) -> float:
        return self.nose_cone.forward_cross_sectional_area

    @property
    def body_cross_sectional_area_at_nose_base(self) -> float:
        return self.nose_cone_base_area

    @property
    def nose_cone_external_volume(self) -> float:
        return self.nose_cone.external_volume

    @property
    def body_base_cross_sectional_area(self) -> float:
        aft_section = self.body_sections[-1] if self.body_sections else self.nose_cone
        return aft_section.aft_cross_sectional_area

    @property
    def body_length(self) -> float:
        """Total nose, body, and transition length, excluding fin overhang."""
        return self.nose_cone_length + sum(
            section.length for section in self.body_sections
        )

    @property
    def body_external_volume(self) -> float:
        """Total enclosed external volume, including the nose cone."""
        return self.nose_cone_external_volume + sum(
            section.external_volume for section in self.body_sections
        )


    def fin_normal_force_coefficient_derivative(self, fin_set: FinSet) -> np.ndarray:
        """Equation 3-12 with the combined body-interference correction for one set."""
        if not hasattr(self, "mach_number"):
            raise ValueError("Set the Mach number before evaluating stability.")
        if np.any(self.mach_number >= 1):
            raise ValueError("Equation 3-12 requires subsonic Mach numbers.")

        return (
            fin_set.number_of_fins
            * pi
            * fin_set.fin_aspect_ratio
            * (fin_set.fin_planform_area / self.reference_cross_sectional_area)
            / (
                2
                + np.sqrt(
                    4
                    + (
                        self.compressibility_factor
                        * fin_set.fin_aspect_ratio
                        / np.cos(fin_set.fin_midchord_sweep_angle)
                    ) ** 2
                )
            )
        ) * (1 + 1 / fin_set.fin_tip_to_body_radius_ratio)

    @property
    def body_normal_force_coefficient_derivative(self) -> float:
        """Equations 3-65 and 3-66: Whole-body normal-force derivative."""
        return (
            2 * (
                self.body_base_cross_sectional_area
                - self.nose_tip_cross_sectional_area
            )
            / self.reference_cross_sectional_area
        )


    @property
    def body_center_of_pressure(self) -> float:
        """Equation 3-89: Subsonic body center of pressure."""
        return (
            self.body_length
            - self.body_external_volume
            / self.body_base_cross_sectional_area
        )


    def normal_force_coefficient_derivative(self) -> np.ndarray:
        """Equation 3-106 with combined interference for each fin set."""
        if not hasattr(self, "mach_number"):
            raise ValueError("Set the Mach number before evaluating stability.")
        if np.any(self.mach_number >= 1):
            raise ValueError("Subsonic stability requires Mach numbers below one.")
        total_derivative = np.full_like(
            self.mach_number, self.body_normal_force_coefficient_derivative,
        )
        for fin_set in self.fin_sets:
            total_derivative += self.fin_normal_force_coefficient_derivative(fin_set)
        return total_derivative

    def center_of_pressure(self) -> np.ndarray:
        """Equation 3-107 with combined interference for each fin set."""
        total_derivative = self.normal_force_coefficient_derivative()
        normal_force_moment_derivative = np.full_like(
            self.mach_number,
            self.body_normal_force_coefficient_derivative * self.body_center_of_pressure,
        )
        for fin_set in self.fin_sets:
            normal_force_moment_derivative += (
                self.fin_normal_force_coefficient_derivative(fin_set)
                * fin_set.fin_center_of_pressure
            )
        return normal_force_moment_derivative / total_derivative
