"""Shared axisymmetric-section and fin-set geometry.

Lengths are in meters, areas in square meters, volumes in cubic meters,
and angles in radians. Axial positions are measured aft from the nose tip.
"""

from math import atan2, isfinite, pi
from typing import Callable


class AxisymmetricSection:
    """An external profile, with axial distance measured from its forward end.
    """

    def __init__(
        self,
        *,
        length: float,
        radius_at_axial_distance: Callable[[float], float],
    ) -> None:
        self.length = length
        self.radius_at_axial_distance = radius_at_axial_distance
        if not isfinite(self.length) or self.length <= 0:
            raise ValueError("Section length must be finite and positive.")

    def cross_sectional_area(self, axial_distance: float) -> float:
        """Return external cross-sectional area at a local axial distance."""
        if not 0 <= axial_distance <= self.length:
            raise ValueError("Axial distance must lie within the section.")
        radius = self.radius_at_axial_distance(axial_distance)
        if not isfinite(radius) or radius < 0:
            raise ValueError("External radius must be finite and nonnegative.")
        return pi * radius**2

    @property
    def forward_cross_sectional_area(self) -> float:
        return self.cross_sectional_area(0.0)

    @property
    def aft_cross_sectional_area(self) -> float:
        return self.cross_sectional_area(self.length)

    @property
    def external_volume(self) -> float:
        """Integrate the profile using composite Simpson's rule (256 intervals).

        This is a numerical approximation for smooth profiles. Represent
        changes in profile as separate sections.
        """
        interval_count = 256
        interval_length = self.length / interval_count
        weighted_area_sum = (
            self.forward_cross_sectional_area + self.aft_cross_sectional_area
        )
        for sample_index in range(1, interval_count):
            weight = 4 if sample_index % 2 else 2
            weighted_area_sum += weight * self.cross_sectional_area(
                sample_index * interval_length
            )
        return weighted_area_sum * interval_length / 3


class FinSet:
    """Geometry of one fin set, with axial positions measured from the nose tip.

    Fin span is exposed span from the body surface. Sweep distance is positive
    aft from root to tip. Each set has its own body attachment diameter.
    """

    def __init__(
        self,
        *,
        number_of_fins: int,
        fin_root_chord: float,
        fin_tip_chord: float,
        fin_span: float,
        fin_leading_edge_sweep_distance: float,
        fin_leading_edge_axial_position: float,
        body_diameter_at_fin_attachment: float,
    ) -> None:
        self.number_of_fins = number_of_fins
        self.fin_root_chord = fin_root_chord
        self.fin_tip_chord = fin_tip_chord
        self.fin_span = fin_span
        self.fin_leading_edge_sweep_distance = fin_leading_edge_sweep_distance
        self.fin_leading_edge_axial_position = fin_leading_edge_axial_position
        self.body_diameter_at_fin_attachment = body_diameter_at_fin_attachment

        if type(self.number_of_fins) is not int or self.number_of_fins <= 0:
            raise ValueError("Number of fins must be a positive integer.")
        for name in (
            "fin_root_chord", "fin_span", "body_diameter_at_fin_attachment",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("fin_tip_chord", "fin_leading_edge_axial_position"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not isfinite(self.fin_leading_edge_sweep_distance):
            raise ValueError("Fin leading-edge sweep distance must be finite.")

    @property
    def fin_planform_area(self) -> float:
        """Planform area of one exposed fin."""
        return (self.fin_root_chord + self.fin_tip_chord) * self.fin_span / 2

    @property
    def fin_aspect_ratio(self) -> float:
        """Barrowman aspect ratio, using the equivalent mirrored fin pair."""
        return 2 * self.fin_span**2 / self.fin_planform_area

    @property
    def fin_midchord_sweep_angle(self) -> float:
        midchord_sweep_distance = (
            self.fin_leading_edge_sweep_distance
            + (self.fin_tip_chord - self.fin_root_chord) / 2
        )
        return atan2(midchord_sweep_distance, self.fin_span)

    @property
    def fin_tip_to_body_radius_ratio(self) -> float:
        """Barrowman's tau: fin tip radius divided by body radius."""
        body_radius = self.body_diameter_at_fin_attachment / 2
        return (body_radius + self.fin_span) / body_radius



    @property
    def fin_center_of_pressure(self) -> float:
        """Equation 3-30: Subsonic tail center of pressure."""
        return (
            self.fin_leading_edge_axial_position
            + self.fin_leading_edge_sweep_distance / 3
            * (
                (self.fin_root_chord + 2 * self.fin_tip_chord)
                / (self.fin_root_chord + self.fin_tip_chord)
            )
            + 1 / 6
            * (
                self.fin_root_chord
                + self.fin_tip_chord
                - self.fin_root_chord * self.fin_tip_chord
                / (self.fin_root_chord + self.fin_tip_chord)
            )
        )
