"""Vehicle design variables and construction from fixed hardware inputs."""

from dataclasses import asdict, dataclass
from math import acos, pi, radians, sin, sqrt, tan

import numpy as np

from geometry import AxisymmetricSection, FinSet
from kraken import Kraken
from mass_model import (FinMassComponent, FixedMassComponent, GeometryOnlyComponent,
                        MotorMassComponent, ShellMassComponent)
from vehicle import Vehicle, VehicleStage

INCH = 0.0254
POUND = 0.45359237
OZ_PER_IN3 = 0.028349523125 / INCH**3

VARIABLE_BOUNDS = {
    "nose_length_in": (12.0, 24.0),
    "stage1_airframe_length_in": (18.0, 24.0),
    "stage2_airframe_length_in": (18.0, 24.0),
    "coast_time_s": (6.0, 12.0),
    "stage1_root_chord_in": (14.0, 18.0),
    "stage2_root_chord_in": (9.0, 12.0),
    "stage1_sweep_deg": (64.0, 69.0),
    "stage2_sweep_deg": (63.0, 68.0),
    "stage1_span_in": (6.0, 7.0),
    "stage2_span_in": (4.0, 5.0),
    "stage1_propellant_mass_lb": (14.0, 16.0),
    "stage2_propellant_mass_lb": (14.0, 16.0),
}


class InvalidCandidate(ValueError):
    """An expected design failure, rather than a programming error."""


@dataclass(frozen=True)
class Design:
    nose_length_in: float = 18.0
    stage1_airframe_length_in: float = 21.0
    stage2_airframe_length_in: float = 21.0
    coast_time_s: float = 9.0
    stage1_root_chord_in: float = 16.0
    stage2_root_chord_in: float = 10.5
    stage1_sweep_deg: float = 66.5
    stage2_sweep_deg: float = 65.5
    stage1_span_in: float = 6.5
    stage2_span_in: float = 4.5
    stage1_propellant_mass_lb: float = 15.0
    stage2_propellant_mass_lb: float = 15.0

    def validate(self):
        for name, (lower, upper) in VARIABLE_BOUNDS.items():
            value = getattr(self, name)
            if not np.isfinite(value) or not lower <= value <= upper:
                raise InvalidCandidate(f"{name} must be between {lower} and {upper}.")

    def normalized(self):
        return np.array([(getattr(self, name) - lo) / (hi - lo)
                         for name, (lo, hi) in VARIABLE_BOUNDS.items()])

    @classmethod
    def from_normalized(cls, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(VARIABLE_BOUNDS),) or not np.isfinite(values).all():
            raise InvalidCandidate(
                f"Expected {len(VARIABLE_BOUNDS)} finite normalized design variables."
            )
        if np.any((values < 0) | (values > 1)):
            raise InvalidCandidate("Normalized design variables must lie in [0, 1].")
        return cls(**{name: float(lo + value * (hi - lo))
                      for (name, (lo, hi)), value in zip(VARIABLE_BOUNDS.items(), values)})


@dataclass(frozen=True)
class VehicleConfig:
    outer_diameter_in: float = 4.02
    inner_diameter_in: float = 3.9
    fiberglass_density_oz_in3: float = 1.07
    carbon_density_oz_in3: float = 1.03
    stage1_fin_density_oz_in3: float = 1.03
    stage2_fin_density_oz_in3: float = 1.56
    fin_thickness_in: float = 0.125
    fin_count: int = 4
    stage1_taper_ratio: float = 0.25
    stage2_taper_ratio: float = 0.20
    nose_internal_mass_lb: float = 3.5
    stage1_internal_mass_lb: float = 5.615
    stage2_internal_mass_lb: float = 1.7
    booster_ignition_time_s: float = 0.0
    separation_time_s: float = 11.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Fixed vehicle inputs must be finite.")
        if not 0 < self.inner_diameter_in < self.outer_diameter_in:
            raise ValueError("Require 0 < inner diameter < outer diameter.")
        if type(self.fin_count) is not int or self.fin_count < 3:
            raise ValueError("Need at least three equally spaced fins.")
        for name, value in asdict(self).items():
            if name in ("booster_ignition_time_s", "separation_time_s"):
                if value < 0:
                    raise ValueError(f"{name} cannot be negative.")
            elif value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.stage1_taper_ratio > 1 or self.stage2_taper_ratio > 1:
            raise ValueError("Taper ratios must not exceed one.")


class CandidateBuilder:
    def __init__(self, config=None):
        self.config = config or VehicleConfig()
        burnout = self.config.booster_ignition_time_s + Kraken.burn_time
        if self.config.separation_time_s < burnout:
            raise ValueError("Separation must occur at or after booster burnout.")

    def build(self, design: Design) -> Vehicle:
        design.validate()
        c = self.config
        stage1_motor = Kraken(propellant_mass=design.stage1_propellant_mass_lb * POUND)
        stage2_motor = Kraken(propellant_mass=design.stage2_propellant_mass_lb * POUND)
        radius = c.outer_diameter_in * INCH / 2
        wall = (c.outer_diameter_in - c.inner_diameter_in) * INCH / 2
        nose_length = design.nose_length_in * INCH
        upper_length = design.stage2_airframe_length_in * INCH
        booster_length = design.stage1_airframe_length_in * INCH

        def shell(length, profile, density):
            return ShellMassComponent(section=AxisymmetricSection(
                length=length, radius_at_axial_distance=profile), wall_thickness=wall,
                material_density=density * OZ_PER_IN3, forward_axial_position=0.0)

        def nose_profile(x):
            theta = acos(np.clip(1 - 2*x/nose_length, -1.0, 1.0))
            return radius * sqrt(max(0.0, (theta - sin(2*theta)/2) / pi))

        def casing(length):
            return GeometryOnlyComponent(section=AxisymmetricSection(
                length=length, radius_at_axial_distance=lambda x: radius))

        upper_motor_start = nose_length + upper_length
        upper_aft = upper_motor_start + stage2_motor.motor_length
        booster_motor_start = upper_aft + booster_length
        booster_aft = booster_motor_start + stage1_motor.motor_length

        def fins(root_in, span_in, sweep_deg, taper, aft, density):
            root, span = root_in * INCH, span_in * INCH
            return FinMassComponent(fin_set=FinSet(
                number_of_fins=c.fin_count, fin_root_chord=root,
                fin_tip_chord=taper*root, fin_span=span,
                fin_leading_edge_sweep_distance=span*tan(radians(sweep_deg)),
                fin_leading_edge_axial_position=aft-root,
                body_diameter_at_fin_attachment=2*radius),
                fin_thickness=c.fin_thickness_in*INCH,
                material_density=density*OZ_PER_IN3)

        upper = VehicleStage(
            body_sections=(shell(upper_length, lambda x: radius, c.carbon_density_oz_in3),
                           casing(stage2_motor.motor_length)),
            fin_sets=(fins(design.stage2_root_chord_in, design.stage2_span_in,
                           design.stage2_sweep_deg, c.stage2_taper_ratio, upper_aft,
                           c.stage2_fin_density_oz_in3),),
            motor=MotorMassComponent(motor=stage2_motor,
                forward_axial_position=upper_motor_start,
                ignition_time=c.separation_time_s + design.coast_time_s),
            fixed_components=(
                FixedMassComponent(mass=c.nose_internal_mass_lb*POUND,
                                   center_of_mass=nose_length/2),
                FixedMassComponent(mass=c.stage2_internal_mass_lb*POUND,
                                   center_of_mass=nose_length+upper_length/2)))
        booster = VehicleStage(
            body_sections=(shell(booster_length, lambda x: radius, c.fiberglass_density_oz_in3),
                           casing(stage1_motor.motor_length)),
            fin_sets=(fins(design.stage1_root_chord_in, design.stage1_span_in,
                           design.stage1_sweep_deg, c.stage1_taper_ratio, booster_aft,
                           c.stage1_fin_density_oz_in3),),
            motor=MotorMassComponent(motor=stage1_motor,
                forward_axial_position=booster_motor_start,
                ignition_time=c.booster_ignition_time_s),
            fixed_components=(FixedMassComponent(mass=c.stage1_internal_mass_lb*POUND,
                                   center_of_mass=upper_aft+booster_length/2),))
        return Vehicle(nose_cone=shell(nose_length, nose_profile, c.fiberglass_density_oz_in3),
                       upper_stage=upper, booster_stage=booster,
                       separation_time=c.separation_time_s,
                       stacked_reference_diameter=2*radius,
                       upper_stage_reference_diameter=2*radius)
