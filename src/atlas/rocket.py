"""The design vector and the vehicle it implies.

FREE VARIABLES (7)
    m_prop_1, m_prop_2        motor propellant mass per stage
    t_ignite_s                coast from separation to stage 2 ignition
    extra_length_1_m, _2_m    body length above the motor + 30 in minimum
    structure_mass_1_kg, _2   dry structure mass per stage, excluding the motor

Fin geometry is NOT free -- it is fixed at the baseline in params.py.

Structure mass is free rather than derived, which makes this a trade study: the
optimum always sits at the mass lower bound, so the useful output is the
landscape -- what apogee a given mass buys you -- not the single best point.

Stations are measured from the nose tip, positive aft. Stage 2 occupies the
forward part of the vehicle, stage 1 the aft.
"""

from __future__ import annotations

from dataclasses import dataclass

from .motor import Motor, build_motor
from .params import Config, Fins

VARIABLES = (
    "m_prop_1",
    "m_prop_2",
    "t_ignite_s",
    "extra_length_1_m",
    "extra_length_2_m",
    "structure_mass_1_kg",
    "structure_mass_2_kg",
)


@dataclass(frozen=True)
class Design:
    """One point in the design space."""

    m_prop_1: float
    m_prop_2: float
    t_ignite_s: float
    extra_length_1_m: float = 0.0
    extra_length_2_m: float = 0.0
    structure_mass_1_kg: float = 11.0
    structure_mass_2_kg: float = 10.0

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in VARIABLES}


@dataclass(frozen=True)
class Stage:
    name: str
    motor: Motor
    fins: Fins
    body_length_m: float
    structure_mass_kg: float

    @property
    def dry_mass_kg(self) -> float:
        return self.structure_mass_kg + self.motor.dry_mass_kg

    @property
    def wet_mass_kg(self) -> float:
        return self.structure_mass_kg + self.motor.wet_mass_kg

    def mass_at(self, t_since_ignition_s: float | None) -> float:
        """Stage mass. None means the motor has not been lit yet."""
        if t_since_ignition_s is None:
            return self.wet_mass_kg
        return self.structure_mass_kg + self.motor.mass(t_since_ignition_s)


@dataclass(frozen=True)
class Rocket:
    design: Design
    config: Config
    stage_1: Stage
    stage_2: Stage

    @property
    def body_diameter_m(self) -> float:
        return self.config.body_diameter_m

    @property
    def reference_area_m2(self) -> float:
        import math

        return math.pi * (self.body_diameter_m / 2.0) ** 2

    @property
    def total_length_m(self) -> float:
        return self.stage_1.body_length_m + self.stage_2.body_length_m

    @property
    def liftoff_mass_kg(self) -> float:
        return self.stage_1.wet_mass_kg + self.stage_2.wet_mass_kg

    @property
    def total_dry_mass_kg(self) -> float:
        return self.stage_1.dry_mass_kg + self.stage_2.dry_mass_kg

    # ---- CG: the only thing 1 DOF needs it for is the stability check ----
    def cg_from_nose_m(self, *, stage2_alone: bool, propellant_burned_kg: float = 0.0) -> float:
        """CG station. Structure CG at section mid-length, propellant at grain mid-length.

        The propellant CG approximation holds because the grain burns radially at
        constant length, so its centroid does not move as it burns.
        """
        s2 = self.stage_2
        s2_len = s2.body_length_m

        if stage2_alone:
            motor_mass = s2.motor.wet_mass_kg - propellant_burned_kg
            moment = (
                s2.structure_mass_kg * 0.5 * s2_len
                + motor_mass * (s2_len - s2.motor.cg_from_aft_m)
            )
            return moment / (s2.structure_mass_kg + motor_mass)

        s1 = self.stage_1
        s1_motor_mass = s1.motor.wet_mass_kg - propellant_burned_kg
        moment = (
            s2.structure_mass_kg * 0.5 * s2_len
            + s2.motor.wet_mass_kg * (s2_len - s2.motor.cg_from_aft_m)
            + s1.structure_mass_kg * (s2_len + 0.5 * s1.body_length_m)
            + s1_motor_mass * (self.total_length_m - s1.motor.cg_from_aft_m)
        )
        mass = (
            s2.structure_mass_kg
            + s2.motor.wet_mass_kg
            + s1.structure_mass_kg
            + s1_motor_mass
        )
        return moment / mass

    def fin_root_le_from_nose_m(self, *, stage2_alone: bool) -> float:
        """Fin root leading edge station. Fins sit at the aft end of their stage."""
        if stage2_alone:
            return self.stage_2.body_length_m - self.stage_2.fins.root_chord_m
        return self.total_length_m - self.stage_1.fins.root_chord_m


def build(design: Design, config: Config) -> Rocket:
    """Resolve a Design into a Rocket."""
    motor_1 = build_motor(design.m_prop_1, config.nozzle_altitude_1_m)
    motor_2 = build_motor(design.m_prop_2, config.nozzle_altitude_2_m)

    # Team rule: body is at least motor length + 30 in.
    length_1 = motor_1.length_m + config.body_clearance_m + design.extra_length_1_m
    length_2 = motor_2.length_m + config.body_clearance_m + design.extra_length_2_m

    return Rocket(
        design=design,
        config=config,
        stage_1=Stage(
            name="stage_1",
            motor=motor_1,
            fins=config.fins_1,
            body_length_m=length_1,
            structure_mass_kg=design.structure_mass_1_kg + config.fins_1.total_mass_kg,
        ),
        stage_2=Stage(
            name="stage_2",
            motor=motor_2,
            fins=config.fins_2,
            body_length_m=length_2,
            structure_mass_kg=design.structure_mass_2_kg + config.fins_2.total_mass_kg,
        ),
    )
