"""All configuration in one place.

Plain dataclasses rather than YAML: fewer moving parts, and the defaults are
visible next to the thing they configure. Nothing is hard-coded deeper in the
model -- every limit and every baseline lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

IN = 0.0254  # inches to metres


@dataclass(frozen=True)
class Limits:
    """Pass/fail criteria."""

    # Stability window, in calibers. Below the minimum the vehicle is unstable;
    # above the maximum it weathercocks into the wind.
    static_margin_min_cal: float = 1.0
    static_margin_max_cal: float = 2.3

    # Motor length cap (team rule: 52 in, applies to both motors).
    motor_length_max_m: float = 52.0 * IN

    # Nozzle must not separate at ignition (Summerfield).
    min_exit_pressure_ratio: float = 0.4

    # Sanity bounds. None means "record it, do not judge it".
    rail_exit_velocity_min_m_s: float | None = 20.0
    max_acceleration_g: float | None = None
    max_mach: float | None = None


@dataclass(frozen=True)
class Fins:
    """One stage's fin set. Fixed at a baseline -- geometry is not a free variable yet."""

    root_chord_m: float
    tip_chord_m: float
    semi_span_m: float
    sweep_length_m: float
    thickness_m: float
    count: int = 4
    mass_each_kg: float = 0.0
    material: str = "TBD"
    youngs_modulus_pa: float | None = None
    shear_modulus_pa: float | None = None
    has_control_surfaces: bool = False

    @property
    def total_mass_kg(self) -> float:
        return self.mass_each_kg * self.count

    @property
    def mid_chord_sweep_m(self) -> float:
        """Sweep at the mid-chord line, which is what Barrowman uses."""
        return self.sweep_length_m + 0.5 * (self.tip_chord_m - self.root_chord_m)


@dataclass(frozen=True)
class Bounds:
    """Free variable ranges. lo == hi pins a variable to a constant."""

    m_prop_1: tuple[float, float] = (4.0, 12.45)
    m_prop_2: tuple[float, float] = (4.0, 12.45)
    t_ignite_s: tuple[float, float] = (0.0, 30.0)
    # Body length above the motor + 30 in minimum.
    extra_length_1_m: tuple[float, float] = (0.0, 0.0)
    extra_length_2_m: tuple[float, float] = (0.0, 0.0)
    # Structure mass per stage, excluding the motor.
    structure_mass_1_kg: tuple[float, float] = (8.0, 14.0)
    structure_mass_2_kg: tuple[float, float] = (7.0, 13.0)


@dataclass(frozen=True)
class Config:
    """Everything the model needs."""

    body_diameter_m: float = 4.0 * IN
    # Team rule: each stage's body is at least motor length + 30 in.
    body_clearance_m: float = 30.0 * IN
    nose_length_m: float = 21.0 * IN
    nose_shape: str = "von_karman"

    rail_length_m: float = 18.0 * 0.3048
    launch_elevation_m: float = 0.0

    # Stage 1 ignites at sea level so its nozzle stays near-matched. Stage 2
    # ignites high, and its expansion saturates at the casing bore by ~10 km.
    nozzle_altitude_1_m: float = 0.0
    nozzle_altitude_2_m: float = 10000.0

    # Separation delay after stage 1 burnout. Hardware-driven, not a free variable.
    t_separation_s: float = 1.0

    limits: Limits = field(default_factory=Limits)
    bounds: Bounds = field(default_factory=Bounds)

    # Baseline fins. Chord and thickness come from the Orpheus & Eurydice CAD;
    # semi-span was then sized so the baseline lands mid-window on static margin.
    #
    # Two things to know. OE's absolute fin dimensions were measured on a 6.17 in
    # booster, so reusing them on a 4 in body gives an oversized fin. And this
    # vehicle is ~41 calibers long, which makes static margin HYPERSENSITIVE to
    # span: 1.6 in of semi-span gives -2.1 cal (unstable) and 2.3 in gives +4.0
    # cal (well overstable). The usable band is only a few tenths of an inch wide.
    #
    # PROVISIONAL. Real fin design is a later task and these will change.
    fins_1: Fins = field(
        default_factory=lambda: Fins(
            root_chord_m=7.0 * IN,
            tip_chord_m=3.0 * IN,
            semi_span_m=2.05 * IN,
            sweep_length_m=5.0 * IN,
            thickness_m=0.257 * IN,
            mass_each_kg=0.30,
            material="GFRP",
            youngs_modulus_pa=2.3e10,
            shear_modulus_pa=8.9e9,
            has_control_surfaces=False,
        )
    )
    fins_2: Fins = field(
        default_factory=lambda: Fins(
            root_chord_m=6.0 * IN,
            tip_chord_m=3.0 * IN,
            semi_span_m=2.40 * IN,
            sweep_length_m=4.0 * IN,
            thickness_m=0.190 * IN,
            mass_each_kg=0.12,
            material="CFRP",
            youngs_modulus_pa=7.0e10,
            shear_modulus_pa=5.0e9,
            has_control_surfaces=True,
        )
    )
