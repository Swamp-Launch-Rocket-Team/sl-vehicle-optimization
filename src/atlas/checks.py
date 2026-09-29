"""Constraint checks.

Each returns a Check carrying its value, limit and status. Status "skipped" means
no limit is set (recorded, not judged); "not_implemented" means the check exists
as plumbing but the model behind it is missing -- currently only fin flutter.

Neither counts as a pass. A design is feasible only when every check passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from .aero import CD_MAX_MACH, centre_of_pressure
from .atmosphere import air_at
from .flight import Flight
from .params import Fins, Limits
from .rocket import Rocket

Status = Literal["pass", "fail", "skipped", "not_implemented"]


@dataclass(frozen=True)
class Check:
    name: str
    value: float | None
    status: Status
    limit_low: float | None = None
    limit_high: float | None = None
    units: str = ""
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "pass"


def _within(
    name: str,
    value: float | None,
    low: float | None,
    high: float | None,
    units: str,
    note: str = "",
) -> Check:
    """Check value against an optional low and/or high limit."""
    if value is None or (low is None and high is None):
        return Check(name, value, "skipped", low, high, units, note or "no limit set")
    if low is not None and value < low:
        return Check(name, value, "fail", low, high, units, note)
    if high is not None and value > high:
        return Check(name, value, "fail", low, high, units, note)
    return Check(name, value, "pass", low, high, units, note)


# --------------------------------------------------------------------------
# Stability -- the primary constraint
# --------------------------------------------------------------------------

def static_margin_cal(rocket: Rocket, *, stage2_alone: bool, mach: float) -> tuple[float, bool]:
    """Static margin in calibers, plus whether the CP estimate is approximate."""
    cp = centre_of_pressure(
        body_diameter_m=rocket.body_diameter_m,
        nose_length_m=rocket.config.nose_length_m,
        nose_shape=rocket.config.nose_shape,
        fins=rocket.stage_2.fins if stage2_alone else rocket.stage_1.fins,
        fin_root_le_from_nose_m=rocket.fin_root_le_from_nose_m(stage2_alone=stage2_alone),
        mach=mach,
    )
    cg = rocket.cg_from_nose_m(stage2_alone=stage2_alone)
    return (cp.station_from_nose_m - cg) / rocket.body_diameter_m, cp.approximate


# Mach at which the static margin criterion is evaluated. Barrowman is a
# subsonic linear theory, so the margin is checked where it is actually valid --
# which is also where the criterion is conventionally specified (rail exit).
STABILITY_EVAL_MACH = 0.3


def check_stability(rocket: Rocket, flight: Flight, limits: Limits) -> list[Check]:
    """Static margin must stay inside the window for both configurations.

    Evaluated subsonically for both the mated stack and stage 2 alone. The stack
    case is the binding one at liftoff, when CG is furthest aft.

    Supersonic static margin is deliberately NOT judged. Barrowman with a
    Prandtl-Glauert correction returns nonsense at high Mach -- at Mach 4.5 it
    scales fin lift by 0.23 and puts CP ahead of CG, predicting an unstable
    vehicle, which is a model artefact rather than a real effect. A supersonic CP
    method is needed before that condition can be checked.
    """
    checks = []
    for name, stage2_alone in (("stability_stack", False), ("stability_stage2", True)):
        margin, _ = static_margin_cal(
            rocket, stage2_alone=stage2_alone, mach=STABILITY_EVAL_MACH
        )
        checks.append(
            _within(
                name,
                margin,
                limits.static_margin_min_cal,
                limits.static_margin_max_cal,
                "cal",
                f"Barrowman at Mach {STABILITY_EVAL_MACH}, where it is valid",
            )
        )

    checks.append(
        Check(
            "stability_supersonic",
            flight.max_mach,
            "not_implemented",
            units="max Mach",
            note="supersonic CP not modelled; Barrowman + Prandtl-Glauert is "
                 "invalid here and would report a spurious negative margin. "
                 "Needs a supersonic CP method before this can be judged.",
        )
    )
    return checks


# --------------------------------------------------------------------------
# Fin flutter -- structure only, no model yet
# --------------------------------------------------------------------------

class FlutterModel(Protocol):
    """Interface a real flutter model must satisfy.

    Implement this and pass it to `check_flutter` to activate the check. Nothing
    else needs to change.
    """

    def flutter_velocity_m_s(
        self, fins: Fins, altitude_m: float, mach: float, air_density_kg_m3: float
    ) -> float | None:
        """Flutter onset velocity, or None where the method does not apply."""
        ...


class NoFlutterModel:
    """Placeholder. Returns None everywhere, so the check reports not_implemented.

    A real implementation needs the 1/k method with the Theodorsen function
    (NACA TR685/496) subsonic and Kearns above Mach 1, plus fin modal frequencies
    -- best from a Fusion Simulation modal study.

    NACA TN4197 and Bennett's correction are explicitly ruled out by the team's
    literature review (Literature/Header.txt), and they are the closed-form
    equations most rocketry tools use, RocketPy included. So there is deliberately
    no easy substitute here: an unimplemented check is honest, a wrong one is not.
    """

    def flutter_velocity_m_s(self, fins, altitude_m, mach, air_density_kg_m3):
        return None


def check_flutter(
    rocket: Rocket,
    flight: Flight,
    model: FlutterModel | None = None,
    min_safety_factor: float | None = None,
) -> list[Check]:
    """Flutter margin per fin set, at the worst condition that fin set sees.

    Each fin set is evaluated at its own peak dynamic pressure, since stage 1 and
    stage 2 experience very different conditions -- the sustainer sees the highest
    Mach at the lowest density.
    """
    model = model or NoFlutterModel()
    checks = []

    fin_sets = (
        ("flutter_stage_1", rocket.stage_1.fins, ("burn_1", "coast_stack")),
        ("flutter_stage_2", rocket.stage_2.fins, ("coast_s2", "burn_2", "coast_final")),
    )
    for name, fins, phases in fin_sets:
        relevant = [s for s in flight.samples if s.phase in phases]
        if not relevant:
            checks.append(Check(name, None, "skipped", units="-", note="no samples"))
            continue
        worst = max(relevant, key=lambda s: s.dynamic_pressure_pa)
        air = air_at(worst.altitude_m)

        flutter_velocity = model.flutter_velocity_m_s(
            fins, worst.altitude_m, worst.mach, air.density_kg_m3
        )
        if flutter_velocity is None:
            note = (
                "no flutter model implemented; needs 1/k Theodorsen (subsonic) or "
                "Kearns (supersonic) plus fin modal frequencies"
            )
            if fins.has_control_surfaces:
                note += (
                    ". This fin set carries movable control surfaces, whose "
                    "rotational degree of freedom binary bending-torsion methods "
                    "do not model at all"
                )
            checks.append(
                Check(
                    name,
                    worst.velocity_m_s,
                    "not_implemented",
                    units="m/s at peak q",
                    note=note,
                )
            )
            continue

        margin = flutter_velocity / worst.velocity_m_s if worst.velocity_m_s > 0 else float("inf")
        checks.append(_within(name, margin, min_safety_factor, None, "V_flutter/V"))

    return checks


# --------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------

def check_basics(rocket: Rocket, flight: Flight, limits: Limits) -> list[Check]:
    """Motor length, nozzle separation, rail exit, and model-validity guards."""
    from .motor import bore_exit_area_m2

    checks = [
        _within(
            "motor_length_stage_1", rocket.stage_1.motor.length_m,
            None, limits.motor_length_max_m, "m", "derived from propellant mass",
        ),
        _within(
            "motor_length_stage_2", rocket.stage_2.motor.length_m,
            None, limits.motor_length_max_m, "m", "derived from propellant mass",
        ),
        _within(
            "nozzle_fits_bore_stage_2", rocket.stage_2.motor.exit_area_m2,
            None, bore_exit_area_m2(), "m^2", "clamped at build; this verifies it",
        ),
        _within(
            "no_separation_stage_1",
            rocket.stage_1.motor.exit_pressure_ratio(air_at(0.0).pressure_pa),
            limits.min_exit_pressure_ratio, None, "pe/pa",
            "stage 1 ignites at sea level",
        ),
        _within(
            "rail_exit_velocity", flight.rail_exit_velocity_m_s,
            limits.rail_exit_velocity_min_m_s, None, "m/s",
            "1 DOF cannot verify fin authority against rail tip-off",
        ),
        _within(
            "max_acceleration", flight.max_acceleration_g,
            None, limits.max_acceleration_g, "g",
            "neutral burn assumed; a progressive grain pushes peak g higher",
        ),
        _within("max_mach", flight.max_mach, None, limits.max_mach, "-"),
    ]

    # Stage 2 may ignite above the atmosphere, where separation cannot occur.
    if flight.ignition_pressure_pa > 0.0:
        checks.append(
            _within(
                "no_separation_stage_2",
                rocket.stage_2.motor.exit_pressure_ratio(flight.ignition_pressure_pa),
                limits.min_exit_pressure_ratio, None, "pe/pa",
                "at the stage 2 ignition altitude",
            )
        )

    # Guard against the optimizer exploiting extrapolated drag data.
    checks.append(
        Check(
            "cd_data_covers_flight",
            flight.max_mach,
            "fail" if flight.cd_extrapolated else "pass",
            limit_high=CD_MAX_MACH,
            units="-",
            note="drag was extrapolated past the CD table; apogee is unreliable"
            if flight.cd_extrapolated
            else "",
        )
    )
    return checks


def run_all(
    rocket: Rocket,
    flight: Flight,
    limits: Limits,
    flutter_model: FlutterModel | None = None,
) -> list[Check]:
    return (
        check_basics(rocket, flight, limits)
        + check_stability(rocket, flight, limits)
        + check_flutter(rocket, flight, flutter_model)
    )
