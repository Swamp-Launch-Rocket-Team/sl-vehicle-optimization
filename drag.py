"""Aerodynamic drag coefficient model for the flight simulator."""

from math import isclose, isfinite, log, pi, sqrt

import numpy as np

from atmosphere import Atmosphere
from two_dof import FlightState
from vehicle import Vehicle


def skin_friction_coefficient(reynolds, mach):
    """OpenRocket normal-finish Cf with compressibility correction.

    Reynolds and Mach inputs may be scalars or broadcastable arrays.
    """
    if np.isscalar(reynolds) and np.isscalar(mach):
        reynolds = float(reynolds)
        mach = float(mach)
        if not isfinite(reynolds) or reynolds <= 0:
            raise ValueError("Reynolds numbers must be finite and positive.")
        if not isfinite(mach) or mach < 0:
            raise ValueError("Mach numbers must be finite and nonnegative.")
        cf = 0.0148
        if reynolds >= 1e4:
            cf = 1 / (1.5 * log(reynolds) - 5.6)**2
        if mach < 0.9:
            correction = 1 - 0.1 * mach**2
        elif mach < 1.1:
            k_sub = 1 - 0.1 * mach**2
            k_sup = (1 + 0.15 * mach**2)**-0.58
            weight = (mach - 0.9) / (1.1 - 0.9)
            correction = (1 - weight) * k_sub + weight * k_sup
        else:
            correction = (1 + 0.15 * mach**2)**-0.58
        return cf * correction

    reynolds, mach = np.broadcast_arrays(
        np.asarray(reynolds, dtype=float), np.asarray(mach, dtype=float),
    )
    if np.any(~np.isfinite(reynolds)) or np.any(reynolds <= 0):
        raise ValueError("Reynolds numbers must be finite and positive.")
    if np.any(~np.isfinite(mach)) or np.any(mach < 0):
        raise ValueError("Mach numbers must be finite and nonnegative.")

    cf = np.full_like(reynolds, 0.0148)
    turbulent = reynolds >= 1e4
    cf[turbulent] = 1 / (1.5 * np.log(reynolds[turbulent]) - 5.6)**2

    k_sub = 1 - 0.1 * mach**2
    k_sup = (1 + 0.15 * mach**2)**-0.58
    # Clipping selects the respective correction outside the transonic band.
    weight = np.clip((mach - 0.9) / (1.1 - 0.9), 0, 1)
    correction = (1 - weight) * k_sub + weight * k_sup
    return cf * correction


def stagnation_cd(mach: float) -> float:
    """OpenRocket stagnation coefficient (BarrowmanDragCalculator, 04d3529)."""
    if not isfinite(mach) or mach < 0:
        raise ValueError("Mach number must be finite and nonnegative.")
    if mach <= 1.0:
        pressure = 1.0 + mach**2 / 4.0 + mach**4 / 40.0
    else:
        pressure = (1.84 - 0.76 / mach**2 + 0.166 / mach**4
                    + 0.035 / mach**6)
    return 0.85 * pressure


def base_cd(mach: float) -> float:
    """OpenRocket base coefficient (BarrowmanDragCalculator, 04d3529)."""
    if not isfinite(mach) or mach < 0:
        raise ValueError("Mach number must be finite and nonnegative.")
    if mach <= 1.0:
        return 0.12 + 0.13 * mach**2
    return 0.25 / mach


def body_wetted_area(vehicle: Vehicle, state: FlightState) -> float:
    """Active body's external lateral area [m^2], including the nose."""
    return vehicle.body_wetted_area(upper_stage_only=state.upper_stage_only)


def _body_length(vehicle: Vehicle, state: FlightState) -> float:
    """Active nose and body length [m], excluding fin overhang."""
    body_length = vehicle.nose_cone.section.length
    stages = [vehicle.upper_stage]
    if not state.upper_stage_only:
        stages.append(vehicle.booster_stage)
    for stage in stages:
        body_length += sum(component.section.length for component in stage.body_sections)
    return body_length


def body_reynolds_number(
    vehicle: Vehicle,
    state: FlightState,
    ambient: Atmosphere,
) -> float:
    """Return Re based on active body length including the nose (SI inputs)."""
    return state.airspeed * _body_length(vehicle, state) / ambient.kinematic_viscosity


def body_friction_drag_coefficient(
    vehicle: Vehicle,
    state: FlightState,
    ambient: Atmosphere,
    *,
    cfc: float | None = None,
) -> float:
    """Body skin-friction CD: Barrowman 4-16 with normal-finish Cfc.

    Uses body length for Reynolds number; fin overhang and surface-roughness
    limits are excluded. At rest, uses the low-Re coefficient's finite limit.
    """
    if not isfinite(state.airspeed) or state.airspeed < 0:
        raise ValueError("Airspeed must be finite and nonnegative.")
    if not isfinite(state.reference_diameter) or state.reference_diameter <= 0:
        raise ValueError("Reference diameter must be finite and positive.")
    if cfc is None:
        reynolds = body_reynolds_number(vehicle, state, ambient)
        if state.airspeed == 0:
            # Any positive Re below 1e4 selects the same low-Re limiting branch.
            reynolds = 1.0
        cfc = skin_friction_coefficient(reynolds, state.mach)
    body_length = _body_length(vehicle, state)
    maximum_diameter = vehicle.maximum_body_diameter(
        upper_stage_only=state.upper_stage_only,
    )
    fineness_correction = 1 + maximum_diameter / (2 * body_length)
    reference_area = pi * (state.reference_diameter / 2)**2
    return float(fineness_correction * cfc * body_wetted_area(vehicle, state)
                 / reference_area)


def fin_friction_drag_coefficient(
    vehicle: Vehicle,
    state: FlightState,
    ambient: Atmosphere,
    *,
    cfc: float | None = None,
) -> float:
    """Active fin-set skin-friction CD using OpenRocket's thickness factor.

    Uses the shared body-length Reynolds number, without roughness limits or
    fin overhang. Planform area is doubled for two faces; fins are counted once.
    """
    if not isfinite(state.airspeed) or state.airspeed < 0:
        raise ValueError("Airspeed must be finite and nonnegative.")
    if not isfinite(state.reference_diameter) or state.reference_diameter <= 0:
        raise ValueError("Reference diameter must be finite and positive.")
    if cfc is None:
        reynolds = body_reynolds_number(vehicle, state, ambient)
        if state.airspeed == 0:
            # Use the same low-Re limiting branch as body skin friction.
            reynolds = 1.0
        cfc = skin_friction_coefficient(reynolds, state.mach)
    reference_area = pi * (state.reference_diameter / 2)**2
    stages = [vehicle.upper_stage]
    if not state.upper_stage_only:
        stages.append(vehicle.booster_stage)
    cd = 0.0
    for stage in stages:
        for component in stage.fin_sets:
            fins = component.fin_set
            thickness_factor = 1 + 2 * component.fin_thickness / fins.fin_mean_aerodynamic_chord
            two_face_area = 2 * fins.number_of_fins * fins.fin_planform_area
            cd += cfc * thickness_factor * two_face_area / reference_area
    return float(cd)


def fin_leading_pressure_drag_coefficient(vehicle: Vehicle, state: FlightState) -> float:
    """OpenRocket fin leading-edge pressure CD (FinSetCalc, 3fb6ec7).

    Rounded and airfoil sections share the rounded-edge correlation. Each
    fin's projected edge area includes the squared leading-edge sweep cosine.
    """
    mach = state.mach
    if not isfinite(mach) or mach < 0:
        raise ValueError("Mach number must be finite and nonnegative.")
    if not isfinite(state.reference_diameter) or state.reference_diameter <= 0:
        raise ValueError("Reference diameter must be finite and positive.")
    if mach < 0.9:
        rounded_edge_cd = (1 - mach**2)**-0.417 - 1
    elif mach < 1.0:
        rounded_edge_cd = 1 - 1.785 * (mach - 0.9)
    else:
        rounded_edge_cd = 1.214 - 0.502 / mach**2 + 0.1095 / mach**4

    reference_area = pi * (state.reference_diameter / 2)**2
    stages = [vehicle.upper_stage]
    if not state.upper_stage_only:
        stages.append(vehicle.booster_stage)
    cd = 0.0
    for stage in stages:
        for component in stage.fin_sets:
            fins = component.fin_set
            if fins.fin_cross_section == "square":
                edge_cd = stagnation_cd(mach)
            elif fins.fin_cross_section in ("rounded", "airfoil"):
                edge_cd = rounded_edge_cd
            else:
                raise ValueError("Fin cross-section must be square, rounded, or airfoil.")
            sweep_cosine_squared = fins.fin_span**2 / (
                fins.fin_span**2 + fins.fin_leading_edge_sweep_distance**2
            )
            edge_area = fins.number_of_fins * fins.fin_span * component.fin_thickness
            cd += edge_cd * sweep_cosine_squared * edge_area / reference_area
    return float(cd)


def fin_trailing_base_drag_coefficient(vehicle: Vehicle, state: FlightState) -> float:
    """OpenRocket fin trailing-edge base CD (FinSetCalc, 3fb6ec7)."""
    shared_base_cd = base_cd(state.mach)
    if not isfinite(state.reference_diameter) or state.reference_diameter <= 0:
        raise ValueError("Reference diameter must be finite and positive.")
    reference_area = pi * (state.reference_diameter / 2)**2
    stages = [vehicle.upper_stage]
    if not state.upper_stage_only:
        stages.append(vehicle.booster_stage)
    cd = 0.0
    for stage in stages:
        for component in stage.fin_sets:
            fins = component.fin_set
            if fins.fin_cross_section == "square":
                trailing_cd = shared_base_cd
            elif fins.fin_cross_section == "rounded":
                trailing_cd = 0.5 * shared_base_cd
            elif fins.fin_cross_section == "airfoil":
                trailing_cd = 0.0
            else:
                raise ValueError("Fin cross-section must be square, rounded, or airfoil.")
            edge_area = fins.number_of_fins * fins.fin_span * component.fin_thickness
            cd += trailing_cd * edge_area / reference_area
    return float(cd)


def body_base_drag_coefficient(vehicle: Vehicle, state: FlightState) -> float:
    """OpenRocket base CD for a continuous body with one exposed aft base.

    Buried interfaces are excluded; exposed steps are unsupported. No nozzle
    exit area is subtracted because nozzle geometry is not supplied.
    """
    shared_base_cd = base_cd(state.mach)
    if not isfinite(state.reference_diameter) or state.reference_diameter <= 0:
        raise ValueError("Reference diameter must be finite and positive.")
    aft_section = vehicle.nose_cone.section
    stages = [vehicle.upper_stage]
    if not state.upper_stage_only:
        stages.append(vehicle.booster_stage)
    for stage in stages:
        for component in stage.body_sections:
            section = component.section
            if not isclose(aft_section.aft_cross_sectional_area,
                           section.forward_cross_sectional_area,
                           rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Body base drag does not support exposed diameter steps.")
            aft_section = section
    reference_area = pi * (state.reference_diameter / 2)**2
    return shared_base_cd * aft_section.aft_cross_sectional_area / reference_area


def body_pressure_drag_coefficient(
    vehicle: Vehicle,
    state: FlightState,
    ambient: Atmosphere,
    *,
    cfc: float | None = None,
) -> float:
    """Body pressure CD from Barrowman Equations 4-41, 4-45, and 4-42.

    The conical correction is deliberately applied to every nose profile.
    For M >= 1, hold Equation 4-42's sonic value constant; this is not a
    physically validated supersonic body pressure-drag correlation.
    Body length excludes fin overhang, as in the current friction model.
    """
    if not isfinite(state.mach) or state.mach < 0:
        raise ValueError("Mach number must be finite and nonnegative.")
    if not isfinite(state.airspeed) or state.airspeed < 0:
        raise ValueError("Airspeed must be finite and nonnegative.")
    if not isfinite(state.reference_diameter) or state.reference_diameter <= 0:
        raise ValueError("Reference diameter must be finite and positive.")
    if not isfinite(ambient.speed_of_sound) or ambient.speed_of_sound <= 0:
        raise ValueError("Speed of sound must be finite and positive.")

    body_length = _body_length(vehicle, state)
    maximum_diameter = vehicle.maximum_body_diameter(
        upper_stage_only=state.upper_stage_only,
    )
    nose = vehicle.nose_cone.section
    nose_base_diameter = 2 * sqrt(nose.aft_cross_sectional_area / pi)
    if maximum_diameter <= 0 or nose_base_diameter <= 0:
        raise ValueError("Body and nose-base diameters must be positive.")
    body_fineness = body_length / maximum_diameter
    nose_fineness = nose.length / nose_base_diameter
    reference_area = pi * (state.reference_diameter / 2)**2
    if state.mach >= 1:
        cd_sonic = 0.88 / (nose_fineness + 0.7)**1.29
        return cd_sonic * nose.aft_cross_sectional_area / reference_area
    wetted_area = body_wetted_area(vehicle, state)

    viscosity = ambient.kinematic_viscosity
    if cfc is None:
        reynolds = body_reynolds_number(vehicle, state, ambient)
        if state.airspeed == 0:
            # Use the same finite low-Re limit as the friction contributions.
            reynolds = 1.0
        cfc = skin_friction_coefficient(reynolds, state.mach)
    reynolds_star = ambient.speed_of_sound * body_length / viscosity
    cfc_star = skin_friction_coefficient(reynolds_star, 1.0)
    k = 1 + (6.82 * wetted_area * cfc_star * (nose_fineness + 0.7)**1.29
             / (body_fineness**3 * reference_area))**(5 / 3)
    return float(6 * wetted_area * cfc
                 / (body_fineness**3 * reference_area * (k - state.mach**2)**0.6))


def drag_contributions(
    vehicle: Vehicle,
    state: FlightState,
    ambient: Atmosphere,
) -> dict[str, float]:
    """Six zero-AoA CD contributions sharing state.reference_diameter.

    Supersonic body pressure uses the constant sonic-value approximation.
    """
    if not isfinite(state.airspeed) or state.airspeed < 0:
        raise ValueError("Airspeed must be finite and nonnegative.")
    reynolds = body_reynolds_number(vehicle, state, ambient)
    if state.airspeed == 0:
        reynolds = 1.0
    cfc = float(skin_friction_coefficient(reynolds, state.mach))
    contributions = {
        "body_friction": body_friction_drag_coefficient(vehicle, state, ambient, cfc=cfc),
        "fin_friction": fin_friction_drag_coefficient(vehicle, state, ambient, cfc=cfc),
        "body_pressure": body_pressure_drag_coefficient(vehicle, state, ambient, cfc=cfc),
        "fin_leading": fin_leading_pressure_drag_coefficient(vehicle, state),
        "body_base": body_base_drag_coefficient(vehicle, state),
        "fin_trailing": fin_trailing_base_drag_coefficient(vehicle, state),
    }
    for name, coefficient in contributions.items():
        if not isfinite(coefficient) or coefficient < 0:
            raise ValueError(f"{name} drag coefficient must be finite and nonnegative.")
    return contributions


def drag_coefficient(
    vehicle: Vehicle,
    state: FlightState,
    ambient: Atmosphere,
) -> float:
    """Return total Cd referenced to state.reference_diameter (SI inputs)."""
    return sum(drag_contributions(vehicle, state, ambient).values())
