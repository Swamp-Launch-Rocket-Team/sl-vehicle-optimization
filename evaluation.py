"""Shared evaluation function with exact-input, persistent JSON caching."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import scipy

from atmosphere import ConstantAtmosphere, isa, isa_with_tail
from candidate import CandidateBuilder, Design, InvalidCandidate
from two_dof import FlightError, constant_drag, simulate


@dataclass(frozen=True)
class SimulationSettings:
    atmosphere: str = "isa-tail"
    launch_altitude: float = 0.0
    wind_x: float = 0.0
    initial_conditions: tuple = (0.0, 0.0, 0.0, 0.0)
    constant_temperature: float = 288.15
    constant_pressure: float = 101325.0
    max_time: float = 600.0
    max_step: float = 0.2
    rtol: float = 1e-7
    atol: float = 1e-8

    def provider(self):
        if self.atmosphere == "isa":
            return isa
        if self.atmosphere == "isa-tail":
            return isa_with_tail
        if self.atmosphere == "constant":
            return ConstantAtmosphere(temperature=self.constant_temperature,
                                      pressure=self.constant_pressure)
        raise ValueError("Atmosphere must be isa, isa-tail, or constant.")


def model_identifier(builder, drag_model=constant_drag, drag_identifier=None):
    """Fingerprint source, motor bytes, and dependencies once per run.

    For custom drag, drag_identifier must describe external tables/settings;
    update it whenever those inputs change. Do not modify models mid-run.
    """
    folder = Path(__file__).resolve().parent
    names = ("candidate.py", "evaluation.py", "thrust_curve.py", "two_dof.py",
             "atmosphere.py", "mass_model.py", "vehicle.py", "geometry.py",
             "stability_barrowman.py", "fin_cp.py", "kraken.py")
    files = {name: sha256((folder/name).read_bytes()).hexdigest() for name in names}
    for name, motor in (("stage1_motor", builder.stage1_motor),
                        ("stage2_motor", builder.stage2_motor)):
        files[name] = sha256(motor.path.read_bytes()).hexdigest()
    if drag_model is not constant_drag:
        if not drag_identifier:
            raise ValueError("Custom drag requires an explicit drag_identifier.")
        source = inspect.getsourcefile(drag_model)
        if source:
            files["drag_source"] = sha256(Path(source).read_bytes()).hexdigest()
    return {"source_hashes": files, "drag": drag_identifier or "placeholder-cd-0.5",
            "numpy": np.__version__, "scipy": scipy.__version__,
            "stability": "OR-document-zero-AoA; linear-force-bridge-M0.8-1.2"}


def _stability_profiles(vehicle, trajectory):
    phases = {}
    for name, include_booster in (("stack", True), ("upper_stage", False)):
        mask = (trajectory.time <= vehicle.separation_time if include_booster
                else trajectory.time >= vehicle.separation_time)
        times, mach = trajectory.time[mask], trajectory.mach[mask]
        if not len(times):
            raise InvalidCandidate(f"No {name} flight samples.")
        mass_model, stability = vehicle._build_models(include_booster)
        stability.set_mach_number(mach)
        cp = stability.center_of_pressure()
        cg = mass_model.center_of_mass(times)
        margin = (cp-cg) / stability.reference_diameter
        if not np.isfinite(margin).all():
            raise InvalidCandidate("Nonfinite stability result.")
        phases[name] = {"time_s": times.tolist(), "mach": mach.tolist(),
                        "cp_m": cp.tolist(), "cg_m": cg.tolist(),
                        "margin_calibers": margin.tolist(),
                        "minimum_calibers": float(margin.min()),
                        "maximum_calibers": float(margin.max())}
    return phases


def evaluate_candidate(values, *, builder: CandidateBuilder, settings: SimulationSettings,
                       cache_dir, identifier=None, drag_model=constant_drag):
    """Objective and constraints call this same function; no evaluator class.

    Cache keys preserve exact floats. Cached values include apogee and both
    stability profiles, not the full flight trajectory. Expected candidate
    failures are cached; unexpected exceptions remain visible.
    """
    identifier = identifier or model_identifier(builder, drag_model)
    inputs = {"normalized_variables": np.asarray(values, dtype=float).tolist(),
              "fixed_inputs": asdict(builder.config), "simulation_settings": asdict(settings),
              "model_identifier": identifier}
    encoded_inputs = json.dumps(inputs, sort_keys=True, allow_nan=False)
    inputs = json.loads(encoded_inputs)
    key = sha256(encoded_inputs.encode()).hexdigest()
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result.get("identifier") != key:
            raise ValueError(f"Invalid cache entry: {path}")
        return result
    provider = settings.provider()
    result = {"identifier": key, **inputs, "status": "failed", "reason": None,
              "apogee_m": None, "stability": None, "design_variables": None}
    try:
        design = Design.from_normalized(values)
        result["design_variables"] = asdict(design)
        vehicle = builder.build(design)
        options = asdict(settings)
        for name in ("atmosphere", "constant_temperature", "constant_pressure"):
            options.pop(name)
        try:
            trajectory = simulate(vehicle, atmosphere_model=provider,
                                  drag_model=drag_model, **options)
        except ValueError as exc:
            # The strict ISA range is an expected model-domain failure.
            if "exceeds the supported ISA" not in str(exc):
                raise
            raise FlightError(str(exc)) from exc
        if trajectory.time[-1] <= vehicle.upper_stage.motor.ignition_time:
            raise InvalidCandidate("First apogee occurs before upper-stage ignition.")
        result.update(status="ok", apogee_m=trajectory.apogee,
                      apogee_time_s=float(trajectory.time[-1]),
                      max_sampled_mach=float(trajectory.mach.max()),
                      max_sampled_airspeed_m_s=float(trajectory.airspeed.max()),
                      motor_data={"stage1": builder.stage1_motor.summary(),
                                  "stage2": builder.stage2_motor.summary()},
                      stability=_stability_profiles(vehicle, trajectory))
        result["atmosphere_tail_used"] = bool(
            settings.atmosphere == "isa-tail"
            and settings.launch_altitude + trajectory.apogee > 86000.)
    except (InvalidCandidate, FlightError) as exc:
        result.update(status="failed", reason=str(exc))
    with tempfile.NamedTemporaryFile(mode="w", dir=folder, suffix=".tmp", delete=False) as out:
        temporary = out.name
        json.dump(result, out, allow_nan=False, indent=2)
    os.replace(temporary, path)
    return result
