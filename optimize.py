"""Command-line bringup and local optimization; see README for configuration."""

import argparse
from dataclasses import asdict, replace
from functools import partial
import importlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from candidate import CandidateBuilder, Design, VehicleConfig, VARIABLE_BOUNDS
from constraint import Constraint
from drag import drag_coefficient
from evaluation import SimulationSettings, evaluate_candidate, model_identifier
from objective import Objective
from two_dof import constant_drag


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON fixed inputs, initial design, and simulation settings")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate the starting design without optimization")
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output", type=Path, default=Path("optimization_result.json"))
    parser.add_argument("--atmosphere", choices=("isa", "isa-tail", "constant"))
    parser.add_argument("--drag", help="Replacement drag callback as module:function (default: drag:drag_coefficient)")
    parser.add_argument("--drag-id", help="Version/settings identifier (default model: drag-v1; required for replacements)")
    args = parser.parse_args(argv)
    if args.maxiter < 1:
        parser.error("--maxiter must be positive")
    data = json.loads(args.config.read_text()) if args.config else {}
    unknown = set(data) - {"fixed_inputs", "initial_design", "simulation", "stability"}
    if unknown:
        parser.error(f"Unknown configuration sections: {sorted(unknown)}")
    builder = CandidateBuilder(VehicleConfig(**data.get("fixed_inputs", {})))
    initial = Design(**data.get("initial_design", {}))
    initial.validate()
    settings = SimulationSettings(**data.get("simulation", {}))
    if args.atmosphere:
        settings = replace(settings, atmosphere=args.atmosphere)
    settings.provider()
    drag_model = drag_coefficient
    drag_identifier = args.drag_id
    if args.drag:
        if not args.drag_id:
            parser.error("--drag requires --drag-id to identify model/data changes")
        module, function = args.drag.split(":", 1)
        drag_model = getattr(importlib.import_module(module), function)
    elif not drag_identifier:
        drag_identifier = "drag-v1"
    identifier = model_identifier(builder, drag_model, drag_identifier)
    evaluate = partial(evaluate_candidate, builder=builder, settings=settings,
                       cache_dir=args.cache_dir, identifier=identifier, drag_model=drag_model)
    objective = Objective(evaluate)
    constraint = Constraint(evaluate, **data.get("stability", {}))
    x0 = initial.normalized()
    print("Drag:", identifier["drag"])
    print("Atmosphere:", settings.atmosphere)
    if settings.atmosphere == "isa-tail":
        print("Above ~86 km, a preliminary isothermal tail replaces ISA; gravity stays constant.")
    if drag_model is constant_drag:
        print("Placeholder Cd=0.5: results test the workflow, not final vehicle dimensions.")
    first = evaluate(x0)
    solver_options = {"maxiter": args.maxiter, "ftol": 1e-6, "eps": 1e-4, "disp": True}
    report = {"fixed_inputs": asdict(builder.config), "bounds": VARIABLE_BOUNDS,
              "initial_design": asdict(initial), "simulation_settings": asdict(settings),
              "stability_limits": {"minimum": constraint.minimum, "maximum": constraint.maximum},
              "model_identifier": identifier,
              "motor_model": identifier["propulsion"],
              "optimizer_settings": {"method": "SLSQP", "variable_scaling": "bounds to [0,1]",
                                     "apogee_scale_m": objective.apogee_scale, **solver_options},
              "initial_result": first, "optimizer": None}
    values = x0
    if not args.evaluate_only and first["status"] == "ok":
        solution = minimize(objective, x0, method="SLSQP",
                            bounds=[(0.0, 1.0)] * len(x0),
                            constraints=[{"type": "ineq", "fun": constraint}],
                            options=solver_options)
        values = solution.x
        report["optimizer"] = {"success": bool(solution.success), "status": int(solution.status),
                               "message": str(solution.message), "iterations": int(solution.nit),
                               "function_evaluations": int(solution.nfev),
                               "normalized_variables": values.tolist(), "cost": float(solution.fun)}
    final = evaluate(values)
    report["final_result"] = final
    # A finer final check is separate from optimization's sampled constraints.
    if final["status"] == "ok":
        finer = replace(settings, max_step=settings.max_step/4,
                        rtol=settings.rtol/10, atol=settings.atol/10)
        checked = evaluate_candidate(values, builder=builder, settings=finer,
                    cache_dir=args.cache_dir, identifier=identifier, drag_model=drag_model)
    else:
        checked = final
    report["final_validation"] = checked
    residuals = constraint.residuals(checked)
    feasible = checked["status"] == "ok" and bool(np.all(residuals >= -1e-6))
    report["feasible"] = feasible
    report["constraint_residuals"] = residuals.tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print("Evaluation:", checked["status"], checked.get("reason") or "")
    if checked["status"] == "ok":
        print(f"Apogee: {checked['apogee_m']:.1f} m; max sampled Mach: {checked['max_sampled_mach']:.3f}")
        for phase, profile in checked["stability"].items():
            print(f"{phase}: {profile['minimum_calibers']:.3f} to {profile['maximum_calibers']:.3f} calibers")
    print("Feasible:", feasible)
    print("Saved:", args.output)
    solver_ok = report["optimizer"] is None or report["optimizer"]["success"]
    return 0 if feasible and solver_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
