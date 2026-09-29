"""Evaluate designs and search for maximum apogee.

`evaluate` never raises on a bad design -- it returns a Result whose checks say
what went wrong. Failed constraints are never folded into a score.

`sweep` walks a grid. With a fast 1 DOF model a grid beats a gradient-free
optimizer here: it returns the whole landscape and the shape of the feasible
region, not just one point. `refine` polishes the best grid point.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .checks import Check, FlutterModel, run_all
from .flight import Flight, simulate
from .params import Config
from .rocket import VARIABLES, Design, Rocket, build


@dataclass
class Result:
    design: Design
    rocket: Rocket | None
    flight: Flight | None
    checks: list[Check] = field(default_factory=list)
    apogee_m: float = 0.0
    error: str | None = None

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def blocked(self) -> list[Check]:
        """Checks that cannot pass yet: no limit set, or no model implemented."""
        return [c for c in self.checks if c.status in ("skipped", "not_implemented")]

    @property
    def feasible(self) -> bool:
        """Every check passes. Blocked checks do not count as passes."""
        return self.error is None and bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def constraints_satisfied(self) -> bool:
        """Nothing failed. Blocked checks tolerated -- useful while flutter is unimplemented."""
        return self.error is None and bool(self.checks) and not self.failed

    def row(self) -> dict:
        row = dict(self.design.as_dict())
        row["apogee_km"] = self.apogee_m / 1000.0
        row["ok"] = self.constraints_satisfied
        row["n_failed"] = len(self.failed)
        row["failed"] = ";".join(c.name for c in self.failed)
        if self.rocket:
            row["liftoff_kg"] = self.rocket.liftoff_mass_kg
            row["dry_kg"] = self.rocket.total_dry_mass_kg
            row["length_m"] = self.rocket.total_length_m
        if self.flight:
            row["max_mach"] = self.flight.max_mach
            row["max_g"] = self.flight.max_acceleration_g
            row["rail_exit_m_s"] = self.flight.rail_exit_velocity_m_s
        for check in self.checks:
            if check.name.startswith("stability_") and check.value is not None:
                row[check.name] = check.value
        return row


# Timestep for an accurate single evaluation, and for cheap grid screening.
# Convergence against dt = 0.005 s: 0.02 costs 0.11% of apogee, 0.05 costs 0.33%.
# Both are far inside the drag model's own uncertainty, so screening coarse and
# confirming fine is free accuracy.
DT_ACCURATE = 0.02
DT_SCREEN = 0.05


def evaluate(
    design: Design,
    config: Config,
    *,
    temperature_offset_k: float = 0.0,
    cd_multiplier: float = 1.0,
    flutter_model: FlutterModel | None = None,
    dt: float = DT_ACCURATE,
) -> Result:
    """Build, fly, and check one design."""
    try:
        rocket = build(design, config)
    except Exception as exc:  # outside the model's domain, not a bug
        return Result(design=design, rocket=None, flight=None, error=str(exc))

    flight = simulate(
        rocket, temperature_offset_k=temperature_offset_k, cd_multiplier=cd_multiplier, dt=dt
    )
    return Result(
        design=design,
        rocket=rocket,
        flight=flight,
        checks=run_all(rocket, flight, config.limits, flutter_model),
        apogee_m=flight.apogee_m,
    )


@dataclass
class Sweep:
    results: list[Result]

    def rows(self) -> list[dict]:
        return [r.row() for r in self.results]

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.rows())

    @property
    def best(self) -> Result | None:
        """Highest apogee with no failed check."""
        ok = [r for r in self.results if r.constraints_satisfied]
        return max(ok, key=lambda r: r.apogee_m) if ok else None

    @property
    def best_ignoring_constraints(self) -> Result | None:
        valid = [r for r in self.results if r.error is None]
        return max(valid, key=lambda r: r.apogee_m) if valid else None

    def sensitivity(self) -> dict[str, float]:
        """Apogee span across each variable's levels -- which variables matter."""
        out = {}
        rows = [r for r in self.rows() if r["apogee_km"] > 0]
        for name in VARIABLES:
            best_by_level: dict[float, float] = {}
            for row in rows:
                level = row[name]
                best_by_level[level] = max(best_by_level.get(level, 0.0), row["apogee_km"])
            if len(best_by_level) > 1:
                out[name] = max(best_by_level.values()) - min(best_by_level.values())
        return out


def levels(bounds: tuple[float, float], n: int) -> list[float]:
    """n evenly spaced levels across a bound. A pinned bound gives one level."""
    low, high = bounds
    if n < 2 or high <= low:
        return [low]
    step = (high - low) / (n - 1)
    return [low + i * step for i in range(n)]


def grid_from_bounds(config: Config, n: int = 5) -> dict[str, list[float]]:
    """Default grid: n levels per variable, from the bounds in config."""
    bounds = config.bounds
    return {
        "m_prop_1": levels(bounds.m_prop_1, n),
        "m_prop_2": levels(bounds.m_prop_2, n),
        "t_ignite_s": levels(bounds.t_ignite_s, n),
        "extra_length_1_m": levels(bounds.extra_length_1_m, n),
        "extra_length_2_m": levels(bounds.extra_length_2_m, n),
        "structure_mass_1_kg": levels(bounds.structure_mass_1_kg, n),
        "structure_mass_2_kg": levels(bounds.structure_mass_2_kg, n),
    }


def sweep(
    grid: dict[str, list[float]],
    config: Config,
    *,
    flutter_model: FlutterModel | None = None,
    progress: bool = False,
    dt: float = DT_SCREEN,
) -> Sweep:
    """Evaluate every combination in the grid, keeping all results.

    Uses the coarse screening timestep by default. Re-evaluate the winner with
    `evaluate(..., dt=DT_ACCURATE)` before quoting its apogee.
    """
    names = list(VARIABLES)
    combinations = list(itertools.product(*(grid[name] for name in names)))
    results = []
    for index, values in enumerate(combinations, start=1):
        design = Design(**dict(zip(names, values)))
        results.append(evaluate(design, config, flutter_model=flutter_model, dt=dt))
        if progress and (index % 500 == 0 or index == len(combinations)):
            print(f"  {index}/{len(combinations)}")
    return Sweep(results=results)


def refine(
    seed: Design,
    config: Config,
    *,
    flutter_model: FlutterModel | None = None,
    max_iterations: int = 300,
) -> Result:
    """Nelder-Mead polish around a seed design, respecting bounds.

    Failed constraints are penalised so the simplex can still cross infeasible
    ground. The penalty only steers the search -- the returned Result carries the
    real check record.
    """
    from scipy.optimize import minimize

    bounds = config.bounds
    ranges = [getattr(bounds, name if name != "t_ignite_s" else "t_ignite_s") for name in VARIABLES]
    start = [getattr(seed, name) for name in VARIABLES]

    def clamp(values):
        return [min(max(v, lo), hi) for v, (lo, hi) in zip(values, ranges)]

    def objective(values) -> float:
        design = Design(**dict(zip(VARIABLES, clamp(values))))
        result = evaluate(design, config, flutter_model=flutter_model)
        if result.error:
            return 1e9
        return -result.apogee_m / 1000.0 + 100.0 * len(result.failed)

    outcome = minimize(
        objective, start, method="Nelder-Mead",
        options={"maxiter": max_iterations, "xatol": 1e-3, "fatol": 1e-3},
    )
    return evaluate(
        Design(**dict(zip(VARIABLES, clamp(outcome.x)))), config, flutter_model=flutter_model
    )
