"""ATLAS apogee optimizer.

    py run.py                  evaluate the baseline design
    py run.py --sweep          grid sweep, then report the best feasible design
    py run.py --sweep -n 4     4 levels per free variable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from atlas import (  # noqa: E402
    Config, Design, Result, evaluate, grid_from_bounds, max_propellant_mass, refine, sweep,
)
from atlas.checks import static_margin_cal  # noqa: E402

IN = 0.0254


def describe(result: Result) -> None:
    if result.error:
        print(f"BUILD FAILED: {result.error}")
        return

    rocket, flight = result.rocket, result.flight
    design = result.design

    print("FREE VARIABLES")
    print(f"  stage 1 propellant     {design.m_prop_1:8.3f} kg")
    print(f"  stage 2 propellant     {design.m_prop_2:8.3f} kg")
    print(f"  coast to S2 ignition   {design.t_ignite_s:8.2f} s")
    print(f"  stage 1 extra length   {design.extra_length_1_m:8.3f} m")
    print(f"  stage 2 extra length   {design.extra_length_2_m:8.3f} m")
    print(f"  stage 1 structure mass {design.structure_mass_1_kg:8.3f} kg")
    print(f"  stage 2 structure mass {design.structure_mass_2_kg:8.3f} kg")

    print("\nVEHICLE")
    for stage in (rocket.stage_1, rocket.stage_2):
        print(f"  {stage.name}: motor {stage.motor.length_m / IN:5.1f} in, "
              f"body {stage.body_length_m / IN:5.1f} in, "
              f"dry {stage.dry_mass_kg:6.2f} kg, "
              f"thrust {stage.motor.thrust(101325.0) / 1000:5.2f} kN, "
              f"eps {stage.motor.expansion_ratio:5.2f}")
    print(f"  total length {rocket.total_length_m / IN:6.1f} in   "
          f"dry {rocket.total_dry_mass_kg:6.2f} kg   "
          f"liftoff {rocket.liftoff_mass_kg:6.2f} kg")

    print(f"\nAPOGEE   {result.apogee_m / 1000.0:8.2f} km"
          f"   (max Mach {flight.max_mach:.2f}, max {flight.max_acceleration_g:.1f} g,"
          f" rail exit {flight.rail_exit_velocity_m_s:.1f} m/s)")
    print(f"  separation {flight.separation_altitude_m / 1000:.2f} km   "
          f"S2 ignition {flight.ignition_altitude_m / 1000:.2f} km   "
          f"S2 burnout {flight.burnout_2_altitude_m / 1000:.2f} km")

    print("\nCHECKS")
    mark = {"pass": "ok  ", "fail": "FAIL", "skipped": "--  ", "not_implemented": "??  "}
    for check in result.checks:
        value = "n/a" if check.value is None else f"{check.value:9.4g}"
        window = ""
        if check.limit_low is not None and check.limit_high is not None:
            window = f"[{check.limit_low:g}, {check.limit_high:g}]"
        elif check.limit_low is not None:
            window = f">= {check.limit_low:g}"
        elif check.limit_high is not None:
            window = f"<= {check.limit_high:g}"
        print(f"  {mark[check.status]} {check.name:28s}{value}  {window:16s}{check.units}")

    if result.failed:
        print("\nFAILED")
        for check in result.failed:
            print(f"  {check.name}: {check.value:.4g} {check.units} outside "
                  f"[{check.limit_low}, {check.limit_high}]")

    blocked = result.blocked
    if blocked:
        print(f"\nNOT YET CHECKABLE ({len(blocked)}) -- these are not passes:")
        for check in blocked:
            print(f"  {check.name}: {check.note}")

    print(f"\nconstraints satisfied: {result.constraints_satisfied}"
          f"   fully feasible: {result.feasible}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS 1 DOF apogee optimizer")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("-n", "--levels", type=int, default=3)
    parser.add_argument("--refine", action="store_true", help="polish the best sweep point")
    args = parser.parse_args()

    config = Config()
    m_max = max_propellant_mass(config.limits.motor_length_max_m)
    print(f"Motor length cap {config.limits.motor_length_max_m / IN:.0f} in "
          f"-> max propellant {m_max:.3f} kg per motor")
    print(f"Stability window {config.limits.static_margin_min_cal}"
          f" - {config.limits.static_margin_max_cal} calibers\n")

    if not args.sweep:
        design = Design(
            m_prop_1=m_max, m_prop_2=m_max, t_ignite_s=10.0,
            structure_mass_1_kg=11.0, structure_mass_2_kg=10.0,
        )
        describe(evaluate(design, config))
        return

    grid = grid_from_bounds(config, args.levels)
    total = 1
    for values in grid.values():
        total *= len(values)
    print(f"Sweeping {total} designs ({args.levels} levels per free variable) ...")
    result = sweep(grid, config, progress=True)

    rows = [r for r in result.rows() if r["apogee_km"] > 0]
    satisfied = sum(1 for r in result.results if r.constraints_satisfied)
    print(f"\n  evaluated {len(result.results)}   constraints satisfied {satisfied}")

    print("\n  SENSITIVITY (apogee span across each variable)")
    for name, span in sorted(result.sensitivity().items(), key=lambda kv: -kv[1]):
        print(f"    {name:24s} {span:7.2f} km")

    print("\n  TOP 10 WITH NO FAILED CHECKS")
    ok_rows = sorted((r for r in rows if r["ok"]), key=lambda r: -r["apogee_km"])
    if ok_rows:
        print(f"    {'apogee':>8}{'mp1':>7}{'mp2':>7}{'t_ig':>6}"
              f"{'str1':>7}{'str2':>7}{'dry':>7}{'SM_stk':>8}{'SM_s2':>7}")
        for row in ok_rows[:10]:
            print(f"    {row['apogee_km']:8.2f}{row['m_prop_1']:7.2f}{row['m_prop_2']:7.2f}"
                  f"{row['t_ignite_s']:6.1f}{row['structure_mass_1_kg']:7.2f}"
                  f"{row['structure_mass_2_kg']:7.2f}{row['dry_kg']:7.2f}"
                  f"{row.get('stability_stack', float('nan')):8.2f}"
                  f"{row.get('stability_stage2', float('nan')):7.2f}")
    else:
        print("    none -- every design failed a constraint")
        worst = result.best_ignoring_constraints
        if worst:
            print(f"    best ignoring constraints: {worst.apogee_m / 1000:.2f} km, "
                  f"failing {', '.join(c.name for c in worst.failed)}")

    best = result.best
    if best:
        print("\n" + "=" * 70)
        print("BEST DESIGN WITH NO FAILED CHECKS  (re-flown at the accurate timestep)")
        print("=" * 70)
        describe(evaluate(best.design, config))

        if args.refine:
            print("\n" + "=" * 70)
            print("REFINED")
            print("=" * 70)
            describe(refine(best.design, config))


if __name__ == "__main__":
    main()
