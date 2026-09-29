# ATLAS apogee optimizer

Preliminary **1 DOF** optimizer for **ATLAS** — two stages, 4 in / 4 in, one SRAD
**Kraken** motor per stage. It searches the design space for the highest apogee
that still holds static margin inside an allowed window.

`kraken.py` is imported, never modified.

## Running it

Open PowerShell, then (the quotes matter — the path has a space in it):

```powershell
cd "C:\Users\kritt\Desktop\Controls 2027\Optimizer"
py run.py
```

| Command | Designs | Time |
|---|---|---|
| `py run.py` | 1 (baseline) | 0.4 s |
| `py run.py --sweep -n 3` | 243 | 6 s |
| `py run.py --sweep -n 4` | 1024 | 27 s |
| `py run.py --sweep -n 5` | 3125 | 90 s |
| `py run.py --sweep --refine` | + Nelder-Mead polish on the winner | +a few s |

`-n` is levels per free variable, so cost grows as n⁵ (five variables are unpinned
by default). Start at `-n 3` to check it works, then go up.

**Use `py`, not `python`.** On this machine `python` hits the Microsoft Store alias
and fails; `py` is the real launcher. No virtual environment or install step is
needed — Python 3.14.3 with numpy 2.4.2 and scipy 1.17.0 is already present.

All configuration is in `src/atlas/params.py`.

---

## How the optimizer works

**One design at a time.** `evaluate()` runs a five-step pipeline:

```
Design          7 free variables (propellant, timing, length, mass)
  |
  v  build()          resolve motors via the Kraken adapter, then body
  |                   lengths, structure masses, and CG
Rocket
  |
  v  simulate()       fly it vertically, 1 DOF, phase by phase:
  |                   burn_1 -> coast_stack -> coast_s2 -> burn_2 -> coast_final
Flight              apogee + full (altitude, velocity, Mach, q) history
  |
  v  run_all()        check every constraint against that history
  |
Result              apogee AND every check, pass or fail, with its limit
```

`evaluate()` never raises on a bad design. A design that busts the motor length cap
or flies unstably comes back as a `Result` whose checks say so — exceptions are
reserved for real bugs. Failed constraints are never folded into a score.

**Then across the space.** `sweep()` walks a grid of designs, evaluating each one
and keeping them all. Designs with any failed check are discarded; the survivors
are sorted by apogee and the best is reported. `Sweep.sensitivity()` returns the
apogee span each variable accounts for.

**Why a grid rather than a gradient-free optimizer.** The model is fast and there
are few variables, so a grid is affordable — and it returns the whole landscape and
the shape of the feasible region, not just one point. That matters here because it
shows *how sharp* the optimum is: a flat peak is slack you can spend on
manufacturability or margin. `refine()` adds a Nelder-Mead polish when you want the
local best rather than the grid's nearest node.

**Screening cost.** The sweep flies at dt = 0.05 s and the winner is re-flown at
dt = 0.02 s. Against dt = 0.005 s those cost 0.33% and 0.11% of apogee — far inside
the drag model's own uncertainty, so screening coarse is free accuracy.

---

## Free variables

| Variable | What it is |
|---|---|
| `m_prop_1`, `m_prop_2` | Propellant mass per motor. Motor length, thrust and throat area all follow from it. |
| `t_ignite_s` | Coast from separation to stage 2 ignition. |
| `extra_length_1_m`, `_2_m` | Body length above the motor + 30 in minimum. Pinned to 0 by default. |
| `structure_mass_1_kg`, `_2_kg` | Dry structure mass per stage, excluding the motor. |

Fin geometry is **not** free — it is fixed at the baseline in `params.py`.

Structure mass and length being free makes this partly a trade study: longer and
heavier are always worse for apogee, so those pin to their lower bounds. The useful
output is the landscape — what apogee a given mass buys — not the single best point.

## Current result

5-level sweep, 3125 designs: best is **75.3 km** at 10.34 kg propellant per motor,
15 s coast, 8.0 / 7.0 kg structure. Static margin 1.55 cal (stack), 1.05 cal
(sustainer). 268 designs satisfy every checkable constraint.

The optimum is **not** at the 12.45 kg propellant cap. Heavier motors shift CG and
push static margin out of the 1–2.3 caliber window, so stability actively trades
against propellant instead of merely filtering.

| Variable | Apogee span |
|---|---|
| `m_prop_1` | 83.3 km |
| `m_prop_2` | 80.6 km |
| `structure_mass_2_kg` | 48.5 km |
| `t_ignite_s` | 26.2 km |
| `structure_mass_1_kg` | 17.5 km |

Sustainer structure mass matters roughly 3× more than booster structure mass —
worth knowing before trimming anything.

## Layout

```
kraken.py                 provisional motor model — DO NOT EDIT
run.py                    entry point
src/atlas/
  params.py               all configuration: limits, bounds, baseline fins
  atmosphere.py           US Standard 1976 + gravity
  motor.py                Kraken adapter, frozen nozzle
  rocket.py               Design (free vars) -> Rocket (masses, CG, geometry)
  aero.py                 CD(Mach), Barrowman CP
  flight.py               1 DOF simulation
  checks.py               stability, flutter plumbing, sanity limits
  optimize.py             evaluate, sweep, refine
tools/                    Fusion mass-property exporter and export fixer
inputs/vehicle_baseline.yaml   Orpheus & Eurydice measurements (reference only)
vehicle_sizing_handoff.md      study scope and open questions
```

---

## What each function does

### `params.py`
`Config` holds everything: body diameter, nose, rail, nozzle design altitudes,
plus `Limits` (pass/fail criteria) and `Bounds` (free variable ranges). `Fins`
describes one fin set; `mid_chord_sweep_m` is the sweep measure Barrowman needs.

### `atmosphere.py`
`air_at(altitude, dT)` returns pressure, density, temperature and sound speed from
US Standard 1976 — analytic, valid to 86 km, vacuum above (density there is
~7e-6 kg/m³). `Air.mach(v)` returns 0 in vacuum where Mach is undefined.
`gravity(h)` is inverse-square — g is 3% lower at 100 km.

### `motor.py` — the Kraken adapter
`kraken.py` sets exit pressure equal to ambient at every altitude, i.e. a perfectly
expanded nozzle everywhere. That implies an 11.3 in exit diameter on a 4 in motor at
30 km and over-predicts thrust ~13% high up — which an apogee-maximising optimizer
will find and exploit. So:

| Function | Purpose |
|---|---|
| `build_motor(m_prop, design_alt)` | Calls Kraken once, freezes throat and exit area, clamps exit area to the casing bore, recovers true exit pressure if that clamp binds. |
| `Motor.thrust(p_ambient)` | `F = ṁ·u_e + (p_e − p_a)·A_e` from the frozen geometry. |
| `Motor.mass(t)` | Mass after `t` seconds of burn. |
| `Motor.exit_pressure_ratio(p_a)` | `p_e/p_a`; below ~0.4 the flow separates. |
| `max_propellant_mass(max_len)` | Inverts Kraken's length build-up → 12.45 kg at 52 in. |

**The two stages cannot share a nozzle.** The grain is a core burner, so burn area
scales with grain length; a common throat would drive chamber pressure to hundreds
of bar. Throat area is derived per stage, never chosen.

### `rocket.py`
`Design` is the free variable vector. `build(design, config)` resolves it into a
`Rocket` — motors, body lengths (motor + 30 in minimum), masses.
`Rocket.cg_from_nose_m(stage2_alone=...)` gives the two CG values the stability
check needs; 1 DOF needs no CG for the trajectory itself, only mass vs. time.
`fin_root_le_from_nose_m()` locates the fins for Barrowman.

### `aero.py`
`drag_coefficient(mach, stage2_alone, powered)` interpolates the CD table and
`drag_force_n(...)` applies it. `centre_of_pressure(...)` is Barrowman — nose plus
fins with body interference, Prandtl-Glauert corrected.

> The CD table is a generic slender-rocket **placeholder**, not ATLAS data. Drag is
> the largest single driver of predicted apogee. Replace it with a RASAero II
> export — it handles supersonic properly and gives booster and sustainer
> configurations separately.

### `flight.py`
`simulate(rocket, ...)` — vertical 1 DOF, state `(altitude, velocity)`, fixed-step
RK4 per phase, apogee interpolated so the answer doesn't depend on landing on a step
boundary. Returns a `Flight` with the sample history plus summary values;
`Flight.max_mach_in(phases)` gives each fin set its own worst case.

### `checks.py`
Each check returns a `Check` with value, limit and status. `skipped` means no limit
is set; `not_implemented` means the plumbing exists but the model doesn't. **Neither
counts as a pass.**

| Function | Checks |
|---|---|
| `static_margin_cal(rocket, ...)` | Static margin in calibers |
| `check_stability` | Margin inside the window, stack and sustainer |
| `check_flutter` | Flutter margin per fin set — **plumbing only** |
| `check_basics` | Motor length, nozzle bore, flow separation, rail exit, drag-data validity |

**Stability is evaluated at Mach 0.3**, where Barrowman is valid and where the
criterion is conventionally specified. Supersonic margin reports
`not_implemented` rather than guessing: Barrowman with Prandtl-Glauert at Mach 4.5
scales fin lift by 0.23 and puts CP ahead of CG, predicting an unstable vehicle.
That's a model artefact, and a fake number is worse than an admitted gap.

**Flutter is structure without a model.** Implement the `FlutterModel` protocol —
one method, `flutter_velocity_m_s(fins, altitude, mach, density)` — and pass it to
`check_flutter`; nothing else changes. `NoFlutterModel` returns `None`, so the check
reports `not_implemented`. The approved methods are the 1/k Theodorsen function
(NACA TR685/496) subsonic and Kearns above Mach 1, both needing fin modal
frequencies. `Literature/Header.txt` rules out NACA TN4197 and Bennett's correction,
which are the closed-form equations most tools use (RocketPy included), so there is
deliberately no easy substitute here.

### `optimize.py`
`evaluate(design, config)` builds, flies and checks one design. `Result.feasible`
requires every check to pass; `constraints_satisfied` tolerates blocked checks,
which is the useful signal while flutter is unimplemented.

`sweep(grid, config)` walks the grid keeping every result. `Sweep.best` is the
highest apogee with no failed check. `grid_from_bounds(config, n)` builds a default
grid — it grows as nᵏ, so 5 levels over 5 unpinned variables is 3125 designs.
`refine(seed)` is a Nelder-Mead polish; failed constraints are penalised so the
simplex can cross infeasible ground, but the returned `Result` carries the real
check record.

---

## Known weaknesses

**The vehicle is ~41 calibers long, so static margin is hypersensitive to fin
span.** At the baseline, 1.6 in of semi-span gives −2.1 cal (unstable) and 2.3 in
gives +4.0 cal (well overstable). The usable band is a few tenths of an inch.
Baseline spans were chosen to land mid-window; real fin design will change them, and
fin geometry will dominate this constraint once it becomes a free variable.

**Fin planforms are estimates.** Chord and thickness come from the OE CAD, but a
bounding box cannot separate root chord from tip chord from sweep.

**Supersonic CP and fin flutter are both unmodelled** — see `checks.py` above.

**Mass inputs are OE-derived, not ATLAS.** OE is a stepped multi-diameter prototype
(6.17 in booster, 5.49 / 4.02 in sustainer, 5.52 in nose) while ATLAS is 4 in to
4 in, so OE numbers are a reference. OE's export also carried a broken material
(`PLA (orange)`, density 1e6× too large on eight Wago lever-nut bodies) inflating
its mass from 34.8 kg to 1153.5 kg. `tools/fix_mass_properties_export.py` corrects
an export; the authoritative fix is the Fusion material definition.

## What would move the answer most

1. **Real CD data** (RASAero II) — largest lever on apogee
2. **Real ATLAS structure masses** — currently a free variable with guessed bounds
3. **Fin modal frequencies** (Fusion Simulation modal study) — unblocks flutter
4. **A supersonic CP method** — unblocks the high-Mach stability check
5. **A measured Kraken thrust curve** — replaces the neutral-burn assumption, which
   under-predicts peak loads
