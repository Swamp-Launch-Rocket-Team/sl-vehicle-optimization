# Vehicle sizing study — handoff

**Status: optimizer built and running.** See `README.md` for how it works and what
each function does. This document covers scope, what has been resolved, and what is
still open.

## Goal

Compare two-stage vehicle concepts for predicted apogee and report whether each
meets the team's constraints. The optimizer maximises apogee subject to a static
margin window; there is no altitude target.

This is a real-flight project. **Simulation results are not flight approval.** Every
check in this study is screening only — a screening pass means a candidate is worth
spending higher-fidelity effort on, nothing more.

## Vehicle identity

**ATLAS** is the vehicle being sized: two stages, **4 in to 4 in**, one Kraken motor
per stage with different propellant masses.

**Orpheus & Eurydice (OE)** is the two-stage prototype for ATLAS and the source of
all mass and geometry inputs. OE is *not* 4 in to 4 in — its CAD measures a 6.17 in
booster, 5.49 in and 4.02 in sustainer sections, and a 5.52 in nosecone. OE numbers
are therefore a **reference, area-scaled to 4 in where OE differed**, not ATLAS
measurements. Never report an OE-derived figure as ATLAS performance, and never
report a study output as OE performance.

## Resolved since the original handoff

| Item | Resolution |
|---|---|
| Motor length limit | **52 in applies to both motors.** Implies ≤ 12.45 kg propellant each. |
| Control surfaces | **All four stage 2 fins** carry them. Stage 1 has none. |
| Fin count | Fixed at four per stage. |
| Model fidelity | **1 DOF**, not 3 DOF. Vertical flight only. |
| Objective | Maximise apogee. No altitude target. |
| Stability criterion | **1.0 – 2.3 calibers.** |
| Shared nozzle? | **Impossible.** The grain is a core burner, so burn area scales with grain length; a common throat would drive chamber pressure past 600 bar across the design range. Throat area is derived per stage. |
| Stage 2 nozzle expansion | A fixed setting, not a design variable — the benefit saturates once the casing bore caps the expansion ratio (~10 km design altitude). Worth about +5 km over a sea-level-matched nozzle. |
| Code layout | `src/atlas/`, eight modules. Configuration is dataclasses in `params.py`, not YAML. |

## Free variables

Motor propellant mass per stage, coast time to stage 2 ignition, body length above
the motor + 30 in minimum, and structure mass per stage. **Fin geometry is not a
free variable yet.**

## `kraken.py` — read, never edit

Takes propellant mass and returns motor mass, length and one thrust estimate. Its
burn history and predictions are **unvalidated while Kraken is being built**.
`src/atlas/motor.py` adapts it without modification. Documented assumptions:

- **Burn time is hard-coded at 9.78 s**, and this is self-consistent: grain OD and
  ID are constants, so web thickness is constant, and a radially burning grain's
  burn time is web over burn rate regardless of length (implied rate 3.57 mm/s).
- **Perfect expansion at every altitude.** This is the one thing the adapter must
  correct — left alone it implies an 11.3 in exit diameter on a 4 in motor at 30 km
  and over-predicts thrust ~13% high up, which an optimizer will exploit.
- **Constant mass flow (neutral burn).** A single tubular core burner going from
  1.0 in to 3.75 in core diameter is strongly progressive in reality, so true thrust
  would rise toward burnout. Total impulse is unaffected, but peak acceleration and
  peak dynamic pressure would occur later and higher than modelled — which matters
  for load and flutter checks.
- Confirm whether the team's phrase "motor mass" means this propellant-mass input.

## Open questions for the team

**Blocking a meaningful apogee number:**

1. **CD vs Mach** for stack and sustainer, power-on and power-off. The current table
   is a generic placeholder and drag is the largest single driver of apogee. RASAero
   II is the right tool — free, built for supersonic high-power, exports booster and
   sustainer separately. The sweep reaches Mach 4.5, where OpenRocket's correlations
   are not credible.
2. **Real ATLAS structure masses.** Currently a free variable with guessed bounds.
3. **Launch site elevation and rail length.** Placeholders at 0 m and 18 ft.

**Blocking specific checks:**

4. **Fin modal frequencies** (bending and torsion) — unblocks flutter. A Fusion
   Simulation modal study on a single fin with realistic root fixity is the cheapest
   path.
5. **Approved pass/fail values** for max acceleration and max Mach. Both currently
   report `skipped`. Max acceleration is likely to bind: the sweep predicts 15–20 g,
   and many COTS altimeters and IMUs saturate at 16–24 g.
6. **Grain configuration and a predicted thrust curve** from propulsion — single
   tubular or BATES segments? This changes the load predictions, not the apogee.
7. **Fin planform** — root chord, tip chord and sweep separately. A CAD bounding box
   cannot separate them, so the current values are estimates reconciled to OE.

## Not in scope, and why

1 DOF cannot assess attitude, weathercocking, rail tip-off, upper-stage pitch-back
at separation, roll behaviour, or control surface authority. Those need 6 DOF and a
separate separation analysis. Supersonic static margin is also excluded: Barrowman
with a Prandtl-Glauert correction returns a spurious negative margin above about
Mach 1.2, so a supersonic CP method is needed before that condition can be judged.

## Known data issue

OE's Fusion export reported 1153.5 kg because the custom material `PLA (orange)` has
a density 1e6× too large, applied to eight Wago 221-2411 lever-nut bodies in the
Stage 2 Avionics Bay. Corrected mass is **34.80 kg** (stage 1 12.68 kg, stage 2
22.12 kg including a 15.0 kg Aerotech placeholder standing in for Kraken).

`tools/fix_mass_properties_export.py` corrects an export, but **the authoritative
fix is the Fusion material definition** — the Wago is a linked external component,
so the material lives in its own document, not the parent assembly. Also worth
fixing: `Wires` has zero mass, and two components (TeleMega, Marfik DAQ) sit far off
the vehicle axis.
