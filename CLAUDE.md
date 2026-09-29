# ATLAS apogee optimizer

Two stages, 4 in / 4 in, one SRAD **Kraken** per stage. Preliminary **1 DOF**.
Maximise apogee subject to a static margin window. No altitude target.

Mass inputs derive from **Orpheus & Eurydice**, a multi-diameter prototype; OE
numbers are a reference, never an ATLAS figure.

## Ground rules

- **Never edit `kraken.py`.** Wrap it in `motor.py`.
- All math **SI**; convert only at IO boundaries.
- Config lives in `src/atlas/params.py`. Nothing hard-coded deeper.
- `evaluate()` **never raises on a bad design** — it returns a `Result` whose
  checks say why. Never hide failures in a score.
- `skipped` and `not_implemented` are **not** passes.
- Every check is screening; 1 DOF gives no flight-readiness evidence.

## Known model weaknesses

- `kraken.py` assumes perfect expansion at every altitude (~13% optimistic high
  up). The adapter freezes throat/exit area and adds the pressure term.
- Motors cannot share a nozzle (core burner: burn area scales with length), so
  throat is derived per stage. Stage 2 expansion saturates at the bore.
- Constant mass flow understates peak loads if the grain is progressive.
- **Supersonic CP is unmodelled.** Barrowman with Prandtl-Glauert puts CP ahead
  of CG at Mach 4.5 — an artefact. Stability is checked at Mach 0.3 only.
- **Flutter is plumbing only.** Implement `FlutterModel`. Never use NACA TN4197
  or Bennett's correction (see `Literature/Header.txt`).
- The CD table is a placeholder; drag dominates apogee. Use RASAero II.
- ~41 calibers long, so static margin is hypersensitive to fin span.
