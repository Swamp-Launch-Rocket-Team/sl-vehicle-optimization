"""ATLAS apogee optimizer -- preliminary 1 DOF.

Two stages, 4 in / 4 in, one SRAD Kraken motor per stage. Maximise apogee subject
to a static margin window. kraken.py is imported, never modified.
"""

from .atmosphere import Air, air_at, gravity
from .checks import Check, FlutterModel, NoFlutterModel, run_all, static_margin_cal
from .flight import Flight, simulate
from .motor import Motor, build_motor, max_propellant_mass
from .optimize import Result, Sweep, evaluate, grid_from_bounds, refine, sweep
from .params import Bounds, Config, Fins, Limits
from .rocket import VARIABLES, Design, Rocket, build

__all__ = [
    "Air", "air_at", "gravity",
    "Check", "FlutterModel", "NoFlutterModel", "run_all", "static_margin_cal",
    "Flight", "simulate",
    "Motor", "build_motor", "max_propellant_mass",
    "Result", "Sweep", "evaluate", "grid_from_bounds", "refine", "sweep",
    "Bounds", "Config", "Fins", "Limits",
    "VARIABLES", "Design", "Rocket", "build",
]
