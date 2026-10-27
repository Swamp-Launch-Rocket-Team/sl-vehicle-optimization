"""RASP .eng motor data, linear thrust, and impulse-based propellant depletion."""

from pathlib import Path

import numpy as np


class ThrustCurve:
    def __init__(self, path):
        self.path = Path(path).resolve()
        lines = [line.strip() for line in self.path.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith(";")]
        if not lines or len(lines[0].split()) != 7:
            raise ValueError(f"Invalid RASP motor header: {self.path}")
        name, diameter, length, _, propellant, loaded, manufacturer = lines[0].split()
        self.name, self.manufacturer = name, manufacturer
        self.motor_diameter = float(diameter) / 1000.0
        self.motor_length = float(length) / 1000.0
        self.p_mass, self.wet_mass = float(propellant), float(loaded)
        self.dry_mass = self.wet_mass - self.p_mass
        values = (self.motor_diameter, self.motor_length, self.p_mass, self.dry_mass)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("Motor dimensions and masses must be finite and positive.")
        points = np.array([[float(v) for v in line.split()] for line in lines[1:]])
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                or not np.isfinite(points).all() or np.any(points < 0)
                or np.any(np.diff(points[:, 0]) <= 0) or points[-1, 1] != 0):
            raise ValueError("Thrust data need increasing times, nonnegative thrust, and a zero endpoint.")
        if points[0, 0] > 0:
            points = np.vstack(([0.0, 0.0], points))
        elif points[0, 1] != 0:
            raise ValueError("Thrust at time zero must be zero.")
        self.times, self.forces = points.T
        self.burn_time = float(self.times[-1])
        increments = np.diff(self.times) * (self.forces[:-1] + self.forces[1:]) / 2
        self.impulses = np.r_[0.0, np.cumsum(increments)]
        self.total_impulse = float(self.impulses[-1])
        if self.total_impulse <= 0:
            raise ValueError("Motor total impulse must be positive.")

    def thrust(self, elapsed_time, pressure_pa=None):
        """Seconds since ignition; use supplied thrust without pressure correction."""
        return np.interp(elapsed_time, self.times, self.forces, left=0.0, right=0.0)

    def remaining_propellant_mass(self, elapsed_time):
        elapsed = np.clip(np.asarray(elapsed_time, dtype=float), 0.0, self.burn_time)
        index = np.clip(np.searchsorted(self.times, elapsed, side="right") - 1,
                        0, len(self.times) - 2)
        dt = elapsed - self.times[index]
        slope = ((self.forces[index + 1] - self.forces[index])
                 / (self.times[index + 1] - self.times[index]))
        impulse = self.impulses[index] + self.forces[index] * dt + 0.5 * slope * dt**2
        return self.p_mass * np.clip(1.0 - impulse / self.total_impulse, 0.0, 1.0)

    def summary(self):
        return {"name": self.name, "diameter_m": self.motor_diameter,
                "length_m": self.motor_length, "wet_mass_kg": self.wet_mass,
                "dry_mass_kg": self.dry_mass, "propellant_mass_kg": self.p_mass,
                "burn_time_s": self.burn_time, "total_impulse_ns": self.total_impulse}
