"""Four fixed-size stability residuals; nonnegative means feasible."""

import numpy as np


class Constraint:
    def __init__(self, evaluate, minimum=2.1, maximum=7.5):
        if not np.isfinite([minimum, maximum]).all() or minimum >= maximum:
            raise ValueError("Stability limits must be finite and minimum < maximum.")
        self.evaluate = evaluate
        self.minimum, self.maximum = minimum, maximum

    def residuals(self, result):
        if result["status"] != "ok":
            return np.full(4, -1000.0)
        residuals = []
        for phase in ("stack", "upper_stage"):
            profile = result["stability"][phase]
            residuals.extend((profile["minimum_calibers"] - self.minimum,
                              self.maximum - profile["maximum_calibers"]))
        return np.asarray(residuals)

    def __call__(self, values):
        return self.residuals(self.evaluate(values))
