"""Apogee objective; evaluation and caching live in the supplied function."""


class Objective:
    def __init__(self, evaluate, apogee_scale=10000.0):
        if apogee_scale <= 0:
            raise ValueError("Apogee scale must be positive.")
        self.evaluate = evaluate
        self.apogee_scale = apogee_scale

    def __call__(self, values):
        result = self.evaluate(values)
        if result["status"] != "ok":
            return 1e6
        return -result["apogee_m"] / self.apogee_scale
