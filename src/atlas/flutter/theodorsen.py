"""Theodorsen unsteady aerodynamic function.
"""

from __future__ import annotations

import numpy as np
from scipy.special import hankel2


def theodorsen(k: float | np.ndarray) -> complex | np.ndarray:
    """Evaluate Theodorsen's function C(k)."""

    values = np.asarray(k, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Provide finite reduced frequency.")
    if np.any(values < 0):
        raise ValueError("Provide nonnegative reduced frequency.")

    result = np.empty(values.shape, dtype=complex)

    zero = values == 0
    result[zero] = 1.0 + 0.0j
    nonzero = ~zero

    if np.any(nonzero):
        knz = values[nonzero]
        h0 = hankel2(0, knz)
        h1 = hankel2(1, knz)

        result[nonzero] = h1 / (h1 + 1j * h0)

    if values.ndim == 0:
        return complex(result.item())

    return result

    """
    Notes
    -----
    Returns Complex Theodorsen function evaluated at k.
    C(k) = H1^(2)(k) / [H1^(2)(k) + 1j*H0^(2)(k)]

    K is the nonnegative reduced frequency, k = omega*b/V,
    where b is the airfoil semichord.
    
    The zero-frequency limit is evaluated analytically
    to avoid the singularity of the Hankel functions at k=0.
    """