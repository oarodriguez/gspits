"""Collection of common one-dimensional external potentials."""

import numpy as np
from attr import dataclass
from numba import njit

from .system import SupportsExternalPotential

__all__ = [
    "HarmonicOscillator",
]


@dataclass(frozen=True)
class HarmonicOscillator(SupportsExternalPotential):
    """Represent an harmonic oscillator potential in 1D."""

    # Particle mass.
    mass: float

    # Trap angular frequency.
    freq: float

    def __attrs_post_init__(self):
        """Post-initialization checks."""
        if not self.mass > 0:
            raise ValueError
        if not self.freq > 0:
            raise ValueError

    def __external_potential__(self):
        """Get a callable that evaluates the harmonic potential."""
        freq = self.freq
        mass = self.mass

        @njit
        def _ho_potential(
            domain_mesh: np.ndarray,
        ) -> np.ndarray:  # pragma: no cover
            """Evaluate the harmonic potential in the mesh."""
            return 0.5 * mass * (freq * domain_mesh) ** 2

        return _ho_potential
