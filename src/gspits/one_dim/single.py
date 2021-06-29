"""Hamiltonians for one-dimensional, single-species Bose gases."""

import numpy as np
from attr import dataclass
from numba import njit
from numpy import pi

from .system import Hamiltonian

__all__ = [
    "HTHamiltonian",
]


@dataclass(frozen=True)
class HTHamiltonian(Hamiltonian):
    """Harmonic Trap Hamiltonian.

    Represent a Bose gas within a 1D harmonic trap.
    """

    # Particle mass.
    mass: float

    # Trap angular frequency.
    freq: float

    # Scattering length.
    scat_length: float

    # Number of particles.
    num_bosons: int

    def __attrs_post_init__(self):
        """Post-initialization checks."""
        if np.nan in (self.mass, self.freq, self.scat_length):
            raise ValueError
        if not self.mass > 0:
            raise ValueError
        if not self.freq > 0:
            raise ValueError
        if not self.num_bosons > 2:
            raise ValueError

    @property
    def int_factor(self) -> float:
        """Gas interaction factor."""
        return 4 * pi * self.num_bosons * self.scat_length

    @property
    def external_potential(self):
        """External potential function."""
        freq = self.freq
        mass = self.mass

        @njit
        def _ht_potential(
            domain_mesh: np.ndarray,
        ) -> np.ndarray:  # pragma: no cover
            """Evaluate the harmonic trap potential in the mesh."""
            return 0.5 * mass * (freq * domain_mesh) ** 2

        return _ht_potential