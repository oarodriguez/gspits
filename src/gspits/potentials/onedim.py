"""Hamiltonians for one-dimensional, single-species Bose gases."""

from math import sqrt

import numpy as np
from attr import dataclass
from numpy import pi

from gspits import ExternalPotential, Mesh, State

__all__ = ["HarmonicTrap"]


@dataclass(frozen=True)
class HarmonicTrap(ExternalPotential):
    """One-dimensional harmonic trap external potential.

    :param float freq:
        The trap angular frequency. It must be a non-zero, positive number.
        `NaNs` are not allowed.

    :raises ValueError: If the frequency is negative or zero.
    :raises ValueError: If the frequency is a ``nan``.
    """

    # Trap angular frequency.
    freq: float

    def __attrs_post_init__(self) -> None:
        """Post-initialization checks."""
        # Avoid NaNs.
        if np.isnan(self.freq):
            raise ValueError(
                "'nan' is not a valid value for the trap frequency ('freq')."
            )
        if not self.freq > 0:
            raise ValueError(
                "the frequency must have positive, non-zero values."
            )

        # Make transformations.
        object.__setattr__(self, "freq", float(self.freq))

    @property
    def size(self) -> float:
        """Characteristic size of the trap.

        :rtype: float
        """
        return sqrt(1 / self.freq)

    def gaussian_state(self, mesh: Mesh) -> State:
        """Build a normalized Gaussian state.

        :param Mesh mesh:
            A mesh where the state will be defined.
        :rtype: State
        """
        freq = self.freq
        (x_array,) = mesh.arrays
        wave_func = (freq / pi) ** 0.25 * np.exp(-freq * x_array ** 2 / 2)
        return State(mesh=mesh, wave_func=wave_func)

    def __call__(self, mesh: Mesh) -> np.ndarray:
        """External potential callable interface.

        :param mesh:
            A ``Mesh`` instance representing a domain mesh.
        :rtype: numpy.ndarray
        """
        freq = self.freq
        (x_array,) = mesh.arrays
        return 0.5 * (freq * x_array) ** 2
