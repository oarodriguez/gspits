"""Collection of common one-dimensional external potentials."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
from attr import dataclass
from numba import njit

__all__ = [
    "ExternalPotential",
    "HarmonicOscillator",
    "SupportsExternalPotential",
    "external_potential",
]


@runtime_checkable
class ExternalPotential(Protocol):
    """Define the basic structure of an external potential function."""

    __slots__ = ()

    @abstractmethod
    def __call__(self, domain_mesh: np.ndarray) -> np.ndarray:
        """Evaluate an external potential on a mesh.

        :param domain_mesh: A one-dimensional NumPy array representing a
            domain mesh.
        """
        raise NotImplementedError


@runtime_checkable
class SupportsExternalPotential(Protocol):
    """Define the basic structure of an external potential."""

    __slots__ = ()

    @abstractmethod
    def __external_potential__(self) -> ExternalPotential:
        """Get a function that evaluate the external potential."""
        raise NotImplementedError


def external_potential(obj: SupportsExternalPotential):
    """Get the callable that evaluates the potential from an object.

    :param obj: An object that implements the `SupportsExternalPotential`
                protocol.
    :return: The callable object that evaluates the external potential as
             a function of the position.
    """
    return obj.__external_potential__()


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
