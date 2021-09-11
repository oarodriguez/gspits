"""Implement the protocols and classes to analyze 1D Bose gases."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
from attr import dataclass

from gspits import Mesh

__all__ = [
    "ExternalPotential",
    "Hamiltonian",
    "State",
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


@runtime_checkable
class Hamiltonian(Protocol):
    """Define the basic attributes and methods of a Hamiltonian."""

    __slots__ = ()

    @property
    @abstractmethod
    def interaction_factor(self) -> float:
        """Gas interaction factor."""
        raise NotImplementedError

    @property
    @abstractmethod
    def external_potential(self) -> ExternalPotential:
        """External potential function."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class State:
    """Represent a quantum state."""

    __slots__ = ()

    # The domain mesh.
    mesh: Mesh

    # Array with the wave function values at the mesh points.
    wave_func: np.ndarray

    @property
    def norm(self):
        """Get the state discrete norm.

        The discrete norm is equivalent to use a Riemann sum to approximate
        the exact wave function norm.
        TODO: Can we replace the Riemann sum with the trapezoidal rule?
        """
        wave_func = self.wave_func
        dx = self.mesh.step_size
        return dx * np.sum(np.abs(wave_func) ** 2)
