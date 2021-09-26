"""Implement the protocols and classes to analyze 1D Bose gases."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
from attr import dataclass

from gspits import Partition as Mesh

__all__ = [
    "BlochState",
    "ExternalPotential",
    "Hamiltonian",
    "State",
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


@dataclass(frozen=True, slots=True)
class BlochState(State):
    r"""Represent a Bloch state.

    A Bloch state :math:`\psi(z)` with  wave vector :math:`k` is a wave
    function of the form

    :math:`\psi(z) = \phi(z) e^{i k z},`

    where :math:`\phi(z)` is a periodic function with the same periodicity
    :math:`L` as the Hamiltonian, i.e., :math:`\phi(z + L) = \phi(z)`.
    """

    __slots__ = ()

    # The domain mesh.
    mesh: Mesh

    # Array with the wave function values at the mesh points.
    wave_func: np.ndarray

    # Lattice wave vector of this state.
    wave_vector: float

    @property
    def periodic_component(self) -> State:
        r"""Return the periodic component of this Bloch state.

        The periodic component of a Bloch state :math:`\psi(z)` with
        wave vector :math:`k` is the periodic function :math:`\phi(z)`
        that fulfills the relation

        :math:`\psi(z) = \phi(z) e^{i k z}.`

        Therefore, :math:`\phi(z) = \psi(z) e^{-i k z}`.
        """
        wave_func = self.wave_func * np.exp(
            -1j * self.wave_vector * self.mesh.array
        )
        return State(mesh=self.mesh, wave_func=wave_func)

    @classmethod
    def plane_wave(cls, mesh: Mesh, wave_vector: float):
        r"""Build a normalized plane wave.

        A plane wave with wave vector :math:`k` is a special case of Bloch
        state whose periodic component is the constant function

        :math:`\phi(z) = \frac{1}{\sqrt{L}},`

        where :math:`L` is the domain size where the function is defined,
        so :math:`\psi(z) = \phi(z) e^{i k z}` is normalized.
        """
        mesh_array = mesh.array
        domain_extent = mesh.upper_bound - mesh.lower_bound
        wave_func = np.exp(1j * wave_vector * mesh_array) / np.sqrt(
            domain_extent
        )
        return cls(mesh=mesh, wave_func=wave_func, wave_vector=wave_vector)
