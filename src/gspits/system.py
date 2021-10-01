"""Define protocols and classes that define bosonic system."""
from abc import ABCMeta, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
from attr import dataclass

from gspits import Mesh

__all__ = [
    "BlochState",
    "ExternalPotential",
    "State",
    "WaveVector",
]


@dataclass(frozen=True, slots=True)
class State:
    """Represent a quantum state.

    :param Mesh mesh:
        A spatial ``Mesh`` instance where the state is defined.
    :param numpy.ndarray wave_func:
        The state complex wave function.
    """

    # The domain mesh.
    mesh: Mesh

    # Array with the wave function values at the mesh points.
    wave_func: np.ndarray

    def __attrs_post_init__(self) -> None:
        """Post-initialization tasks."""
        if self.wave_func.shape != self.mesh.shape:
            raise ValueError

    @property
    def norm(self) -> float:
        """Get the state discrete norm.

        The discrete norm is equivalent to use a Riemann sum to approximate
        the exact wave function norm.

        :rtype: float
        """
        # TODO: Can we replace the Riemann sum with the trapezoidal rule?
        wave_func = self.wave_func
        dv = self.mesh.element_size
        return dv * float(np.sum(np.abs(wave_func) ** 2))


# BlochState attributes types.
# See bug https://github.com/python/mypy/issues/9980.
WaveVector = tuple[float, ...]  # type: ignore


@dataclass(frozen=True, slots=True)
class BlochState:
    """Represent a Bloch state.

    :param Mesh mesh:
        A spatial :py:class:`Mesh` instance where the state is defined.
    :param numpy.ndarray periodic_wave_func:
        The state complex wave function.
    :param tuple[float, ...] wave_vector:
        Lattice wave vector of the state.
    """

    # The domain mesh.
    mesh: Mesh

    # An array with the periodic-part wave function values at the mesh points.
    periodic_wave_func: np.ndarray

    # Lattice wave vector of this state.
    wave_vector: WaveVector

    def __attrs_post_init__(self) -> None:
        """Post-initialization tasks."""
        wave_vector_dimension = len(self.wave_vector)
        if wave_vector_dimension != self.mesh.dimension:
            raise ValueError
        if self.periodic_wave_func.shape != self.mesh.shape:
            raise ValueError

    @classmethod
    def plane_wave(
        cls: type["BlochState"], mesh: Mesh, wave_vector: WaveVector
    ) -> "BlochState":
        r"""Build a normalized plane wave.

        :param Mesh mesh:
            A spatial :py:class:`Mesh` instance where the plane wave is
            defined.
        :param tuple[float, ...] wave_vector:
            Lattice wave vector of the state.
        :rtype: BlochState
        """
        periodic_wave_func = np.ones(mesh.shape) / np.sqrt(mesh.size)
        return cls(
            mesh=mesh,
            periodic_wave_func=periodic_wave_func,
            wave_vector=wave_vector,
        )

    @property
    def wave_func(self) -> np.ndarray:
        """Get the state complex wave function.

        :rtype: numpy.ndarray
        """
        exp_func_arg = 0.0
        mesh_arrays = self.mesh.arrays
        for wv, mesh_array in zip(self.wave_vector, mesh_arrays):
            exp_func_arg += wv * mesh_array
        wave_func = self.periodic_wave_func * np.exp(1j * exp_func_arg)
        return wave_func

    @property
    def periodic_component(self) -> State:
        """Return the periodic component of this Bloch state.

        :rtype: State
        """
        return State(mesh=self.mesh, wave_func=self.periodic_wave_func)

    @property
    def norm(self) -> float:
        """Get the state discrete norm.

        The discrete norm is equivalent to use a Riemann sum to approximate
        the exact wave function norm.

        :rtype: float
        """
        # TODO: Can we replace the Riemann sum with the trapezoidal rule?
        wave_func = self.periodic_wave_func
        dv = self.mesh.element_size
        return dv * float(np.sum(np.abs(wave_func) ** 2))


@runtime_checkable
class ExternalPotential(Protocol, metaclass=ABCMeta):
    """Define an external potential.

    This protocol sets the signature of a callable object that serves as
    an external potential. Examples:

    - A regular function with a compatible signature.
    - An instance of a class derived from this protocol.
    """

    @abstractmethod
    def __call__(self, mesh: Mesh) -> np.ndarray:
        """External potential callable interface.

        :param Mesh mesh:
            A ``Mesh`` instance representing a domain mesh.
        :rtype: numpy.ndarray
        """
        raise NotImplementedError
