"""Hamiltonians for one-dimensional, single-species Bose gases."""
from math import sqrt

import numpy as np
from attr import dataclass
from numba import njit
from numpy import cos, pi

from gspits import Mesh

from .system import Hamiltonian, State

__all__ = [
    "HTHamiltonian",
    "MRHamiltonian",
    "OLHTHamiltonian",
    "plane_wave_state",
]


def plane_wave_state(mesh: Mesh):
    """Build a normalized plane wave."""
    domain_mesh = mesh.array
    domain_extent = domain_mesh.max() - domain_mesh.min()
    wave_vector = 2 * pi / domain_extent
    wave_func = np.exp(1j * wave_vector * domain_mesh) / sqrt(domain_extent)
    return State(mesh=mesh, wave_func=wave_func)


@dataclass(frozen=True)
class HTHamiltonian(Hamiltonian):
    """Harmonic Trap Hamiltonian.

    Represent a Bose gas within a 1D harmonic trap.
    """

    # Trap angular frequency.
    freq: float

    # Pairwise interaction energy.
    int_energy: float

    def __attrs_post_init__(self):
        """Post-initialization checks."""
        if np.nan in (self.freq, self.int_energy):
            raise ValueError
        if not self.freq > 0:
            raise ValueError

    @property
    def trap_size(self):
        """Characteristic size of the harmonic trap."""
        return sqrt(1 / self.freq)

    @property
    def int_factor(self) -> float:
        """Gas interaction factor."""
        return self.int_energy

    @property
    def external_potential(self):
        """External potential function."""
        freq = self.freq

        @njit
        def _ht_potential(
            domain_mesh: np.ndarray,
        ) -> np.ndarray:  # pragma: no cover
            """Evaluate the harmonic trap potential in the mesh."""
            return 0.5 * (freq * domain_mesh) ** 2

        return _ht_potential

    @staticmethod
    def plane_wave_state(mesh: Mesh):
        """Build a normalized plane wave."""
        return plane_wave_state(mesh)

    def gaussian_state(self, mesh: Mesh):
        """Build a normalized Gaussian state.

        :param mesh: A mesh where the state will be defined.
        :return: A corresponding :class:`State` instance.
        """
        freq = self.freq
        domain_mesh = mesh.array
        wave_func = (freq / pi) ** 0.25 * np.exp(-freq * domain_mesh ** 2 / 2)
        return State(mesh=mesh, wave_func=wave_func)


@dataclass(frozen=True)
class OLHTHamiltonian(HTHamiltonian):
    """Optical Lattice plus Harmonic Trap Hamiltonian.

    Represent a Bose gas within a 1D optical lattice plus an harmonic trap.
    """

    # Lattice depth.
    lattice_depth: float

    # Trap angular frequency.
    freq: float

    # Lattice wavelength.
    wavelength: float

    # Pairwise interaction energy.
    int_energy: float

    def __attrs_post_init__(self):
        """Post-initialization checks."""
        if np.nan in (
            self.lattice_depth,
            self.freq,
            self.wavelength,
            self.int_energy,
        ):
            raise ValueError
        if not self.freq > 0:
            raise ValueError
        if not self.wavelength > 0:
            raise ValueError

    @property
    def external_potential(self):
        """External potential function."""
        lattice_depth = self.lattice_depth
        freq = self.freq
        wavelength = self.wavelength

        @njit
        def _olht_potential(
            domain_mesh: np.ndarray,
        ) -> np.ndarray:  # pragma: no cover
            """Evaluate the external potential in the mesh."""
            return (
                0.5 * (freq * domain_mesh) ** 2
                + lattice_depth * cos(2 * pi * domain_mesh / wavelength) ** 2
            )

        return _olht_potential


@dataclass(frozen=True)
class MRHamiltonian(Hamiltonian):
    """Hamiltonian of a Bose gas in Multi-Rods.

    Represent a Bose gas within a 1D multi-rod structure.
    """

    # Lattice depth.
    lattice_depth: float

    # Lattice period.
    lattice_period: float

    # Barrier width.
    barrier_width: float

    # Pairwise interaction energy.
    int_energy: float

    def __attrs_post_init__(self):
        """Post-initialization checks."""
        if np.nan in (
            self.lattice_depth,
            self.barrier_width,
            self.lattice_period,
            self.int_energy,
        ):
            raise ValueError
        if not self.barrier_width > 0:
            raise ValueError
        if not self.lattice_period > 0:
            raise ValueError
        if not self.barrier_width <= self.lattice_period:
            raise ValueError

    @property
    def well_width(self):
        """Width between two consecutive barriers."""
        return self.lattice_period - self.barrier_width

    @property
    def int_factor(self) -> float:
        """Gas interaction factor."""
        return self.int_energy

    @property
    def external_potential(self):
        """External potential function."""
        lattice_depth = self.lattice_depth
        barrier_width = self.barrier_width
        lattice_period = self.lattice_period

        @njit
        def _mr_potential(
            domain_mesh: np.ndarray,
        ) -> np.ndarray:  # pragma: no cover
            """Evaluate a multi-rods potential over a mesh."""
            shifted_domain_mesh = domain_mesh + lattice_period / 2
            _, shifted_domain_offset = np.divmod(
                shifted_domain_mesh, lattice_period
            )
            return np.where(
                np.fabs(shifted_domain_offset - lattice_period / 2)
                <= barrier_width / 2,
                lattice_depth,
                0,
            )

        return _mr_potential

    @staticmethod
    def plane_wave_state(mesh: Mesh):
        """Build a normalized plane wave."""
        # We define the plane wave in the same way.
        return OLHTHamiltonian.plane_wave_state(mesh)
