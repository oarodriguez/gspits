"""Hamiltonians for one-dimensional, single-species Bose gases."""
from math import sqrt
from typing import Sequence

import numpy as np
from attr import dataclass
from numba import njit
from numpy import cos, pi

from gspits import Mesh
from gspits.one_dim import Hamiltonian, State

__all__ = [
    "DCHamiltonian",
    "DeltaSpec",
    "HTHamiltonian",
    "MRHamiltonian",
    "OLHTHamiltonian",
    "SuperDCHamiltonian",
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
    interaction_strength: float

    def __attrs_post_init__(self):
        """Post-initialization checks."""
        if np.nan in (self.freq, self.interaction_strength):
            raise ValueError
        if not self.freq > 0:
            raise ValueError

    @property
    def trap_size(self):
        """Characteristic size of the harmonic trap."""
        return sqrt(1 / self.freq)

    @property
    def interaction_factor(self) -> float:
        """Gas interaction factor."""
        return self.interaction_strength

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
    interaction_strength: float

    def __attrs_post_init__(self):
        """Post-initialization checks."""
        if np.nan in (
            self.lattice_depth,
            self.freq,
            self.wavelength,
            self.interaction_strength,
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
    interaction_strength: float

    def __attrs_post_init__(self):
        """Post-initialization checks."""
        if np.nan in (
            self.lattice_depth,
            self.barrier_width,
            self.lattice_period,
            self.interaction_strength,
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
    def interaction_factor(self) -> float:
        """Gas interaction factor."""
        return self.interaction_strength

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
        return plane_wave_state(mesh)


@dataclass(frozen=True)
class DCHamiltonian(Hamiltonian):
    r"""Hamiltonian of a Bose gas in a Dirac-Comb potential.

    Represent an interacting Bose gas within a 1D, multi-layer lattice
    modeled by a Dirac-comb potential.

    The argument ``delta_strength`` is the strength of any Dirac-delta
    potential barrier, ``lattice_period`` sets the distance between two
    consecutive delta barriers and the lattice period, and the pairwise
    interaction magnitude between bosons is set by the
    ``interaction_strength`` argument.
    """

    # Delta potential strength.
    delta_strength: float

    # Lattice period.
    lattice_period: float

    # Pairwise interaction energy.
    interaction_strength: float

    def __attrs_post_init__(self):
        """Post-initialization checks."""
        if np.nan in (
            self.delta_strength,
            self.lattice_period,
            self.interaction_strength,
        ):
            raise ValueError
        if not self.lattice_period > 0:
            raise ValueError

    @property
    def interaction_factor(self) -> float:
        """Gas interaction factor."""
        return self.interaction_strength

    @property
    def external_potential(self):
        """External potential function."""
        delta_strength = self.delta_strength
        lattice_period = self.lattice_period

        def _dc_potential(
            domain_mesh: np.ndarray,
        ) -> np.ndarray:  # pragma: no cover
            """Evaluate a multi-rods potential over a mesh."""
            shifted_domain_mesh = domain_mesh + lattice_period / 2
            _, shifted_domain_offset = np.divmod(
                shifted_domain_mesh, lattice_period
            )
            barrier_width = np.diff(domain_mesh)[0]
            return np.where(
                np.fabs(shifted_domain_offset - lattice_period / 2)
                <= barrier_width / 2,
                delta_strength / barrier_width,
                0,
            )

        return _dc_potential


@dataclass(frozen=True, slots=True)
class DeltaSpec:
    """Dirac-delta potential barrier specification.

    The argument ``strength`` indicates the barrier strength, while the
    ``rel_position`` argument indicates the barrier relative position respect
    to the domain lower bound where it lies, so ``rel_position`` is
    a number that lies in the interval :math:`[0, 1]`.

    For instance, if a barrier lies in the interval :math:`[a, b]` and its
    relative position is ``rel_position = 0``, then the barrier absolute
    position is :math:`a`. If a barrier relative position is :math:`0.5`,
    then its absolute position is :math:`a + 0.5 (b - a)`.
    """

    # Delta potential strength.
    strength: float

    # Delta potential relative position.
    rel_position: float

    def __attrs_post_init__(self):
        """Post-initialization checks."""
        if np.nan in (
            self.strength,
            self.rel_position,
        ):
            raise ValueError
        if self.rel_position < 0 or self.rel_position >= 1:
            raise ValueError


@dataclass(frozen=True)
class SuperDCHamiltonian(Hamiltonian):
    r"""Hamiltonian of a Bose gas in a super-Dirac-comb potential.

    Represent an interacting Bose gas within a 1D, multi-layer super-lattice
    modeled by a super Dirac-comb potential.

    The argument `deltas_seq` indicates the strength and relative positions
    of each delta potential in the super-lattice. It is a sequence of
    `DeltaSpec` objects with a given `strength` and `rel_position`. The
    argument `lattice_period` sets the super-lattice period, while the
    pairwise interaction magnitude between bosons is set by the
    `interaction_strength` argument.
    """

    # Sequence with Dirac-delta potential strengths and locations.
    deltas_seq: Sequence[DeltaSpec]

    # Lattice period.
    lattice_period: float

    # Pairwise interaction energy.
    interaction_strength: float

    def __attrs_post_init__(self):
        """Post-initialization checks."""
        if np.nan in (
            self.deltas_seq,
            self.lattice_period,
            self.interaction_strength,
        ):
            raise ValueError
        if not self.lattice_period > 0:
            raise ValueError

        def _position(spec: DeltaSpec):
            """Get a Dirac-delta relative position."""
            return spec.rel_position

        sorted_deltas_seq = sorted(self.deltas_seq, key=_position)
        object.__setattr__(self, "deltas_seq", sorted_deltas_seq)

    @property
    def interaction_factor(self) -> float:
        """Gas interaction factor."""
        return self.interaction_strength

    @property
    def external_potential(self):
        """External potential function."""
        deltas_seq = self.deltas_seq
        lattice_period = self.lattice_period
        delta_strengths = np.array([spec.strength for spec in deltas_seq])
        delta_rel_positions = np.array(
            [spec.rel_position for spec in deltas_seq]
        )
        num_deltas = len(delta_strengths)

        @njit()
        def _dc_potential(
            domain_mesh: np.ndarray,
        ) -> np.ndarray:  # pragma: no cover
            """Evaluate a multi-rods potential over a mesh."""
            barrier_width = np.diff(domain_mesh)[0]
            shifted_domain_mesh = domain_mesh + lattice_period / 2
            _, shifted_domain_mesh_periodic = np.divmod(
                shifted_domain_mesh, lattice_period
            )

            # Find the closest delta potential that is less than the periodic
            # domain starting value.
            delta_idx = 0
            delta_position = lattice_period * (
                -0.5 + delta_rel_positions[delta_idx]
            )
            domain_mesh_periodic = (
                shifted_domain_mesh_periodic - lattice_period / 2
            )
            # Avoid a never ending cycle.
            if num_deltas > 1:
                while delta_position < domain_mesh_periodic[0]:
                    # Due to periodicity, reset the index value when it reaches
                    # the largest possible index value. Otherwise, increment it
                    # by one.
                    if delta_idx == num_deltas - 1:
                        delta_idx = 0
                    else:
                        delta_idx += 1
                    delta_position = lattice_period * (
                        -0.5 + delta_rel_positions[delta_idx]
                    )

            # Retrieve the current delta barrier strength.
            delta_strength = delta_strengths[delta_idx]
            dc_potential = np.zeros_like(domain_mesh_periodic)
            for idx, position in enumerate(domain_mesh_periodic):
                # When the current position is sufficiently close to the delta
                # barrier position, we set the potential in that point to a
                # value such that it is equivalent to a rectangular barrier
                # of width ``barrier_width`` with area ``delta_strength``,
                # simulating a delta-barrier. As the domain mesh becomes
                # finner, this procedure results in a better approximation of
                # a delta barrier.
                if np.fabs(position - delta_position) <= barrier_width / 2:
                    potential_height = delta_strength / barrier_width
                    dc_potential[idx] = potential_height
                    # Once we have set the potential value for a given delta
                    # barrier, continue with the rest of the barriers. Again, t
                    # take into account the super-lattice periodicity.
                    if delta_idx == num_deltas - 1:
                        delta_idx = 0
                    else:
                        delta_idx += 1
                    delta_strength = delta_strengths[delta_idx]
                    delta_position = lattice_period * (
                        -0.5 + delta_rel_positions[delta_idx]
                    )

            return dc_potential

        return _dc_potential
