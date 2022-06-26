"""Routines to get the ground-state using the PS method."""

import logging
from abc import ABCMeta
from collections import deque
from itertools import islice, product, zip_longest
from math import pi
from typing import Iterable, Iterator

import numpy as np
from attr import dataclass
from scipy import fft

from gspits import Mesh, TimePartition
from gspits.system import BlochState, ExternalPotential, WaveVector

# Initialize a logger for this module.
logger = logging.getLogger(__name__)

__all__ = ["PSSolver", "PSSolverState", "DFTWaveVectorsMesh"]

# The variable type we use to identify a set of FFT wave vectors.
# See bug https://github.com/python/mypy/issues/9980.
DFTWaveVectorsMeshArrays = list[np.ndarray]


@dataclass(frozen=True, slots=True)
class DFTWaveVectorsMesh:
    """Wave vectors of the pseudo-spectral wave function approximation.

    :param tuple[np.ndarray, ...] wave_vectors:
        A tuple of `numpy.ndarray` instances. Each tuple element is a sparse
        mesh containing a component of the wave vectors used to approximate
        the wave function. The tuple must have at most three elements.
    """

    # Mesh arrays.
    wave_vectors: list[np.ndarray]

    @property
    def arrays(self) -> DFTWaveVectorsMeshArrays:
        """Return the NumPy arrays representing the mesh.

        **NOTE**: The returned arrays are sparse.
        """
        return list(
            np.meshgrid(*self.wave_vectors, indexing="ij", sparse=True)
        )


@dataclass(frozen=True, slots=True)
class PSSolverState:
    # noinspection PyUnresolvedReferences
    """Represent the state of a :class:`BEPSSolver` instance each time step.

    :param Mesh mesh:
        A spatial :py:class:`Mesh` instance where the GPE is solved.
    :param  numpy.ndarray wave_func:
        Complex wave function approximation of the system state.
    :param numpy.ndarray wave_func_dft:
        Discrete Fourier Transform of the complex wave function.
    :param DFTWaveVectorsMesh wave_vectors_mesh:
        Wave vectors of the Discrete Fourier Transform approximation.
    :param float energy:
        System energy approximation.
    :param float kinetic_energy:
        System kinetic energy approximation.
    :param WaveVector lattice_wave_vector:
        If the approximated state is a Bloch state, this attribute equals
        the Bloch state wave vector. Otherwise, it is ``None``.
    """

    # The spatial mesh.
    mesh: Mesh

    # The final state.
    periodic_wave_funcs: list[np.ndarray]

    # DFT of the wave function.
    periodic_wave_funcs_dft: list[np.ndarray]

    # Wave vectors of the wave function Fourier expansion.
    dft_wave_vectors_mesh: DFTWaveVectorsMesh

    # Energy.
    energies: list[float]

    # Kinetic energy.
    kinetic_energies: list[float]

    # Energy contribution due to the external potential.
    potential_energies: list[float]

    # Bloch state and/or lattice wave vector.
    lattice_wave_vector: WaveVector

    @property
    def states(self) -> list[BlochState]:
        """Quantum states associated with the solver state."""
        return list(
            BlochState(
                mesh=self.mesh,
                periodic_wave_func=periodic_wave_func,
                wave_vector=self.lattice_wave_vector,
            )
            for periodic_wave_func in self.periodic_wave_funcs
        )

    @classmethod
    def with_properties(
        cls,
        mesh: Mesh,
        periodic_wave_funcs: list[np.ndarray],
        external_potential_array: np.ndarray,
        lattice_wave_vector: WaveVector,
        dft_wave_vectors_mesh: DFTWaveVectorsMesh,
        dft_wave_vectors_norms_sqr: np.ndarray,
        wave_vectors_prod: np.ndarray,
    ):
        """Get the system properties and add them to the solver state."""
        element_size = mesh.element_size
        num_elements = mesh.num_elements

        wave_funcs_dft = []
        energies = []
        kinetic_energies = []
        potential_energies = []
        for periodic_wave_func in periodic_wave_funcs:
            wave_func_abs_sqr = np.abs(periodic_wave_func) ** 2
            wave_func_dft = fft.fftn(periodic_wave_func)
            wave_func_tdx_dft_abs_sqr = np.abs(wave_func_dft) ** 2

            # We have to correct the kinetic energy due to the presence
            # of the Bloch state lattice wave vector.
            lattice_wave_vector_norm = float(
                sum([vector_part**2 for vector_part in lattice_wave_vector])
            )
            kinetic_energy = (
                0.5
                * (element_size / num_elements)
                * np.sum(
                    dft_wave_vectors_norms_sqr * wave_func_tdx_dft_abs_sqr
                )
                - (element_size / num_elements)
                * np.sum(wave_vectors_prod * wave_func_tdx_dft_abs_sqr)
                + 0.5 * lattice_wave_vector_norm
            )

            # Calculate the corresponding. potential energy.
            potential_energy = element_size * np.sum(
                external_potential_array * wave_func_abs_sqr
            )

            wave_funcs_dft.append(wave_func_dft)
            kinetic_energies.append(kinetic_energy)
            potential_energies.append(potential_energy)
            energies.append(kinetic_energy + potential_energy)

        return cls(
            mesh=mesh,
            periodic_wave_funcs=periodic_wave_funcs,
            periodic_wave_funcs_dft=wave_funcs_dft,
            dft_wave_vectors_mesh=dft_wave_vectors_mesh,
            energies=energies,
            kinetic_energies=kinetic_energies,
            potential_energies=potential_energies,
            lattice_wave_vector=lattice_wave_vector,
        )


class PSSolverCommon(metaclass=ABCMeta):
    """Implement common properties of a pseudo-spectral solver."""

    # The mesh that represents the domain where the solver will find the
    # eigenstates.
    mesh: Mesh

    # The lattice wave-vector associated with the Bloch eigenstates the
    # solver calculates.
    lattice_wave_vector: WaveVector

    @property
    def lattice_wave_vector_norm(self) -> float:
        """Get the initial quantum state lattice wave vector norm."""
        return float(
            sum([vector_part**2 for vector_part in self.lattice_wave_vector])
        )

    @property
    def dft_wave_vectors_norms_sqr(self) -> np.ndarray:
        """Get the DFT wave vector norms."""
        mesh = self.mesh
        vectors_parts = self.dft_wave_vectors_mesh(mesh).arrays
        vectors_norms = np.zeros(mesh.shape, dtype=np.float64)
        for dft_wave_vector_part in vectors_parts:
            vectors_norms += dft_wave_vector_part**2
        return vectors_norms

    @property
    def wave_vectors_prod(self) -> np.ndarray:
        """Get the dot product of the DFT and lattice wave vectors."""
        # Dot product of wave vectors appears in several places, so it is
        # worth to calculate it only once.
        mesh = self.mesh
        vectors_prod = np.zeros(mesh.shape, dtype=np.float64)
        for wave_vector_part, dft_wave_vector_part in zip_longest(
            self.lattice_wave_vector,
            self.dft_wave_vectors_mesh(mesh).arrays,
        ):
            vectors_prod += wave_vector_part * dft_wave_vector_part
        return vectors_prod

    @staticmethod
    def dft_wave_vectors_mesh(mesh: Mesh) -> DFTWaveVectorsMesh:
        """Wave vectors of the pseudo-spectral wave function approximation.

        :param mesh:
            A ``Mesh`` instance that defines the spatial domain where
            the GPE will be solved.
        """
        wave_vectors_set: list[np.ndarray] = [
            2 * pi * fft.fftfreq(partition.num_segments, partition.step_size)
            for partition in mesh.partitions
        ]
        return DFTWaveVectorsMesh(wave_vectors_set)

    @property
    def plane_waves(self) -> Iterator[np.ndarray]:
        """Generate plane waves to be used as initial approximations."""
        spatial_dimension = self.mesh.dimension
        max_num_segments = max(
            [partition.num_segments for partition in self.mesh.partitions]
        )
        indexes: tuple[int, ...]
        indexes_iterator = sorted(
            product(range(max_num_segments), repeat=spatial_dimension),
            key=lambda value: sum(value),
        )
        for indexes in indexes_iterator:
            yield self.plane_wave(indexes)

    def plane_wave(self, indexes: tuple[int, ...]) -> np.ndarray:
        """Get the wave function of the initial quantum state.

        For a Bloch state, it is periodic-part wave function.
        """
        mesh_arrays = self.mesh.arrays
        mesh_sizes = [partition.size for partition in self.mesh.partitions]
        reciprocal_primitive_vectors = [2 * pi / size for size in mesh_sizes]
        print(indexes)

        exp_func_arg = np.zeros(self.mesh.shape)
        for reciprocal_vector, mesh_array, index in zip(
            reciprocal_primitive_vectors, mesh_arrays, indexes
        ):
            exp_func_arg += reciprocal_vector * index * mesh_array
        plane_wave_func = np.exp(1j * exp_func_arg)
        return plane_wave_func


@dataclass(frozen=True, slots=True)
class PSSolver(PSSolverCommon, Iterable[PSSolverState]):
    # noinspection PyUnresolvedReferences
    """Eigen-solver based on a pseudo-spectral algorithm.

    :param mesh:
        The mesh that represents the domain where the solver will find the
        eigenstates.
    :param lattice_wave_vector:
        The lattice wave-vector associated with the Bloch eigenstates the
        solver calculates.
    :param ExternalPotential external_potential:
        An external potential. It affects the geometry of the system and
        the Bose gas properties.
    :param num_eigenstates:
        The number of eigenstates to calculate with the lowest energies.
    :param TimePartition time_partition:
        A time partition. It defines the imaginary time evolution.
    :param float abs_tol:
        Set the absolute minimum tolerance between wave function
        approximations at successive time-steps the solver must reach
        before stopping.
    """

    # The spatial mesh.
    mesh: Mesh

    # The lattice wave vector.
    lattice_wave_vector: WaveVector

    # The system external potential.
    external_potential: ExternalPotential

    # Number of eigenstates to calculate.
    num_eigenstates: int

    # Temporal mesh used for the imaginary-time evolution.
    time_partition: TimePartition

    # Absolute tolerance for the convergence test at each time step.
    abs_tol: float = 1e-4

    @property
    def final_state(self) -> PSSolverState:
        """Get the last state yield by this solver.

        :rtype: PSSolverState
        """
        states_deque = deque(self, maxlen=1)
        last_solver_state: PSSolverState = states_deque.pop()
        return last_solver_state

    def evolve_with_strang_splitting(
        self,
        ini_wave_func: np.ndarray,
        time_step: float,
        external_potential_factor: np.ndarray,
    ) -> np.ndarray:
        """Evolve an initial wave function using the Strang splitting.

        :param ini_wave_func:
            The initial wave function
        :param time_step:
            The imaginary time-step.
        :param external_potential_factor:
            A term that accounts for the contribution of the external
            potential to the imaginary time evolution.
        :return:
            The evolved wave function.
        """
        # Apply first step of the Strang splitting.
        next_wave_func = external_potential_factor * ini_wave_func

        # Apply second step of the Strang splitting. Go forth to the
        # momentum space using a DFT.
        next_wave_func_dft = np.exp(
            -0.5 * time_step * self.dft_wave_vectors_norms_sqr
        ) * fft.fftn(next_wave_func)

        # Apply third step of the Strand splitting. Go back from the
        # momentum space using an inverse DFT.
        next_wave_func = external_potential_factor * fft.ifftn(
            next_wave_func_dft
        )
        return next_wave_func

    def __iter__(self) -> Iterator[PSSolverState]:
        """Make this class instances iterable objects."""
        # Start the imaginary time evolution.
        mesh = self.mesh
        element_size = mesh.element_size
        time_step = self.time_partition.time_step
        num_time_steps = self.time_partition.num_steps
        external_potential_array = self.external_potential(mesh)
        dft_wave_vectors_mesh = self.dft_wave_vectors_mesh(mesh)
        external_potential_factor = np.exp(
            -0.5 * time_step * external_potential_array
        )
        wave_funcs_tdx = list(islice(self.plane_waves, self.num_eigenstates))
        wave_func_tdx_diff = [np.nan] * self.num_eigenstates
        break_states_evolution = [False] * self.num_eigenstates

        # Logging information.
        logger.debug(
            "Starting the procedure to get the ground-state using"
            "a pseudo-spectral approach."
        )

        for step_index in range(num_time_steps + 1):
            # At this point, we are ready to retrieve the current solver
            # state to the caller.
            yield PSSolverState.with_properties(
                mesh=mesh,
                periodic_wave_funcs=wave_funcs_tdx,
                external_potential_array=external_potential_array,
                dft_wave_vectors_mesh=dft_wave_vectors_mesh,
                dft_wave_vectors_norms_sqr=self.dft_wave_vectors_norms_sqr,
                wave_vectors_prod=self.wave_vectors_prod,
                lattice_wave_vector=self.lattice_wave_vector,
            )

            # If we have reached the maximum number of steps, then do not
            # make any other calculations. Just try to go to the next loop
            # step and exhaust it.
            if step_index == num_time_steps:
                continue
            if np.all(break_states_evolution):
                break

            evolved_wave_funcs_tdx = []
            for idx, wave_func_tdx in enumerate(wave_funcs_tdx):
                if break_states_evolution[idx]:
                    evolved_wave_funcs_tdx.append(wave_func_tdx)
                    continue

                # Apply the Strand splitting.
                next_wave_func_tdx = self.evolve_with_strang_splitting(
                    wave_func_tdx, time_step, external_potential_factor
                )
                evolved_wave_funcs_tdx.append(next_wave_func_tdx)

            orthogonal_wave_funcs_tdx = []
            for idx, wave_func_idx in enumerate(evolved_wave_funcs_tdx):
                # Orthonormalize the new approximation.
                ortho_wave_func = wave_func_idx
                for kdx, wave_func_kdx in enumerate(
                    evolved_wave_funcs_tdx[:idx]
                ):
                    # ortho_wave_func = (
                    #     ortho_wave_func
                    #     - element_size
                    #     * evolved_wave_funcs_tdx[kdx]
                    #     * np.sum(
                    #         evolved_wave_funcs_tdx[kdx].conjugate()
                    #         * ortho_wave_func
                    #     )
                    # )
                    ortho_wave_func = (
                        ortho_wave_func
                        - element_size
                        * np.sum(
                            evolved_wave_funcs_tdx[kdx].conjugate()
                            * ortho_wave_func
                        )
                        * evolved_wave_funcs_tdx[kdx]
                    )

                # Get the norm of the next wave function approximation,
                # and normalize it.
                ortho_wave_func_norm = np.sqrt(
                    np.sum(np.abs(ortho_wave_func) ** 2) * element_size
                )
                ortho_wave_func = ortho_wave_func / ortho_wave_func_norm
                orthogonal_wave_funcs_tdx.append(ortho_wave_func)

            next_wave_funcs_tdx = []
            for idx, ortho_wave_func_tdx in enumerate(
                orthogonal_wave_funcs_tdx
            ):
                # Compare the difference between wave functions.
                if not break_states_evolution[idx]:
                    ortho_wave_func_tdx_diff = np.abs(
                        wave_funcs_tdx[idx] - ortho_wave_func_tdx
                    ).max()
                    wave_func_tdx_diff[idx] = ortho_wave_func_tdx_diff
                else:
                    ortho_wave_func_tdx_diff = wave_func_tdx_diff[idx]

                # Reassign the initial wave function for the next time step.
                next_wave_funcs_tdx.append(ortho_wave_func_tdx)

                logger.debug(
                    {
                        "Time step": step_index + 1,
                        "Abs. diff": ortho_wave_func_tdx_diff,
                    }
                )

                # Stop evolving the wave function if we reached the absolute
                # tolerance.
                if ortho_wave_func_tdx_diff <= self.abs_tol:
                    break_states_evolution[idx] = True

            wave_funcs_tdx = next_wave_funcs_tdx
        else:
            message = (
                "The Pseudo-Spectral solver reached the maximum number of "
                "time steps without "
                "converging to the prescribed tolerance."
            )
            logger.warning(
                {
                    "message": message,
                    "abs_tol": self.abs_tol,
                    "wave_func_abs_diff": wave_func_tdx_diff,
                }
            )
