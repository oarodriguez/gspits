"""Routines to get the ground-state of single-species Hamiltonians."""

import logging
from collections import deque
from math import pi
from typing import Iterable, Iterator, Optional, Union

import numpy as np
from attr import dataclass
from scipy import fft

from gspits import Partition as Mesh
from gspits import TimePartition as TimeMesh
from gspits.one_dim import BlochState, Hamiltonian, State

# Initialize a logger for this module.
logger = logging.getLogger(__name__)

__all__ = ["BEPSSolver", "BEPSSolverState"]


# Type of states the BEPS solver calculates.
State_T = Union[State, BlochState]


@dataclass(frozen=True, slots=True)
class BEPSSolverState:
    """BEPSSolver state at each time step."""

    __slots__ = ()

    # The spatial mesh.
    mesh: Mesh

    # The final state.
    wave_func: np.ndarray

    # Wave vectors of the wave function Fourier expansion.
    fft_wave_vectors: np.ndarray

    # FFT of the wave function.
    wave_func_fft: np.ndarray

    # Energy.
    energy: float

    # Chemical potential
    chemical_potential: float

    # Kinetic energy.
    kinetic_energy: float

    # Energy contribution due to the external potential.
    potential_energy: float

    # Interaction energy.
    interaction_energy: float

    # Bloch state and/or lattice wave vector.
    lattice_wave_vector: Optional[float] = None

    @property
    def state(self) -> State_T:
        """Quantum state associated with the solver state."""
        if self.lattice_wave_vector is None:
            return State(self.mesh, self.wave_func)
        return BlochState(
            mesh=self.mesh,
            periodic_wave_func=self.wave_func,
            wave_vector=self.lattice_wave_vector,
        )


@dataclass(frozen=True, slots=True)
class BEPSSolver(Iterable[BEPSSolverState]):
    """Backward Euler Pseudo-Spectral solver."""

    __slots__ = ()

    # The system Hamiltonian.
    hamiltonian: Hamiltonian

    # State used for starting the procedure.
    ini_state: State_T

    # Temporal mesh used for the imaginary-time evolution.
    time_mesh: TimeMesh

    # Absolute tolerance for the convergence test at each time step.
    abs_tol: float = 1e-4

    # Absolute tolerance for the convergence test used at the inner
    # loop executed at each time step.
    time_step_abs_tol: float = 1e-8

    # Maximum number of steps for the inner loop executed at each time step.
    max_time_step_iters: int = 128

    @property
    def final_state(self) -> BEPSSolverState:
        """Get the last state yield by this solver."""
        states_dq = deque(self, maxlen=1)
        last_solver_state: BEPSSolverState = states_dq.pop()
        return last_solver_state

    @staticmethod
    def fft_wave_vectors(mesh: Mesh) -> np.ndarray:
        """Get the wave vectors of the wave function Fourier expansion."""
        return 2 * pi * fft.fftfreq(mesh.num_segments, mesh.step_size)

    def __iter__(self) -> Iterator[BEPSSolverState]:
        """Make this class instances iterable objects."""
        ini_state = self.ini_state
        mesh = ini_state.mesh
        interaction_factor = self.hamiltonian.interaction_factor
        domain_mesh = mesh.array
        step_size = mesh.step_size
        num_segments = mesh.num_segments
        domain_size = mesh.upper_bound - mesh.lower_bound
        fft_wave_vectors = self.fft_wave_vectors(mesh)
        external_potential = self.hamiltonian.external_potential
        external_potential_array = external_potential(domain_mesh)
        lattice_wave_vector = (
            ini_state.wave_vector
            if isinstance(ini_state, BlochState)
            else None
        )
        if isinstance(ini_state, BlochState):
            wave_func_tdx = ini_state.periodic_wave_func
        else:
            wave_func_tdx = ini_state.wave_func

        # Start the imaginary time evolution.
        time_step = self.time_mesh.time_step
        num_time_steps = self.time_mesh.num_steps
        max_time_step_iters = self.max_time_step_iters
        abs_tol = self.abs_tol
        time_step_abs_tol = self.time_step_abs_tol
        wave_func_tdx_diff: float = np.nan
        break_evolve = False
        # Logging information.
        logger.debug("Starting BEPS")
        for step_tdx in range(num_time_steps + 1):
            wave_func_tdx_abs_sqr = np.abs(wave_func_tdx) ** 2
            wave_func_tdx_abs_quartic = wave_func_tdx_abs_sqr ** 2
            wave_func_tdx_fft = fft.fft(wave_func_tdx)
            wave_func_tdx_fft_abs_sqr = np.abs(wave_func_tdx_fft) ** 2
            kinetic_energy = (
                0.5
                * (step_size / num_segments)
                * np.sum(fft_wave_vectors ** 2 * wave_func_tdx_fft_abs_sqr)
            )
            potential_array = (
                external_potential_array
                + interaction_factor * domain_size * wave_func_tdx_abs_sqr
            )
            # We have to correct the kinetic energy due to the presence
            # of the Bloch state lattice wave vector.
            if lattice_wave_vector is not None:
                kinetic_energy += (
                    -(step_size / num_segments)
                    * np.sum(
                        (lattice_wave_vector * fft_wave_vectors)
                        * wave_func_tdx_fft_abs_sqr
                    )
                    + 0.5 * lattice_wave_vector ** 2
                )
                potential_array += 0.5 * lattice_wave_vector ** 2
            interaction_energy = (
                0.5
                * interaction_factor
                * domain_size
                * step_size
                * np.sum(wave_func_tdx_abs_quartic)
            )
            potential_energy = step_size * np.sum(
                external_potential_array * wave_func_tdx_abs_sqr
            )
            energy = kinetic_energy + potential_energy + interaction_energy
            chemical_potential = energy + interaction_energy
            yield BEPSSolverState(
                mesh=mesh,
                wave_func=wave_func_tdx,
                fft_wave_vectors=fft_wave_vectors,
                wave_func_fft=wave_func_tdx_fft,
                energy=energy,
                chemical_potential=chemical_potential,
                kinetic_energy=kinetic_energy,
                potential_energy=potential_energy,
                interaction_energy=interaction_energy,
                lattice_wave_vector=lattice_wave_vector,
            )
            # If we have reached the maximum number of steps, then do not
            # make any other calculations. Just try to go to the next loop
            # step and exhaust it.
            if step_tdx == num_time_steps:
                continue
            if break_evolve:
                break
            # Start iterative procedure to find the wave function
            # after a single time step.
            wave_func_idx = wave_func_tdx
            wave_func_idx_diff: float = np.nan
            stabilization_param = 0.5 * (
                potential_array.min() + potential_array.max()
            )
            for idx in range(max_time_step_iters + 1):
                stabilized_potential = (
                    stabilization_param - potential_array
                ) * wave_func_idx
                # TODO: Extract this FFT (and similar expressions) in a
                #  separate routine, so we can override it in derived classes.
                next_wave_func_idx_fft_dividend = 2 * (
                    wave_func_tdx_fft
                    + time_step * fft.fft(stabilized_potential)
                )
                next_wave_func_idx_fft_divisor = 2 + time_step * (
                    2 * stabilization_param + fft_wave_vectors ** 2
                )
                # If the initial state is a Bloch state, include the
                # contribution arising from the lattice wave vector to the
                # divisor.
                if lattice_wave_vector is not None:
                    next_wave_func_idx_fft_divisor += (
                        -2 * time_step * lattice_wave_vector * fft_wave_vectors
                    )
                next_wave_func_idx_fft = (
                    next_wave_func_idx_fft_dividend
                    / next_wave_func_idx_fft_divisor
                )
                next_wave_func_idx: np.ndarray = fft.ifft(
                    next_wave_func_idx_fft
                )
                wave_func_idx_diff = np.abs(
                    wave_func_idx - next_wave_func_idx
                ).max()
                wave_func_idx = next_wave_func_idx
                logger.debug(
                    {"BEPS step iter": idx, "Abs. diff": wave_func_idx_diff}
                )
                if wave_func_idx_diff <= time_step_abs_tol:
                    break
            else:
                message = (
                    "The BEPS Solver reached the maximum number of "
                    "iterations during a single time step without "
                    "converging to the prescribed tolerance."
                )
                logger.warning(
                    {
                        "message": message,
                        "time_step_abs_tol": time_step_abs_tol,
                        "wave_func_abs_diff": wave_func_idx_diff,
                    }
                )
            # Get the norm of the next wave function approximation,
            # and normalize it.
            norm_wave_func_idx = np.sqrt(
                np.sum(np.abs(wave_func_idx) ** 2) * step_size
            )
            next_wave_func_tdx = wave_func_idx / norm_wave_func_idx
            # logger.debug({"wave_func": next_wave_func_tdx})
            # Compare the difference between wave functions.
            wave_func_tdx_diff = np.abs(
                wave_func_tdx - next_wave_func_tdx
            ).max()
            # Reassign the initial wave function for the next time step.
            wave_func_tdx = next_wave_func_tdx
            logger.debug(
                {"Time step": step_tdx + 1, "Abs. diff": wave_func_tdx_diff}
            )
            if wave_func_tdx_diff <= abs_tol:
                break_evolve = True
        else:
            message = (
                "The BEPS Solver reached the maximum number of "
                "time steps without "
                "converging to the prescribed tolerance."
            )
            logger.warning(
                {
                    "message": message,
                    "abs_tol": abs_tol,
                    "wave_func_abs_diff": wave_func_tdx_diff,
                }
            )
