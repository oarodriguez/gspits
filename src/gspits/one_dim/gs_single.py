"""Routines for getting the ground-state and related properties."""

import logging
from collections import deque
from math import pi
from typing import Iterable, Iterator

import numpy as np
from attr import dataclass
from scipy import fft

from gspits import Mesh, TimeMesh
from gspits.one_dim import Hamiltonian, State

# Initialize a logger for this module.
logger = logging.getLogger(__name__)

__all__ = ["BEPSSolver", "BEPSSolverState"]


@dataclass(frozen=True, slots=True)
class BEPSSolverState:
    """BEPSSolver state at each time step."""

    __slots__ = ()

    # The spatial mesh.
    mesh: Mesh

    # The final state.
    wave_func: np.ndarray

    # Wave vectors of the wave function Fourier expansion.
    wave_vectors: np.ndarray

    # FFT of the wave function.
    wave_func_fft: np.ndarray

    # Energy.
    energy: float

    # Chemical potential
    chemical_potential: float

    # Kinetic energy.
    kinetic_energy: float

    # Interaction energy.
    interaction_energy: float

    @property
    def state(self):
        """Quantum state associated with the solver state."""
        return State(self.mesh, self.wave_func)


@dataclass(frozen=True, slots=True)
class BEPSSolver(Iterable[BEPSSolverState]):
    """Backward Euler Pseudo-Spectral solver."""

    __slots__ = ()

    # The system Hamiltonian.
    hamiltonian: Hamiltonian

    # State used for starting the procedure.
    ini_state: State

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
    def ground_state(self):
        """Get the ground state yield by this solver."""
        states_dq = deque(self, maxlen=1)
        last_solver_state: BEPSSolverState = states_dq.pop()
        return State(
            mesh=last_solver_state.mesh, wave_func=last_solver_state.wave_func
        )

    @staticmethod
    def wave_vectors(mesh: Mesh) -> np.ndarray:
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
        wave_vectors = self.wave_vectors(mesh)
        time_step = self.time_mesh.time_step
        num_time_steps = self.time_mesh.num_steps
        max_time_step_iters = self.max_time_step_iters
        abs_tol = self.abs_tol
        time_step_abs_tol = self.time_step_abs_tol
        ext_potential = self.hamiltonian.external_potential
        ext_pot_array = ext_potential(domain_mesh)
        # Imaginary time evolution.
        wave_func_tdx = ini_state.wave_func
        wave_func_tdx_diff = np.nan
        break_evolve = False
        # Logging information.
        logger.debug("Starting BEPS")
        for step_tdx in range(num_time_steps + 1):
            wave_func_tdx_abs_sqr = np.abs(wave_func_tdx) ** 2
            wave_func_tdx_abs_quartic = wave_func_tdx_abs_sqr ** 2
            full_pot_array = (
                ext_pot_array + interaction_factor * wave_func_tdx_abs_sqr
            )
            alpha = 0.5 * (full_pot_array.min() + full_pot_array.max())
            wave_func_tdx_fft = fft.fft(wave_func_tdx)
            kinetic_energy = (
                0.5
                * (step_size / num_segments)
                * np.sum(wave_vectors ** 2 * np.abs(wave_func_tdx_fft) ** 2)
            )
            interaction_energy = (
                0.5
                * interaction_factor
                * step_size
                * np.sum(wave_func_tdx_abs_quartic)
            )
            potential_energy = (
                step_size * np.sum(ext_pot_array * wave_func_tdx_abs_sqr)
                + interaction_energy
            )
            energy = kinetic_energy + potential_energy
            chemical_pot = energy + interaction_energy
            yield BEPSSolverState(
                mesh=mesh,
                wave_func=wave_func_tdx,
                wave_vectors=wave_vectors,
                wave_func_fft=wave_func_tdx_fft,
                energy=energy,
                chemical_potential=chemical_pot,
                kinetic_energy=kinetic_energy,
                interaction_energy=interaction_energy,
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
            wave_func_idx_diff = np.nan
            for idx in range(max_time_step_iters + 1):
                stabilization_term = (alpha - full_pot_array) * wave_func_idx
                next_wave_func_idx_fft = (
                    2
                    * (
                        wave_func_tdx_fft
                        + time_step * fft.fft(stabilization_term)
                    )
                    / (2 + time_step * (2 * alpha + wave_vectors ** 2))
                )
                next_wave_func_idx: np.ndarray = fft.ifft(
                    next_wave_func_idx_fft
                )
                wave_func_idx_diff: float = np.abs(
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
            wave_func_tdx_diff: float = np.abs(
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
