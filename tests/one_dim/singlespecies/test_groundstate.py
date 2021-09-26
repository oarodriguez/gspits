"""Validate the `gspits.one_dim.singlespecies.groundstate` module."""
import logging

import attr
import pytest
from hypothesis import given, settings
from hypothesis import strategies as stg
from rich.logging import RichHandler

from gspits import Partition as Mesh
from gspits import TimePartition as TimeMesh
from gspits.one_dim import BlochState
from gspits.one_dim.singlespecies import BEPSSolver, HTHamiltonian

# Some parameter values that define a valid harmonic trap potential.
FREQ = 1
INTERACTION_STRENGTH = 1

logger = logging.getLogger("gspits.one_dim.gs_single")
logger.addHandler(RichHandler())


@given(freq=stg.floats(min_value=1, max_value=5))
@settings(deadline=10000, max_examples=5)
def test_beps_solver_ground_state(freq: float):
    """Check that the solver runs."""
    hamiltonian = HTHamiltonian(
        freq=freq, interaction_strength=INTERACTION_STRENGTH
    )
    trap_size = hamiltonian.trap_size
    mesh = Mesh(
        lower_bound=-8 * trap_size, upper_bound=8 * trap_size, num_segments=256
    )
    ini_state = hamiltonian.gaussian_state(mesh)
    time_mesh = TimeMesh(time_step=2 ** (-4), num_steps=256)
    beps_solver = BEPSSolver(
        hamiltonian=hamiltonian,
        ini_state=ini_state,
        time_mesh=time_mesh,
        abs_tol=1e-8,
        max_time_step_iters=256,
    )
    ground_state = beps_solver.final_state.state
    assert ground_state.norm == pytest.approx(1, abs=1e-8)


@given(freq=stg.floats(min_value=1, max_value=5))
@settings(deadline=10000, max_examples=5)
def test_beps_solver_bloch_ground_state(freq: float):
    """Verify solver results when we use Bloch states."""
    hamiltonian = HTHamiltonian(
        freq=freq, interaction_strength=INTERACTION_STRENGTH
    )
    trap_size = hamiltonian.trap_size
    mesh = Mesh(
        lower_bound=-8 * trap_size, upper_bound=8 * trap_size, num_segments=256
    )
    time_mesh = TimeMesh(time_step=2 ** (-4), num_steps=256)

    # Solve the GPE using a Gaussian state as the starting point.
    ini_gaussian_state = hamiltonian.gaussian_state(mesh)
    beps_solver_1 = BEPSSolver(
        hamiltonian=hamiltonian,
        ini_state=ini_gaussian_state,
        time_mesh=time_mesh,
        abs_tol=1e-8,
        max_time_step_iters=256,
    )
    final_state_solver_1 = beps_solver_1.final_state
    ground_state_1 = final_state_solver_1.state

    # Solve the GPE using a Bloch state as the starting point.
    ini_bloch_state = BlochState.plane_wave(mesh, wave_vector=0)
    beps_solver_2 = attr.evolve(beps_solver_1, ini_state=ini_bloch_state)
    final_state_solver_2 = beps_solver_2.final_state
    ground_state_2 = final_state_solver_2.state

    # Compare results.
    # TODO: Add better assertions.
    energy_2 = final_state_solver_2.energy
    energy_1 = final_state_solver_1.energy
    assert energy_2 == pytest.approx(energy_1, abs=1e-5)
    assert ground_state_1.norm == pytest.approx(ground_state_2.norm, abs=1e-8)
