"""Validate the routines in `gspits.one_dim.gs_single` module."""
import logging

import attr
import pytest
from hypothesis import given, settings
from hypothesis import strategies as stg
from rich.logging import RichHandler

from gspits import Mesh, TimeMesh
from gspits.one_dim import BlochState, HTHamiltonian
from gspits.one_dim.gs_single import BEPSSolver

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
    ground_state = beps_solver.ground_state
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
    beps_solver = BEPSSolver(
        hamiltonian=hamiltonian,
        ini_state=ini_gaussian_state,
        time_mesh=time_mesh,
        abs_tol=1e-8,
        max_time_step_iters=256,
    )
    ground_state_solver_1 = beps_solver.ground_state

    # Solve the GPE using a Bloch state as the starting point.
    ini_bloch_state = BlochState.plane_wave(mesh, wave_vector=0)
    beps_solver = attr.evolve(beps_solver, ini_state=ini_bloch_state)
    ground_state_solver_2 = beps_solver.ground_state

    # Compare results.
    # TODO: Add better assertions.
    assert ground_state_solver_1.norm == pytest.approx(
        ground_state_solver_2.norm, abs=1e-8
    )
