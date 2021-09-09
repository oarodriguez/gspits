"""Validate the routines in `gspits.one_dim.gs_single` module."""
import logging

import pytest
from hypothesis import given, settings
from hypothesis import strategies as stg
from rich.logging import RichHandler

from gspits import Mesh, TimeMesh
from gspits.one_dim import HTHamiltonian
from gspits.one_dim.gs_single import BEPSSolver

# Some parameter values that define a valid harmonic trap potential.
FREQ = 1
INT_STRENGTH = 1

logger = logging.getLogger("gspits.one_dim.gs_single")


@given(freq=stg.floats(min_value=1, max_value=5))
@settings(deadline=10000, max_examples=5)
def test_beps_solver_ground_state(freq: float):
    """Check that the solver runs."""
    logger.addHandler(RichHandler())
    hamiltonian = HTHamiltonian(freq=freq, int_strength=INT_STRENGTH)
    trap_size = hamiltonian.trap_size
    mesh = Mesh(
        lower_bound=-8 * trap_size, upper_bound=8 * trap_size, num_segments=256
    )
    ini_state = hamiltonian.gaussian_state(mesh)
    time_mesh = TimeMesh(time_step=2 ** (-4), num_steps=128)
    beps_solver = BEPSSolver(
        hamiltonian=hamiltonian,
        mesh=mesh,
        time_mesh=time_mesh,
        ini_state=ini_state,
        abs_tol=1e-8,
    )
    ground_state = beps_solver.ground_state
    assert ground_state.norm == pytest.approx(1, abs=1e-8)
