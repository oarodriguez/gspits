import logging

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as stg
from matplotlib import pyplot
from rich.logging import RichHandler

from gspits import Mesh, TimeMesh
from gspits.one_dim import HTHamiltonian
from gspits.one_dim.gs_single import BEPSSolver, BEPSSolverState

# Some parameter values that define a valid harmonic trap potential.
FREQ = 1
INT_ENERGY = 1

logger = logging.getLogger("gspits.one_dim.gs_single")


@given(freq=stg.floats(min_value=1, max_value=5))
@settings(deadline=10000, max_examples=5)
def test_beps_solver_ground_state(freq: float):
    """Check that the solver runs."""
    logger.addHandler(RichHandler())
    hamiltonian = HTHamiltonian(freq=freq, int_energy=INT_ENERGY)
    trap_size = hamiltonian.trap_size
    logger.debug({"trap_size": trap_size})
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


@pytest.mark.mpl_interactive
def test_beps_solver_states():
    """Check how the initial state evolves as the BEPSSolver runs."""
    # logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler())
    hamiltonian = HTHamiltonian(freq=FREQ, int_energy=INT_ENERGY)
    trap_size = hamiltonian.trap_size
    mesh = Mesh(
        lower_bound=-4 * trap_size, upper_bound=4 * trap_size, num_segments=128
    )
    time_mesh = TimeMesh(time_step=2 ** (-4), num_steps=256)
    ini_state = hamiltonian.gaussian_state(mesh)
    beps_solver = BEPSSolver(
        hamiltonian=hamiltonian,
        mesh=mesh,
        time_mesh=time_mesh,
        ini_state=ini_state,
        abs_tol=1e-8,
    )
    ground_state = ini_state
    solver_state: BEPSSolverState
    for idx, solver_state in enumerate(beps_solver):
        logger.debug(
            {
                "energy": solver_state.energy,
                "chemical_potential": solver_state.chemical_potential,
            }
        )
        pyplot.plot(
            solver_state.mesh.array, np.abs(solver_state.wave_func) ** 2
        )
        ground_state = solver_state.state

    assert ground_state.norm == pytest.approx(1, abs=1e-8)
    pyplot.show()


@pytest.mark.mpl_interactive
def test_view_beps_solver_ground_state():
    """Shows how the BEPSSolver evolves a initial state to the ground state."""
    # logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler())
    hamiltonian = HTHamiltonian(freq=FREQ, int_energy=INT_ENERGY)
    mesh = Mesh(lower_bound=-8, upper_bound=8, num_segments=256)
    time_mesh = TimeMesh(time_step=2 ** (-4), num_steps=128)
    ini_state = hamiltonian.gaussian_state(mesh)
    beps_solver = BEPSSolver(
        hamiltonian=hamiltonian,
        mesh=mesh,
        time_mesh=time_mesh,
        ini_state=ini_state,
        abs_tol=1e-5,
    )
    ground_state = beps_solver.ground_state
    pyplot.plot(ground_state.mesh.array, np.abs(ground_state.wave_func) ** 2)
    pyplot.plot(ini_state.mesh.array, np.abs(ini_state.wave_func) ** 2)
    pyplot.show()
