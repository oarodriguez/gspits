"""Validate the `gspits.noninteracting.tsps` module."""
import logging

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as stg
from matplotlib import pyplot
from rich.logging import RichHandler
from tests.groundstate.common import meshes_1d_stg, meshes_2d_stg

from gspits import Mesh, TimePartition
from gspits.noninteracting.tsps import PSSolver
from gspits.potentials.harmonic import HarmonicTrap

logger = logging.getLogger("gspits.noninteracting.tsps")
logger.addHandler(RichHandler())

# Some parameter values that define a valid harmonic trap potential.
FREQUENCY = 1


@pytest.mark.is_end_to_end_test
@pytest.mark.is_interactive_test
@given(freq=stg.integers(min_value=1, max_value=10), mesh=meshes_1d_stg)
@settings(deadline=10000, max_examples=1)
def test_solver_ground_state_1d(freq: float, mesh: Mesh):
    """Check that the solver runs."""
    potential = HarmonicTrap(
        frequencies=(FREQUENCY,) * mesh.dimension,
    )
    scaling_factors = tuple(size * 6 for size in potential.sizes)
    scaled_mesh = mesh.scaled(factors=scaling_factors)
    time_step = 2 ** (-5)
    time_partition = TimePartition(time_step=time_step, num_steps=1024)
    beps_solver = PSSolver(
        mesh=scaled_mesh,
        lattice_wave_vector=[0],
        external_potential=potential,
        num_eigenstates=5,
        time_partition=time_partition,
        abs_tol=1e-8,
    )
    solver_final_state = beps_solver.final_state
    eigen_states = solver_final_state.states
    for eigen_state in eigen_states:
        assert eigen_state.norm == pytest.approx(1, abs=1e-8)

    periodic_wave_funcs = list(solver_final_state.periodic_wave_funcs)
    for idx, periodic_wave_func in enumerate(periodic_wave_funcs):
        pyplot.plot(
            *scaled_mesh.arrays,
            np.absolute(np.array(periodic_wave_func)) ** 2 * mesh.size,
            lw=1 + 0.5 * idx,
        )
    pyplot.xlabel(r"$z/a_c$")
    pyplot.title(rf"$\omega = {freq}$")
    pyplot.show()


@pytest.mark.is_end_to_end_test
@pytest.mark.is_interactive_test
@given(freq=stg.integers(min_value=1, max_value=10), mesh=meshes_2d_stg)
@settings(deadline=40000, max_examples=1)
def test_solver_ground_state_2d(freq: float, mesh: Mesh):
    """Check that the solver runs."""
    potential = HarmonicTrap(
        frequencies=(1, 1),
    )
    scaling_factors = tuple(size * 8 for size in potential.sizes)
    scaled_mesh = mesh.scaled(factors=scaling_factors)
    time_step = 2 ** (-5)
    time_partition = TimePartition(time_step=time_step, num_steps=1024)
    beps_solver = PSSolver(
        mesh=scaled_mesh,
        lattice_wave_vector=[0, 0],
        external_potential=potential,
        num_eigenstates=10,
        time_partition=time_partition,
        abs_tol=1e-8,
    )
    solver_final_state = beps_solver.final_state
    eigen_energies = solver_final_state.energies
    eigen_states = solver_final_state.states
    for eigen_state in eigen_states:
        assert eigen_state.norm == pytest.approx(1, abs=1e-8)
    periodic_wave_funcs = list(solver_final_state.periodic_wave_funcs)
    partition_arrays = [
        partition.array for partition in scaled_mesh.partitions
    ]
    for idx, periodic_wave_func in enumerate(periodic_wave_funcs):
        pyplot.figure()
        pyplot.contourf(
            *partition_arrays,
            np.absolute(np.array(periodic_wave_func.T)) ** 2 * mesh.size,
            levels=100,
        )
        pyplot.xlabel(r"$x/a_c$")
        pyplot.ylabel(r"$y/a_c$")
        pyplot.title(rf"$\omega = {freq}$, $E = {eigen_energies[idx]:.5f}$")
    # pyplot.plot(*mesh.arrays, np.absolute(ground_state.wave_func) ** 2)
    pyplot.show()
