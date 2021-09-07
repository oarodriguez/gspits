"""Validate the functionality of the `gspits.one_dim.single` module."""
from typing import NamedTuple

import numpy as np
import pytest
from attr import evolve
from hypothesis import given, settings
from hypothesis import strategies as stg

from gspits import Mesh
from gspits.one_dim import (
    HTHamiltonian,
    MRHamiltonian,
    OLHTHamiltonian,
    plane_wave_state,
)

# Some parameter values that define a valid harmonic trap potential.
FREQ = 1
INT_ENERGY = 1


@given(
    freq=stg.floats(allow_infinity=False, allow_nan=False, max_value=0),
)
def test_ht_invalid_params(freq: float):
    """Check that invalid parameters are correctly managed."""
    valid_ho = HTHamiltonian(freq=FREQ, int_energy=INT_ENERGY)
    with pytest.raises(ValueError):
        # Check invalid frequencies.
        evolve(valid_ho, freq=freq)
    # NOTE: The interaction energy can take, in principle, any real value.
    # Raise an error with NaNs.
    with pytest.raises(ValueError):
        evolve(valid_ho, freq=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_ho, int_energy=np.nan)


# TODO: Maybe this mesh should be a fixture with arbitrary bounds and
#  number of steps. How?
_domain_mesh = Mesh(lower_bound=-10, upper_bound=10, num_segments=128)


@given(
    freq=stg.floats(min_value=1, max_value=1e2, allow_nan=False),
    int_energy=stg.floats(min_value=-1e-2, max_value=1e2, allow_nan=False),
)
@settings(max_examples=3, deadline=None)
def test_ht(freq: float, int_energy: float):
    """Check that the core function works."""
    ho = HTHamiltonian(freq=freq, int_energy=int_energy)
    domain_array = _domain_mesh.array
    ho_pot_func = ho.external_potential
    func_array = ho_pot_func(domain_array)
    assert func_array.shape == domain_array.shape


@given(
    freq=stg.floats(min_value=1, max_value=5),
    num_segments=stg.integers(min_value=128, max_value=256),
)
def test_ht_gaussian_state(freq: float, num_segments: int):
    """Check that Gaussian states are normalized.

    The normalization condition accuracy depends on the number of
    steps in the mesh. Here, we test for a minimum value of 128 steps
    and an absolute tolerance for the state norm of 1e-8.
    NOTE: How can we improve this test?
    """
    valid_ho = HTHamiltonian(
        freq=freq,
        int_energy=INT_ENERGY,
    )
    trap_size = valid_ho.trap_size
    mesh = Mesh(
        lower_bound=-8 * trap_size,
        upper_bound=8 * trap_size,
        num_segments=num_segments,
    )
    state = valid_ho.gaussian_state(mesh)
    assert state.norm == pytest.approx(1, abs=1e-8)


# Optical lattice Hamiltonian parameters.
LATTICE_DEPTH = 1
WAVELENGTH = 1


@given(
    freq=stg.floats(allow_infinity=False, allow_nan=False, max_value=0),
    wavelength=stg.floats(allow_infinity=False, allow_nan=False, max_value=0),
)
def test_olht_invalid_params(freq: float, wavelength: float):
    """Check that invalid Hamiltonian parameters are handled correctly."""
    valid_hamiltonian = OLHTHamiltonian(
        lattice_depth=LATTICE_DEPTH,
        freq=FREQ,
        wavelength=WAVELENGTH,
        int_energy=INT_ENERGY,
    )
    with pytest.raises(ValueError):
        # Check invalid frequencies.
        evolve(valid_hamiltonian, freq=freq)
    with pytest.raises(ValueError):
        # Check invalid wavelengths.
        evolve(valid_hamiltonian, wavelength=wavelength)
    # NOTE: The lattice_depth and int_energy can take, in principle,
    #  any real value.
    # Raise an error with NaNs.
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, lattice_depth=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, freq=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, wavelength=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, int_energy=np.nan)


@given(
    lattice_depth=stg.floats(min_value=-1e2, max_value=1e2, allow_nan=False),
    freq=stg.floats(min_value=1, max_value=1e2, allow_nan=False),
    wavelength=stg.floats(min_value=1e-2, max_value=1e2, allow_nan=False),
    int_energy=stg.floats(min_value=-1e-2, max_value=1e2, allow_nan=False),
)
@settings(max_examples=3, deadline=None)
def test_olht(
    lattice_depth: float, freq: float, wavelength: float, int_energy: float
):
    """Check the Hamiltonian core functionality works correctly."""
    hamiltonian = OLHTHamiltonian(
        lattice_depth=lattice_depth,
        freq=freq,
        wavelength=wavelength,
        int_energy=int_energy,
    )
    domain_array = _domain_mesh.array
    ho_pot_func = hamiltonian.external_potential
    func_array = ho_pot_func(domain_array)
    assert func_array.shape == domain_array.shape


# Multi-rod Hamiltonian parameters.
LATTICE_DEPTH = 1
LATTICE_PERIOD = 1
BARRIER_WIDTH = 1 / 2
INT_ENERGY = 1

lattice_period_stg = stg.floats(
    min_value=0, max_value=1e2, allow_infinity=False, allow_nan=False
)


class _MRParams(NamedTuple):
    """Multi-Rod potential parameters."""

    lattice_period: float
    barrier_width: float


def _invalid_params(params: _MRParams):
    """Confirm if provided Hamiltonian parameters are invalid."""
    if params.barrier_width < 0:
        return True
    if params.lattice_period <= 0:
        return True
    return params.barrier_width > params.lattice_period


lattice_period_stg = stg.floats(
    min_value=-1e2, max_value=1e2, allow_infinity=False, allow_nan=False
)
barrier_width_stg = stg.floats(
    min_value=-1e2, max_value=1e2, allow_infinity=False, allow_nan=False
)
invalid_params_stg = (
    stg.tuples(lattice_period_stg, barrier_width_stg)
    .map(lambda v: _MRParams(*v))
    .filter(_invalid_params)
)


@given(params=invalid_params_stg)
def test_mr_invalid_params(params: _MRParams):
    """Check that invalid Hamiltonian parameters are handled correctly."""
    valid_hamiltonian = MRHamiltonian(
        lattice_depth=LATTICE_DEPTH,
        lattice_period=LATTICE_PERIOD,
        barrier_width=BARRIER_WIDTH,
        int_energy=INT_ENERGY,
    )
    with pytest.raises(ValueError):
        # Check invalid lattice periods and barrier widths.
        evolve(
            valid_hamiltonian,
            lattice_period=params.lattice_period,
            barrier_width=params.barrier_width,
        )
    # NOTE: The lattice_depth and int_energy can take, in principle,
    #  any real value.
    # Raise an error with NaNs.
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, lattice_depth=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, lattice_period=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, barrier_width=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, int_energy=np.nan)


def _valid_params(params: _MRParams):
    """Confirm if provided Hamiltonian parameters are valid."""
    return params.barrier_width <= params.lattice_period


lattice_period_stg = stg.floats(
    min_value=1e-2,
    max_value=1e2,
    allow_infinity=False,
    allow_nan=False,
    exclude_min=True,
)
barrier_width_stg = stg.floats(
    min_value=1e-2,
    max_value=1e2,
    allow_infinity=False,
    allow_nan=False,
    exclude_min=True,
)
valid_params_stg = (
    stg.tuples(lattice_period_stg, barrier_width_stg)
    .map(lambda v: _MRParams(*v))
    .filter(_valid_params)
)


@given(
    lattice_depth=stg.floats(min_value=-1e2, max_value=1e2, allow_nan=False),
    params=valid_params_stg,
    int_energy=stg.floats(min_value=-1e-2, max_value=1e2, allow_nan=False),
)
@settings(max_examples=3, deadline=None)
def test_mr(
    lattice_depth: float,
    params: _MRParams,
    int_energy: float,
):
    """Check the Hamiltonian core functionality works correctly."""
    hamiltonian = MRHamiltonian(
        lattice_depth=lattice_depth,
        lattice_period=params.lattice_period,
        barrier_width=params.barrier_width,
        int_energy=int_energy,
    )
    # Some natural but important assertions about the Hamiltonian attributes.
    assert (
        hamiltonian.well_width + hamiltonian.barrier_width
    ) == pytest.approx(hamiltonian.lattice_period, 1e-8)
    assert hamiltonian.int_factor == int_energy
    # Check that the external potential function works correctly.
    domain_array = _domain_mesh.array
    mr_pot_func = hamiltonian.external_potential
    func_array = mr_pot_func(domain_array)
    assert func_array.shape == domain_array.shape


@given(
    lower_bound=stg.floats(min_value=-1e2, max_value=-1, allow_nan=False),
    upper_bound=stg.floats(
        min_value=1, max_value=1e2, exclude_min=True, allow_nan=False
    ),
    num_segments=stg.integers(min_value=128, max_value=256),
)
def test_plane_wave_state(
    lower_bound: float, upper_bound: float, num_segments: int
):
    """Check that plane wave states are normalized.

    The normalization condition accuracy depends on the number of
    steps in the mesh. Here, we test for a minimum value of 128 steps
    and an absolute tolerance for the state norm of 1e-8.
    NOTE: How can we improve this test?
    """
    mesh = Mesh(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_segments=num_segments,
    )
    state = plane_wave_state(mesh)
    # Using Riemann sums to get the norm is not the most accurate method
    # for the plane wave norms. Therefore, we use an absolute tolerance
    # equal to the mesh step size.
    assert state.norm == pytest.approx(1, abs=mesh.step_size)

    # The following tests should catch refactorings.
    _state = HTHamiltonian.plane_wave_state(mesh)
    assert np.all(_state.wave_func == state.wave_func)
    _state = OLHTHamiltonian.plane_wave_state(mesh)
    assert np.all(_state.wave_func == state.wave_func)
    _state = MRHamiltonian.plane_wave_state(mesh)
    assert np.all(_state.wave_func == state.wave_func)
