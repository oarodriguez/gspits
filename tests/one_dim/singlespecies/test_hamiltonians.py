"""Validate the `gspits.one_dim.singlespecies.hamiltonians` module."""
from typing import NamedTuple

import numpy as np
import pytest
from attr import evolve
from hypothesis import given, note, settings
from hypothesis import strategies as stg

from gspits import Partition as Mesh
from gspits.one_dim.singlespecies import (
    DCHamiltonian,
    DeltaSpec,
    HTHamiltonian,
    MRHamiltonian,
    OLHTHamiltonian,
    SuperDCHamiltonian,
    plane_wave_state,
)

# Some parameter values that define a valid harmonic trap potential.
FREQ = 1
INTERACTION_STRENGTH = 1


@given(
    freq=stg.floats(allow_infinity=False, allow_nan=False, max_value=0),
)
def test_ht_invalid_params(freq: float):
    """Check that invalid parameters are correctly managed."""
    valid_ho = HTHamiltonian(
        freq=FREQ, interaction_strength=INTERACTION_STRENGTH
    )
    with pytest.raises(ValueError):
        # Check invalid frequencies.
        evolve(valid_ho, freq=freq)
    # NOTE: The interaction energy can take, in principle, any real value.
    # Raise an error with NaNs.
    with pytest.raises(ValueError):
        evolve(valid_ho, freq=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_ho, interaction_strength=np.nan)


# TODO: Maybe this mesh should be a fixture with arbitrary bounds and
#  number of steps. How?
_domain_mesh = Mesh(lower_bound=-10, upper_bound=10, num_segments=128)


@given(
    freq=stg.floats(min_value=1, max_value=1e2, allow_nan=False),
    interaction_strength=stg.floats(
        min_value=-1e-2, max_value=1e2, allow_nan=False
    ),
)
@settings(max_examples=3, deadline=None)
def test_ht(freq: float, interaction_strength: float):
    """Check that the core function works."""
    ho = HTHamiltonian(freq=freq, interaction_strength=interaction_strength)
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
        interaction_strength=INTERACTION_STRENGTH,
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
        interaction_strength=INTERACTION_STRENGTH,
    )
    with pytest.raises(ValueError):
        # Check invalid frequencies.
        evolve(valid_hamiltonian, freq=freq)
    with pytest.raises(ValueError):
        # Check invalid wavelengths.
        evolve(valid_hamiltonian, wavelength=wavelength)
    # NOTE: The lattice_depth and interaction_strength can take, in principle,
    #  any real value.
    # Raise an error with NaNs.
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, lattice_depth=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, freq=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, wavelength=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, interaction_strength=np.nan)


@given(
    lattice_depth=stg.floats(min_value=-1e2, max_value=1e2, allow_nan=False),
    freq=stg.floats(min_value=1, max_value=1e2, allow_nan=False),
    wavelength=stg.floats(min_value=1e-2, max_value=1e2, allow_nan=False),
    interaction_strength=stg.floats(
        min_value=-1e-2, max_value=1e2, allow_nan=False
    ),
)
@settings(max_examples=3, deadline=None)
def test_olht(
    lattice_depth: float,
    freq: float,
    wavelength: float,
    interaction_strength: float,
):
    """Check the Hamiltonian core functionality works correctly."""
    hamiltonian = OLHTHamiltonian(
        lattice_depth=lattice_depth,
        freq=freq,
        wavelength=wavelength,
        interaction_strength=interaction_strength,
    )
    domain_array = _domain_mesh.array
    ho_pot_func = hamiltonian.external_potential
    func_array = ho_pot_func(domain_array)
    assert func_array.shape == domain_array.shape


# Multi-rod Hamiltonian parameters.
LATTICE_DEPTH = 1
LATTICE_PERIOD = 1
BARRIER_WIDTH = 1 / 2
INTERACTION_STRENGTH = 1

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
        interaction_strength=INTERACTION_STRENGTH,
    )
    with pytest.raises(ValueError):
        # Check invalid lattice periods and barrier widths.
        evolve(
            valid_hamiltonian,
            lattice_period=params.lattice_period,
            barrier_width=params.barrier_width,
        )
    # NOTE: The lattice_depth and interaction_strength can take, in principle,
    #  any real value.
    # Raise an error with NaNs.
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, lattice_depth=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, lattice_period=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, barrier_width=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_hamiltonian, interaction_strength=np.nan)


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
    interaction_strength=stg.floats(
        min_value=-1e-2, max_value=1e2, allow_nan=False
    ),
)
@settings(max_examples=3, deadline=None)
def test_mr(
    lattice_depth: float,
    params: _MRParams,
    interaction_strength: float,
):
    """Check the Hamiltonian core functionality works correctly."""
    hamiltonian = MRHamiltonian(
        lattice_depth=lattice_depth,
        lattice_period=params.lattice_period,
        barrier_width=params.barrier_width,
        interaction_strength=interaction_strength,
    )
    # Some natural but important assertions about the Hamiltonian attributes.
    assert (
        hamiltonian.well_width + hamiltonian.barrier_width
    ) == pytest.approx(hamiltonian.lattice_period, 1e-8)
    assert hamiltonian.interaction_factor == interaction_strength
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


@given(
    delta_strength=stg.floats(min_value=1e-2, max_value=1e2, allow_nan=False),
    interaction_strength=stg.floats(
        min_value=-1e-2, max_value=1e2, allow_nan=False
    ),
    lattice_period=stg.floats(min_value=1, max_value=1e2, allow_nan=False),
    lower_bound=stg.integers(min_value=-100, max_value=100),
    num_segments_half=stg.integers(min_value=1, max_value=256),
)
@settings(max_examples=20, deadline=20000)
def test_dc_hamiltonian(
    delta_strength: float,
    interaction_strength: float,
    lattice_period: int,
    lower_bound: float,
    num_segments_half: int,
):
    """Check the core functionality of `SuperDCHamiltonian` class."""
    # Non-uniform Dirac-comb potential are closely tied to the mesh where
    # we evaluate them.
    upper_bound = lower_bound + lattice_period
    num_segments = 2 * num_segments_half
    mesh = Mesh(lower_bound, upper_bound, num_segments=num_segments)
    hamiltonian = DCHamiltonian(
        delta_strength=delta_strength,
        lattice_period=lattice_period,
        interaction_strength=interaction_strength,
    )

    # Some natural but important assertions about the Hamiltonian attributes.
    assert hamiltonian.interaction_factor == interaction_strength

    # Check that the external potential function works correctly.
    domain_array = mesh.array
    potential_func = hamiltonian.external_potential
    potential_array = potential_func(domain_array)
    assert potential_array.shape == domain_array.shape

    # For a 1D array, take only the first tuple element returned by
    # ``nonzero`` method.
    nonzero_potential_indices = potential_array.nonzero()[0]
    assert len(nonzero_potential_indices) == 1

    # The nonzero elements of the `potential_array` array represent the
    # delta barriers. They must be equally spaced by the same number of
    # domain mesh segments.
    assert np.all(np.diff(nonzero_potential_indices) == num_segments)

    # Show some debug information.
    note(f"{mesh.array}")
    note(f"{potential_array}")
    note(f"{nonzero_potential_indices}")


# Fix the lower and upper bounds to test the
LOWER_BOUND = -0.5
UPPER_BOUND = 0.5


@given(
    num_deltas=stg.integers(min_value=1, max_value=32),
    interaction_strength=stg.floats(
        min_value=-1e-2, max_value=1e2, allow_nan=False
    ),
)
@settings(max_examples=20, deadline=20000)
def test_super_dc_hamiltonian(
    num_deltas: int,
    interaction_strength: float,
):
    """Check the core functionality of `SuperDCHamiltonian` class."""
    # Non-uniform Dirac-comb potential are closely tied to the mesh where
    # we evaluate them.
    num_inter_delta_segments = 2 * 9
    num_segments = num_deltas * num_inter_delta_segments
    mesh = Mesh(LOWER_BOUND, UPPER_BOUND, num_segments=num_segments)
    _rel_positions = np.linspace(0, 1, num=num_deltas, endpoint=False)
    _strengths = 2 * np.ones_like(_rel_positions)
    deltas_seq = [
        DeltaSpec(strength=strength, rel_position=rel_pos)
        for strength, rel_pos in zip(_strengths, _rel_positions)
    ]
    lattice_period = UPPER_BOUND - LOWER_BOUND
    hamiltonian = SuperDCHamiltonian(
        deltas_seq=deltas_seq,
        lattice_period=lattice_period,
        interaction_strength=interaction_strength,
    )

    # Some natural but important assertions about the Hamiltonian attributes.
    assert hamiltonian.interaction_factor == interaction_strength

    # Check that the external potential function works correctly.
    domain_array = mesh.array
    potential_func = hamiltonian.external_potential
    potential_array = potential_func(domain_array)
    assert potential_array.shape == domain_array.shape

    # For a 1D array, take only the first tuple element returned by
    # ``nonzero`` method.
    nonzero_potential_indices = potential_array.nonzero()[0]
    assert len(nonzero_potential_indices) == num_deltas

    # The nonzero elements of the `potential_array` array represent the
    # delta barriers. They must be equally spaced by the same number of
    # domain mesh segments.
    assert np.all(
        np.diff(nonzero_potential_indices) == num_inter_delta_segments
    )

    # Show some debug information.
    note(f"{mesh.array}")
    note(f"{potential_array}")
    note(f"{nonzero_potential_indices}")
