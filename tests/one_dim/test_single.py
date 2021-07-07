import numpy as np
import pytest
from attr import evolve
from hypothesis import given, settings
from hypothesis import strategies as stg

from gspits import Mesh
from gspits.one_dim import HTHamiltonian

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
