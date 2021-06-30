import numpy as np
import pytest
from attr import evolve
from hypothesis import given, settings
from hypothesis import strategies as stg

from gspits import Mesh
from gspits.one_dim import HTHamiltonian

# Some parameter values that define a valid harmonic trap potential.
FREQ = 1
SCAT_LENGTH = 1
NUM_BOSONS = 1000


@given(
    freq=stg.floats(allow_infinity=False, allow_nan=False, max_value=0),
    num_bosons=stg.integers(min_value=-1_000_000, max_value=1),
)
def test_ht_invalid_params(freq: float, num_bosons: int):
    """Check that invalid parameters are correctly managed."""
    valid_ho = HTHamiltonian(
        freq=FREQ, scat_length=SCAT_LENGTH, num_bosons=NUM_BOSONS
    )
    with pytest.raises(ValueError):
        # Check invalid frequencies.
        evolve(valid_ho, freq=freq)
    with pytest.raises(ValueError):
        # Check invalid number of bosons.
        evolve(valid_ho, num_bosons=num_bosons)
    # NOTE: The scattering length can take, in principle, any real value.
    # Raise an error with NaNs.
    with pytest.raises(ValueError):
        evolve(valid_ho, freq=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_ho, scat_length=np.nan)


# TODO: Maybe this mesh should be a fixture with arbitrary bounds and
#  number of steps. How?
_domain_mesh = Mesh(lower_bound=-10, upper_bound=10, num_segments=128)


@given(
    freq=stg.floats(min_value=1, max_value=1e2, allow_nan=False),
    scat_length=stg.floats(min_value=-1e-2, max_value=1e2, allow_nan=False),
    num_bosons=stg.integers(min_value=2, max_value=1_000_000),
)
@settings(max_examples=3, deadline=None)
def test_ht(freq: float, scat_length: float, num_bosons: int):
    """Check that the core function works."""
    ho = HTHamiltonian(
        freq=freq, scat_length=scat_length, num_bosons=num_bosons
    )
    domain_array = _domain_mesh.array
    ho_pot_func = ho.external_potential
    func_array = ho_pot_func(domain_array)
    assert func_array.shape == domain_array.shape


@given(num_segments=stg.integers(min_value=32, max_value=128))
def test_ht_gaussian_state(num_segments: int):
    """Check that Gaussian states are normalized.

    The normalization condition accuracy depends on the number of
    steps in the mesh. Here, we test for a minimum value of 32 steps
    and an absolute tolerance for the state norm of 1e-8.
    NOTE: How can we improve this test?
    """
    valid_ho = HTHamiltonian(
        freq=1 * FREQ,
        scat_length=SCAT_LENGTH,
        num_bosons=NUM_BOSONS,
    )
    mesh = Mesh(lower_bound=-10, upper_bound=10, num_segments=num_segments)
    state = valid_ho.gaussian_state(mesh)
    assert state.norm == pytest.approx(1, abs=1e-8)
