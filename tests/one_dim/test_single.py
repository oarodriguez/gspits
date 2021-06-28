import numpy as np
import pytest
from attr import evolve
from hypothesis import given, settings
from hypothesis import strategies as stg

from gspits import Mesh
from gspits.one_dim import HTHamiltonian

# Some parameter values that define a valid harmonic trap potential.
MASS = 1
FREQ = 1
SCAT_LENGTH = 1
NUM_BOSONS = 1000


@given(
    mass=stg.floats(allow_infinity=False, allow_nan=False, max_value=0),
    freq=stg.floats(allow_infinity=False, allow_nan=False, max_value=0),
    num_bosons=stg.integers(min_value=-1_000_000, max_value=1),
)
def test_ht_invalid_params(mass: float, freq: float, num_bosons: int):
    """Check that invalid parameters are correctly managed."""
    valid_ho = HTHamiltonian(
        mass=MASS, freq=FREQ, scat_length=SCAT_LENGTH, num_bosons=NUM_BOSONS
    )
    with pytest.raises(ValueError):
        # Check invalid masses.
        evolve(valid_ho, mass=mass)
    with pytest.raises(ValueError):
        # Check invalid frequencies.
        evolve(valid_ho, freq=freq)
    with pytest.raises(ValueError):
        # Check invalid number of bosons.
        evolve(valid_ho, num_bosons=num_bosons)
    # NOTE: The scattering length can take, in principle, any real value.
    # Raise an error with NaNs.
    with pytest.raises(ValueError):
        evolve(valid_ho, mass=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_ho, freq=np.nan)
    with pytest.raises(ValueError):
        evolve(valid_ho, scat_length=np.nan)


# TODO: Maybe this mesh should be a fixture with arbitrary bounds and
#  number of steps. How?
_domain_mesh = Mesh(lower_bound=-10, upper_bound=10, num_segments=128)


@given(
    mass=stg.floats(min_value=1, max_value=1e2, allow_nan=False),
    freq=stg.floats(min_value=1, max_value=1e2, allow_nan=False),
    scat_length=stg.floats(min_value=-1e-2, max_value=1e2, allow_nan=False),
    num_bosons=stg.integers(min_value=3, max_value=1_000_000),
)
@settings(max_examples=3, deadline=None)
def test_ht(mass: float, freq: float, scat_length: float, num_bosons: int):
    """Check that the core function works."""
    ho = HTHamiltonian(
        mass=mass, freq=freq, scat_length=scat_length, num_bosons=num_bosons
    )
    domain_array = _domain_mesh.as_array()
    ho_pot_func = ho.external_potential
    func_array = ho_pot_func(domain_array)
    assert func_array.shape == domain_array.shape
