import pytest
from hypothesis import given, settings
from hypothesis import strategies as stg

from gspits import SpatialMesh
from gspits.one_dim import HarmonicOscillator, external_potential

valid_mass_stg = stg.floats(
    min_value=0,
    max_value=1e3,
    exclude_min=True,
    allow_nan=False,
    allow_infinity=False,
)
invalid_mass_stg = stg.floats(
    allow_infinity=False, allow_nan=False, max_value=0
)
valid_freq_stg = valid_mass_stg
invalid_freq_stg = invalid_mass_stg


@given(mass=invalid_mass_stg, freq=valid_freq_stg)
def test_ho_invalid_mass(mass: float, freq: float):
    """Check for invalid masses."""
    with pytest.raises(ValueError):
        HarmonicOscillator(mass=mass, freq=freq)


@given(mass=valid_mass_stg, freq=invalid_freq_stg)
def test_ho_invalid_freq(mass: float, freq: float):
    """Check for invalid frequencies."""
    with pytest.raises(ValueError):
        HarmonicOscillator(mass=mass, freq=freq)


# TODO: Maybe this mesh should be a fixture with arbitrary bounds and
#  number of steps. How?
_domain_mesh = SpatialMesh(lower_bound=-10, upper_bound=10, num_steps=128)


@given(
    mass=stg.floats(min_value=1, max_value=1e2, allow_nan=False),
    freq=stg.floats(min_value=1, max_value=1e2, allow_nan=False),
)
@settings(max_examples=3, deadline=None)
def test_ho(mass: float, freq: float):
    """Check that the core function works."""
    ho = HarmonicOscillator(mass=mass, freq=freq)
    domain_array = _domain_mesh.as_array()
    ho_pot_func = external_potential(ho)
    func_array = ho_pot_func(domain_array)
    assert func_array.shape == domain_array.shape
