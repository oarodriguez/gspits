import numpy as np
import pytest
from attr import dataclass
from hypothesis import given, settings
from hypothesis import strategies as stg
from numba import njit

from gspits import SpatialMesh
from gspits.one_dim import (
    ExternalPotential,
    HarmonicOscillator,
    SupportsExternalPotential,
    external_potential,
)


@njit
def _ho_potential(domain_mesh: np.ndarray) -> np.ndarray:
    """Evaluate the harmonic potential in the mesh."""
    return 0.5 * domain_mesh ** 2


@dataclass(frozen=True)
class Potential:
    """Represent an dummy harmonic oscillator potential in 1D."""

    def __external_potential__(self) -> ExternalPotential:
        """Get a callable that evaluates the harmonic potential."""
        return _ho_potential


@dataclass(frozen=True)
class PotentialSC(SupportsExternalPotential):
    """Represent an dummy harmonic oscillator potential in 1D."""

    def __external_potential__(self) -> ExternalPotential:
        """Get a callable that evaluates the harmonic potential."""
        return _ho_potential


def test_potential():
    """Verify subtyping relationships."""
    ho_pot = Potential()
    assert isinstance(external_potential(ho_pot), ExternalPotential)
    assert isinstance(ho_pot, SupportsExternalPotential)


def test_potential_sc():
    """Verify subtyping relationships with explicit subclassing."""
    ho_pot = PotentialSC()
    assert isinstance(external_potential(ho_pot), ExternalPotential)
    assert isinstance(ho_pot, SupportsExternalPotential)


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
