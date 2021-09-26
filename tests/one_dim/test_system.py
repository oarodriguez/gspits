"""Validate the functionality of the `gspits.one_dim.system` module."""

import numpy as np
import pytest
from attr import dataclass
from hypothesis import given
from hypothesis import strategies as stg
from numba import njit
from numpy import pi

from gspits import Partition as Mesh
from gspits.one_dim import BlochState, ExternalPotential, Hamiltonian


@njit
def _ho_potential(domain_mesh: np.ndarray) -> np.ndarray:
    """Evaluate the harmonic potential in the mesh."""
    return 0.5 * domain_mesh ** 2


@dataclass(frozen=True, slots=True)
class _TestHamiltonian:
    """Define a testing Hamiltonian."""

    __slots__ = ()

    @property
    def interaction_factor(self):
        """Bose gas interaction factor."""
        return 4 * pi * 1e-2

    @property
    def external_potential(self):
        """External potential function."""
        return _ho_potential


def test_hamiltonian():
    """Verify subtyping relationships for Hamiltonian instances."""
    hamiltonian = _TestHamiltonian()
    assert isinstance(hamiltonian.external_potential, ExternalPotential)
    assert isinstance(hamiltonian, Hamiltonian)


@given(
    lower_bound=stg.floats(min_value=-2, max_value=-1, allow_nan=False),
    upper_bound=stg.floats(min_value=1, max_value=2, allow_nan=False),
    num_segments=stg.integers(min_value=128, max_value=512),
    wave_vector=stg.floats(
        min_value=-3 * pi, max_value=3 * pi, allow_nan=False
    ),
)
def test_bloch_state(
    lower_bound: float,
    upper_bound: float,
    num_segments: int,
    wave_vector: float,
):
    """Check the basic functionality of `BlochState` instances."""
    mesh = Mesh(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_segments=num_segments,
    )
    bloch_state = BlochState.plane_wave(mesh, wave_vector)
    # A plane wave is normalized, and its norm must be approximately one.
    assert bloch_state.norm == pytest.approx(1, abs=1e-8)
    assert bloch_state.norm == pytest.approx(
        bloch_state.periodic_component.norm, abs=1e-8
    )
