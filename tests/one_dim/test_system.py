import hypothesis.strategies as st
import numpy as np
import pytest
from attr import dataclass
from hypothesis import given
from numba import njit
from numpy import pi

from gspits import Mesh
from gspits.one_dim import ExternalPotential, Hamiltonian, State


@njit
def _ho_potential(domain_mesh: np.ndarray) -> np.ndarray:
    """Evaluate the harmonic potential in the mesh."""
    return 0.5 * domain_mesh ** 2


@dataclass(frozen=True, slots=True)
class _TestHamiltonian:
    """Define a testing Hamiltonian."""

    __slots__ = ()

    @property
    def int_factor(self):
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


def gaussian_state(mesh: Mesh):
    """Build a normalized Gaussian state.

    :param mesh: A mesh where the state will be defined.
    :return: A corresponding :class:`State` instance.
    """
    domain_mesh = mesh.array
    wave_func = np.exp(-(domain_mesh ** 2) / 2) / pi ** 0.25
    return State(mesh=mesh, wave_func=wave_func)


@given(num_segments=st.integers(min_value=32, max_value=128))
def test_state(num_segments: int):
    """Check that Gaussian states are normalized.

    The normalization condition accuracy depends on the number of
    steps in the mesh. Here, we test for a minimum value of 32 steps
    and an absolute tolerance for the state norm of 1e-8.
    NOTE: How can we improve this test?
    """
    spatial_mesh = Mesh(
        lower_bound=-10, upper_bound=10, num_segments=num_segments
    )
    state = gaussian_state(spatial_mesh)
    assert state.norm == pytest.approx(1, abs=1e-8)
