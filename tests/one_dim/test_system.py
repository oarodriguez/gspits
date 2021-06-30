import numpy as np
from attr import dataclass
from numba import njit
from numpy import pi

from gspits.one_dim import ExternalPotential, Hamiltonian


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
