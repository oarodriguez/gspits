"""Harmonic potential in one, two, and three dimensions."""

from itertools import zip_longest
from math import sqrt

import numpy as np
from attr import dataclass
from numpy import pi

from gspits import ExternalPotential, Mesh, State

__all__ = ["HarmonicTrap"]


@dataclass(frozen=True)
class HarmonicTrap(ExternalPotential):
    """:math:`N`-dimensional harmonic trap external potential.

    :param tuple[float, ...] frequencies:
        The trap angular frequencies. It must be a tuple of non-zero,
        positive numbers.

    :raises ValueError:
        If any of the frequencies is negative or zero.
    :raises ValueError:
        If any of the frequencies is a ``nan``.
    """

    # Trap angular frequency.
    frequencies: tuple[float, ...]

    def __attrs_post_init__(self) -> None:
        """Post-initialization checks."""
        # Avoid NaNs.
        if np.any(np.isnan(self.frequencies)):
            raise ValueError(
                "some of the trap frequencies are 'nan', which is "
                "not a valid value."
            )
        # TODO: Maybe we can relax this condition and allow zero-value
        #  frequencies.
        if not np.all(np.asarray(self.frequencies) > 0):
            raise ValueError(
                "all the frequencies must have positive, non-zero values."
            )

        # Make transformations.
        object.__setattr__(self, "frequencies", tuple(self.frequencies))

    @property
    def sizes(self) -> tuple[float, ...]:
        """Characteristic sizes of the trap.

        The returned tuple contains the corresponding size for each of the
        trap frequencies.

        :rtype: tuple[float, ...]
        """
        return tuple([sqrt(1 / freq) for freq in self.frequencies])

    def gaussian_state(self, mesh: Mesh) -> State:
        """Build a normalized Gaussian state.

        :param Mesh mesh:
            A mesh where the state will be defined.
        :rtype: State

        :raises ValueError:
            If the ``mesh`` is incompatible with the trap dimension.
        """
        if len(self.frequencies) != mesh.dimension:
            raise ValueError(
                "'mesh' is incompatible with the current potential, since "
                f"the mesh dimension is '{mesh.dimension}, and the potential "
                f"has '{len(self.frequencies)}' frequencies."
            )
        exp_arg: np.ndarray = np.zeros(mesh.shape, dtype=np.float64)
        for freq, mesh_array in zip_longest(self.frequencies, mesh.arrays):
            exp_arg += freq * mesh_array ** 2

        # The full wave function is a product of one-dimensional
        # gaussian wave functions.
        freq_prod = np.prod(self.frequencies)
        wave_func = (freq_prod / pi ** mesh.dimension) ** 0.25 * np.exp(
            -exp_arg / 2
        )
        return State(mesh=mesh, wave_func=wave_func)

    def __call__(self, mesh: Mesh) -> np.ndarray:
        """External potential callable interface.

        :param mesh:
            A ``Mesh`` instance representing a domain mesh.
        :rtype: numpy.ndarray

        :raises ValueError:
            If the ``mesh`` is incompatible with the trap dimension.
        """
        if len(self.frequencies) != mesh.dimension:
            raise ValueError(
                "'mesh' is incompatible with the current potential, since "
                f"the mesh dimension is '{mesh.dimension}, and the potential "
                f"has '{len(self.frequencies)}' frequencies."
            )
        potential_contrib: np.ndarray = np.zeros(mesh.shape, dtype=np.float64)
        for freq, mesh_array in zip_longest(self.frequencies, mesh.arrays):
            potential_contrib += (freq * mesh_array) ** 2
        return 0.5 * potential_contrib
