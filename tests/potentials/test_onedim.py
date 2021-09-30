"""Validate the `gspits.potentials.onedim` module."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import builds, floats, integers, lists

from gspits import Mesh, Partition
from gspits.potentials.onedim import HarmonicTrap


def _make_partition(num_segments: int):
    """Make a spatial partition."""
    return Partition.make_origin_centered_unit(
        num_segments=num_segments,
    )


# First, we define a strategy to create spatial partitions. The number of
# segments
coarse_partitions = builds(
    _make_partition,
    num_segments=integers(min_value=16, max_value=64),
)


def _make_mesh(partitions: list[Partition]):
    """Make a spatial mesh."""
    return Mesh(partitions=tuple(partitions))


# Next, we define a strategy to create `Mesh` instances with a variable
# number of partitions using a `lists` hypothesis strategy with the correct
# number of partitions.
coarse_meshes_stg = builds(
    _make_mesh, lists(coarse_partitions, min_size=1, max_size=1)
)


def test_harmonic_trap_nan():
    """Check that harmonic traps handle NaNs correctly."""
    with pytest.raises(ValueError):
        HarmonicTrap(freq=np.nan)


@given(freq=floats(min_value=-10, max_value=10), mesh=coarse_meshes_stg)
def test_harmonic_trap(freq: float, mesh: Mesh):
    """Check the harmonic trap functionality."""
    # Negative frequencies are not allowed, but they raise exceptions.
    if freq <= 0:
        with pytest.raises(ValueError):
            HarmonicTrap(freq=freq)
        return

    # Check the behavior of valid harmonic trap.
    trap = HarmonicTrap(freq=freq)

    # The potential evaluation must succeed.
    potential_array = trap(mesh)
    assert potential_array.shape == mesh.shape

    # Do Gaussian states work correctly?
    # NOTE: A gaussian state norm should approximate its correct value only
    #  if the mesh is significantly larger than the trap size.
    scaling_factors = (trap.size * 10,)
    scaled_mesh = mesh.scaled(scaling_factors)
    gaussian_state = trap.gaussian_state(scaled_mesh)
    assert gaussian_state.wave_func.shape == scaled_mesh.shape
    assert gaussian_state.norm == pytest.approx(1)
