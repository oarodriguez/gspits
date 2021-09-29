"""Validate the functionality of the `gspits.system` module."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as stg
from numpy import pi

from gspits import BlochState, ExternalPotential, Mesh, Partition, State


def _make_partition(lower_bound: float, size: float, num_segments: int):
    """Make a spatial partition."""
    return Partition(
        lower_bound=lower_bound,
        upper_bound=lower_bound + size,
        num_segments=num_segments,
    )


# For our tests, we first define a strategy to create spatial partitions,
# i.e., `Partitions` instances.
partitions_stg = stg.builds(
    _make_partition,
    lower_bound=stg.floats(min_value=-10, max_value=10, allow_nan=False),
    size=stg.floats(min_value=1, max_value=10, allow_nan=False),
    num_segments=stg.integers(min_value=1, max_value=64),
)


def _make_mesh(partitions: list[Partition]):
    """Make a spatial mesh."""
    return Mesh(partitions=tuple(partitions))


# Next, we define a strategy to create `Mesh` instances with a variable
# number of partitions using a `lists` hypothesis strategy with the correct
# number of partitions.
meshes_stg = stg.builds(
    _make_mesh, stg.lists(partitions_stg, min_size=1, max_size=3)
)


@given(mesh=meshes_stg)
def test_state(mesh: Mesh):
    """Check behavior of `State` instances."""
    # Verify that a `State` mesh and wave functions have consistent shapes.
    with pytest.raises(ValueError):
        # Wave function BAD.
        wave_func = np.ones(mesh.shape + (10,))
        State(mesh=mesh, wave_func=wave_func)

    # Build a normalized, constant amplitude state.
    wave_func = np.ones(mesh.shape) / np.sqrt(mesh.size)
    state = State(mesh=mesh, wave_func=wave_func)
    assert state.norm == pytest.approx(state.norm, abs=1e-8)


@given(
    mesh=meshes_stg,
    wave_vector_val=stg.floats(
        min_value=-3 * pi, max_value=3 * pi, allow_nan=False
    ),
)
def test_bloch_state(
    mesh: Mesh,
    wave_vector_val: float,
):
    """Check behavior of `BlochState` instances."""
    # Verify that a `BlochState` mesh, wave function, and wave vector
    # have consistent shapes.
    with pytest.raises(ValueError):
        # Wave function BAD.
        wave_func = np.ones(mesh.shape + (10,))
        # Wave vector OK.
        wave_vector = (wave_vector_val,) * mesh.dimension
        BlochState(
            mesh=mesh, periodic_wave_func=wave_func, wave_vector=wave_vector
        )
    with pytest.raises(ValueError):
        # Wave vector BAD.
        wave_vector = (wave_vector_val,) * (mesh.dimension + 1)
        BlochState.plane_wave(mesh=mesh, wave_vector=wave_vector)

    # Build states with consistent attributes.
    wave_vector = (wave_vector_val,) * mesh.dimension
    bloch_state = BlochState.plane_wave(mesh, wave_vector)

    # A plane wave is normalized, and its norm must be approximately one.
    assert bloch_state.norm == pytest.approx(1, abs=1e-8)
    assert bloch_state.norm == pytest.approx(
        bloch_state.periodic_component.norm, abs=1e-8
    )


def _potential_one(mesh: Mesh) -> np.ndarray:
    """Evaluate an harmonic potential over a mesh."""
    (x_mesh,) = mesh.arrays
    return 0.5 * x_mesh ** 2


class _PotentialTwo(ExternalPotential):
    """Evaluate an harmonic potential over a mesh."""

    def __call__(self, mesh: Mesh) -> np.ndarray:
        """Callable interface."""
        (x_mesh,) = mesh.arrays
        return 0.5 * x_mesh ** 2


# Strategy to generate 1D meshes and test the previously defined functions
# and classes.
meshes_1d_stg = stg.builds(
    _make_mesh, stg.lists(partitions_stg, min_size=1, max_size=1)
)


@given(mesh=meshes_1d_stg)
def test_external_potential(mesh: Mesh):
    """Check behavior of `ExternalPotential` instances."""
    # Passes, since _potential_one is a callable object and the runtime
    # check done by `isinstance` only check if a __call__ attribute exists.
    # Function call also passes.
    assert isinstance(_potential_one, ExternalPotential)
    _potential_one(mesh)

    # Assert and function call pass.
    potential_two = _PotentialTwo()
    assert isinstance(potential_two, ExternalPotential)
    potential_two(mesh)
