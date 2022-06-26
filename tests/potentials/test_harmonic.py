"""Validate the `gspits.potentials.harmonic` module."""

from fractions import Fraction
from math import pi

import numpy as np
import pytest
from attr import dataclass
from hypothesis import given, settings
from hypothesis.strategies import builds, fractions, integers, lists
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gspits import Mesh, Partition
from gspits.potentials.harmonic import HarmonicTrap


@dataclass(frozen=True)
class _FreqPartition:
    """Pair of frequency and spatial partition."""

    freq: float
    partition: Partition


def _make_freq_partition(freq: Fraction, num_segments: int):
    """Make a spatial partition."""
    partition = Partition.make_origin_centered_unit(num_segments=num_segments)
    return _FreqPartition(freq=freq * np.pi, partition=partition)


# Define a strategy to create spatial partitions.
freq_partitions_stg = builds(
    _make_freq_partition,
    freq=fractions(min_value=1, max_value=32, max_denominator=32),
    num_segments=integers(min_value=32, max_value=64),
)

# Define a strategy to create spatial partitions with valid arguments.
valid_freq_partitions_stg = builds(
    _make_freq_partition,
    freq=fractions(min_value=1, max_value=32, max_denominator=32),
    num_segments=integers(min_value=32, max_value=64),
)


@dataclass(frozen=True)
class FreqsMesh:
    """Pair of frequencies and spatial mesh."""

    freqs: tuple[float, ...]
    mesh: Mesh


def _make_freqs_mesh(freq_partitions: list[_FreqPartition]):
    """Make a spatial mesh."""
    freqs = tuple(fp.freq for fp in freq_partitions)
    partitions = tuple(fp.partition for fp in freq_partitions)
    return FreqsMesh(freqs=freqs, mesh=Mesh(partitions=partitions))


# Define a strategy to create `Mesh` instances with a variable
# number of partitions. We use a `lists` hypothesis strategy with
# the correct number of partitions to generate 1D, 2D, or 3D meshes.
freqs_meshes_stg = builds(
    _make_freqs_mesh, lists(freq_partitions_stg, min_size=1, max_size=3)
)


def test_harmonic_trap_nan():
    """Check that harmonic traps handle NaNs correctly."""
    with pytest.raises(ValueError):
        HarmonicTrap(frequencies=(np.nan,))
    with pytest.raises(ValueError):
        HarmonicTrap(frequencies=(np.nan, 1))
    with pytest.raises(ValueError):
        HarmonicTrap(frequencies=(np.nan, 1, 1))


@given(freqs_mesh=freqs_meshes_stg)
def test_harmonic_trap(freqs_mesh: FreqsMesh):
    """Check the harmonic trap functionality."""
    # Negative frequencies are not allowed, but they raise exceptions.
    freqs = freqs_mesh.freqs
    if np.any(np.asarray(freqs) <= 0):
        with pytest.raises(ValueError):
            HarmonicTrap(frequencies=freqs)
        return

    # We expect a ValueError in the following cases, since trap frequencies
    # and mesh are incompatible.
    mesh = freqs_mesh.mesh
    trap = HarmonicTrap(frequencies=(1.0,) * (mesh.dimension + 1))
    with pytest.raises(ValueError):
        trap(mesh)
    with pytest.raises(ValueError):
        trap.gaussian_state(mesh)

    # Check the behavior of valid harmonic trap.
    trap = HarmonicTrap(frequencies=freqs)

    # The potential evaluation must succeed.
    potential_array = trap(mesh)
    assert potential_array.shape == mesh.shape

    # Do Gaussian states work correctly?
    # NOTE: A gaussian state norm should approximate its correct value only
    #  if the mesh is significantly larger than the trap size.
    scaling_factors = tuple(size * 10 for size in trap.sizes)
    scaled_mesh = mesh.scaled(scaling_factors)
    gaussian_state = trap.gaussian_state(scaled_mesh)
    assert gaussian_state.wave_func.shape == scaled_mesh.shape
    assert gaussian_state.norm == pytest.approx(1)


# Strategy to generate 1D meshes for visualization.
valid_freqs_meshes_1d_stg = builds(
    _make_freqs_mesh, lists(valid_freq_partitions_stg, min_size=1, max_size=1)
)


@pytest.mark.is_end_to_end_test
@pytest.mark.is_interactive_test
@given(freqs_mesh=valid_freqs_meshes_1d_stg)
@settings(deadline=10000, max_examples=10)
def test_harmonic_trap_1d_plot(freqs_mesh: FreqsMesh):
    """Visualize a one-dimensional harmonic trap."""
    freqs = freqs_mesh.freqs
    base_mesh = freqs_mesh.mesh
    trap = HarmonicTrap(frequencies=freqs)
    scaling_factors = tuple(size * 10 for size in trap.sizes)
    mesh = base_mesh.scaled(scaling_factors)
    potential_array = trap(mesh)

    fig: Figure = pyplot.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    ax.set_title(rf"$\omega = {freqs[0] / pi:.5G} \pi$")

    # Since the mesh arrays are sparse, we have to broadcast them against
    # each order to visualize the potential (and other functions). For 1D
    # potentials, this is not strictly necessary, but as it works for 2D and
    # 3D potentials, we do it here.
    coord_arrays = np.broadcast_arrays(*mesh.arrays)
    ax.plot(*coord_arrays, potential_array)
    ax.set_xlabel(r"$x / a_c$")
    ax.set_ylabel(r"$V(z) / \hbar \omega_c$")

    pyplot.tight_layout()
    pyplot.show()


# Strategy to generate 2D meshes for visualization.
valid_freqs_meshes_2d_stg = builds(
    _make_freqs_mesh, lists(valid_freq_partitions_stg, min_size=2, max_size=2)
)


@pytest.mark.is_end_to_end_test
@pytest.mark.is_interactive_test
@given(freqs_mesh=valid_freqs_meshes_2d_stg)
@settings(deadline=10000, max_examples=10)
def test_harmonic_trap_2d_plot(freqs_mesh: FreqsMesh):
    """Visualize a two-dimensional harmonic trap."""
    freqs = freqs_mesh.freqs
    base_mesh = freqs_mesh.mesh
    trap = HarmonicTrap(frequencies=freqs)
    scaling_factors = tuple(size * 10 for size in trap.sizes)
    mesh = base_mesh.scaled(scaling_factors)
    potential_array = trap(mesh)

    fig: Figure = pyplot.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    ax.set_title(
        rf"$\omega_x = {freqs[0] / pi:.5G} \pi$, "
        rf"$\omega_y = {freqs[1] / pi:.5G} \pi$"
    )

    # Since the mesh arrays are sparse, we have to broadcast them against
    # each order to visualize the potential (and other functions).
    coord_arrays = np.broadcast_arrays(*mesh.arrays)
    contourf_map = ax.contourf(*coord_arrays, potential_array, levels=10)
    contour_map = ax.contour(
        *coord_arrays, potential_array, levels=10, cmap="hot"
    )
    colorbar = fig.colorbar(contourf_map, ax=ax)
    colorbar.add_lines(contour_map)
    colorbar.set_label(r"$V(x, y) / \hbar \omega_c$")
    ax.set_xlabel(r"$x / a_c$")
    ax.set_ylabel(r"$y / a_c$")

    pyplot.tight_layout()
    pyplot.show()
