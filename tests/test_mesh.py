"""Verify the routines in ``gspits.mesh`` module."""

import numpy as np
import pytest
from attr import evolve
from hypothesis import given
from hypothesis import strategies as stg

from gspits import Mesh, Partition, TimePartition

valid_lower_bound_stg = stg.floats(min_value=0, max_value=2)
valid_upper_bound_stg = stg.floats(min_value=3, max_value=4)
valid_num_segments_steps_stg = stg.integers(min_value=1, max_value=512)


def test_partition_nan():
    """Check that partitions handle NaNs correctly."""
    with pytest.raises(ValueError):
        Partition(lower_bound=np.nan, upper_bound=1, num_segments=2)
    with pytest.raises(ValueError):
        Partition(lower_bound=0, upper_bound=np.nan, num_segments=2)
    with pytest.raises(ValueError):
        Partition(lower_bound=np.nan, upper_bound=np.nan, num_segments=2)


@given(lower_bound=stg.floats(min_value=100, max_value=200))
def test_partition(lower_bound: float):
    """Test spatial partitions with invalid bounds."""
    with pytest.raises(ValueError):
        Partition(lower_bound, upper_bound=0, num_segments=128)


@given(num_segments=stg.integers(min_value=-1000, max_value=0))
def test_partition_num_steps(num_segments: int):
    """Test spatial partitions with invalid number of steps."""
    with pytest.raises(ValueError):
        Partition(lower_bound=0, upper_bound=1, num_segments=num_segments)


@given(
    lower_bound=valid_lower_bound_stg,
    upper_bound=valid_upper_bound_stg,
    num_segments=valid_num_segments_steps_stg,
)
def test_partition_bounds(
    lower_bound: float, upper_bound: float, num_segments: int
):
    """Test routines for creating spatial partitions."""
    # Check partition without the endpoint.
    partition = Partition(lower_bound, upper_bound, num_segments)
    partition_array = partition.array
    assert partition_array.min() == partition.lower_bound
    assert np.allclose(np.diff(partition_array), partition.step_size)

    # Check partition with the endpoint included.
    partition = evolve(partition, endpoint=True)
    partition_array = partition.array
    assert partition_array.max() == partition.upper_bound
    assert np.allclose(np.diff(partition_array), partition.step_size)

    # Check the partition size.
    # There is a very small round off error in this test.
    assert partition.size == pytest.approx(
        partition.step_size * partition.num_segments, abs=1e-8
    )
    assert partition.size == (partition.upper_bound - partition.lower_bound)


@given(
    location=stg.floats(min_value=-2e3, max_value=2e3, allow_nan=False),
    size=stg.floats(min_value=1, max_value=1e3, allow_nan=False),
    num_segments=stg.integers(min_value=1, max_value=512),
    scale=stg.floats(min_value=1, max_value=2, allow_nan=False),
    offset=stg.floats(min_value=-1e2, max_value=1e2, allow_nan=False),
)
def test_partition_transformations(
    location: float,
    size: float,
    num_segments: int,
    scale: float,
    offset: float,
):
    """Check routines to transform partitions."""
    # Reference partition.
    partition = Partition.with_size(
        size=size, lower_bound=location, num_segments=num_segments
    )
    # Centered partition at the origin.
    centered_partition = partition.origin_centered()

    # Check that the important attributes are consistent among the original
    # partition and the centered one.
    assert partition.num_segments == centered_partition.num_segments
    assert partition.endpoint == centered_partition.endpoint
    assert partition.size == pytest.approx(centered_partition.size)
    assert centered_partition.lower_bound == pytest.approx(
        -centered_partition.upper_bound
    )
    assert centered_partition.midpoint == pytest.approx(0)

    # Unit partition located at the origin.
    unit_partition = partition.origin_centered_unit()
    size_quotient = unit_partition.size / partition.size
    step_size_quotient = unit_partition.step_size / partition.step_size
    assert partition.num_segments == unit_partition.num_segments
    assert partition.endpoint == unit_partition.endpoint
    assert unit_partition.size == 1.0
    assert size_quotient == pytest.approx(step_size_quotient)

    # Scaled partition located at the origin.
    scaled_partition = partition.scaled(factor=scale)
    size_quotient = scaled_partition.size / partition.size
    step_size_quotient = scaled_partition.step_size / partition.step_size
    assert partition.num_segments == scaled_partition.num_segments
    assert partition.endpoint == scaled_partition.endpoint
    assert size_quotient == pytest.approx(step_size_quotient)

    # Translated partition.
    translated_partition = partition.translated(offset=offset)
    assert partition.num_segments == translated_partition.num_segments
    assert partition.endpoint == translated_partition.endpoint
    assert partition.size == pytest.approx(translated_partition.size)


def test_time_partition_nan():
    """Check that time partitions handle NaNs correctly."""
    with pytest.raises(ValueError):
        TimePartition(time_step=np.nan, num_steps=2, ini_time=0)
    with pytest.raises(ValueError):
        TimePartition(time_step=1, num_steps=2, ini_time=np.nan)
    with pytest.raises(ValueError):
        TimePartition(time_step=np.nan, num_steps=2, ini_time=np.nan)


# See https://stackoverflow.com/questions/19141432 for details.
valid_time_step_stg = stg.floats(min_value=np.finfo(float).eps, max_value=10)
valid_ini_time_stg = stg.floats(min_value=-128, max_value=128)


@given(
    num_steps=stg.integers(min_value=-1000, max_value=0),
)
def test_time_partition_num_steps(num_steps: int):
    """Check temporal partitions with an invalid number of steps."""
    with pytest.raises(ValueError):
        TimePartition(time_step=1e-2, num_steps=num_steps)


@given(
    time_step=stg.floats(min_value=-1e2, max_value=0),
)
def test_time_partition_time_step(time_step: float):
    """Check temporal partitions with an invalid time step."""
    with pytest.raises(ValueError):
        TimePartition(time_step=time_step, num_steps=128)


@given(
    time_step=valid_time_step_stg,
    num_steps=valid_num_segments_steps_stg,
    ini_time=valid_ini_time_stg,
)
def test_time_partition(time_step: float, num_steps: int, ini_time: float):
    """Check routines for creating temporal partitions."""
    # Check partition without the endpoint.
    time_partition = TimePartition(
        time_step=time_step, num_steps=num_steps, ini_time=ini_time
    )
    partition_array = time_partition.array
    assert partition_array.min() == time_partition.ini_time
    assert np.allclose(np.diff(partition_array), time_partition.time_step)

    # Check partition with the endpoint included.
    time_partition = evolve(time_partition, endpoint=True)
    partition_array = time_partition.array
    assert partition_array.max() == time_partition.finish_time
    assert np.allclose(np.diff(partition_array), time_partition.time_step)

    # Check the partition duration.
    # There is a very small round off error in this test.
    assert time_partition.duration == pytest.approx(
        time_partition.time_step * time_partition.num_steps, abs=1e-8
    )
    assert time_partition.duration == (
        time_partition.finish_time - time_partition.ini_time
    )


@given(
    lower_bound=stg.floats(min_value=0, max_value=2),
    size=stg.floats(min_value=1, max_value=10),
    num_segments=stg.integers(min_value=1, max_value=512),
)
def test_mesh_init(lower_bound: float, size: float, num_segments: int):
    """Check ``Mesh`` instances initialization."""
    upper_bound = lower_bound + size
    partition_x = Partition(lower_bound, upper_bound, num_segments)
    partition_y = Partition(lower_bound, upper_bound, num_segments)
    partition_z = Partition(lower_bound, upper_bound, num_segments)

    with pytest.raises(ValueError):
        # A mesh can have three dimensions at most.
        Mesh((partition_x, partition_y, partition_z, partition_x))


@given(
    lower_bound=stg.floats(min_value=0, max_value=2),
    size=stg.floats(min_value=1, max_value=10),
    num_segments=stg.integers(min_value=1, max_value=512),
)
def test_mesh_elements(lower_bound: float, size: float, num_segments: int):
    """Check ``Mesh`` functionality concerning its elements."""
    upper_bound = lower_bound + size
    partition_x = Partition(lower_bound, upper_bound, num_segments)
    partition_y = Partition(lower_bound, upper_bound, num_segments)
    partition_z = Partition(lower_bound, upper_bound, num_segments)

    # Checks for a one-dimensional mesh.
    partitions_1d = (partition_x,)
    mesh = Mesh(partitions_1d)
    assert mesh.size == pytest.approx(mesh.num_elements * mesh.element_size)

    # Checks for a two-dimensional mesh.
    partitions_2d = partition_x, partition_y
    mesh = Mesh(partitions_2d)
    assert mesh.size == pytest.approx(mesh.num_elements * mesh.element_size)

    # Checks for a three-dimensional mesh.
    partitions_3d = partition_x, partition_y, partition_z
    mesh = Mesh(partitions_3d)
    assert mesh.size == pytest.approx(mesh.num_elements * mesh.element_size)


@given(
    lower_bound=stg.floats(min_value=0, max_value=2),
    size=stg.floats(min_value=1, max_value=10),
    num_segments=stg.integers(min_value=1, max_value=512),
)
def test_mesh_shape(lower_bound: float, size: float, num_segments: int):
    """Check consistency of a ``Mesh`` arrays shapes."""
    upper_bound = lower_bound + size
    partition_x = Partition(lower_bound, upper_bound, num_segments)
    partition_y = Partition(lower_bound, upper_bound, num_segments)
    partition_z = Partition(lower_bound, upper_bound, num_segments)

    # Checks for a one-dimensional mesh.
    partitions_1d = (partition_x,)
    mesh = Mesh(partitions_1d)
    assert mesh.dimension == 1
    assert mesh.shape == (partition_x.num_segments,)

    # Checks for a two-dimensional mesh.
    partitions_2d = partition_x, partition_y
    mesh = Mesh(partitions_2d)
    assert mesh.dimension == 2
    assert mesh.shape == (
        partition_x.num_segments,
        partition_y.num_segments,
    )

    # Checks for a three-dimensional mesh.
    partitions_3d = partition_x, partition_y, partition_z
    mesh = Mesh(partitions_3d)
    assert mesh.dimension == 3
    assert mesh.shape == (
        partition_x.num_segments,
        partition_y.num_segments,
        partition_z.num_segments,
    )


def _make_partition(lower_bound: float, size: float, num_segments: int):
    """Make a spatial partition."""
    return Partition.with_size(
        size=size,
        lower_bound=lower_bound,
        num_segments=num_segments,
    )


# For our tests, we first define a strategy to create spatial partitions,
# i.e., `Partitions` instances.
partitions_stg = stg.builds(
    _make_partition,
    lower_bound=stg.floats(min_value=-10, max_value=10, allow_nan=False),
    size=stg.floats(min_value=1, max_value=10, allow_nan=False),
    num_segments=stg.integers(min_value=1, max_value=128),
)


def _make_mesh(partitions: list[Partition]):
    """Make a spatial mesh."""
    return Mesh(partitions=tuple(partitions))


# Now, we define a strategy to create `Mesh` instances with a variable
# number of partitions using a `lists` hypothesis strategy with the correct
# number of partitions.
meshes_stg = stg.builds(
    _make_mesh, stg.lists(partitions_stg, min_size=1, max_size=3)
)


@given(
    partition=stg.builds(
        _make_partition,
        lower_bound=stg.integers(min_value=-10, max_value=10),
        size=stg.integers(min_value=1, max_value=10),
        num_segments=stg.integers(min_value=1, max_value=128),
    ),
    non_unit_factor=stg.floats(
        allow_nan=False, min_value=1, max_value=1e3, exclude_min=True
    ),
)
def test_partition_equality(partition: Partition, non_unit_factor: float):
    """Test a Partition's instance behavior under equality comparisons."""
    # Scale a partition by an integer value.
    partition_scaled_1 = partition.scaled(1)
    assert partition_scaled_1 == partition

    # Scale a partition by a float value.
    partition_scaled_2 = partition.scaled(1.0)
    assert partition_scaled_2 == partition

    # Check that several equal partitions collapse into a unique element
    # in a set.
    assert len({partition, partition_scaled_1, partition_scaled_2}) == 1

    # Check a simple inequality.
    assert partition.scaled(non_unit_factor) != partition


@given(mesh=meshes_stg)
def test_mesh_transformations(mesh: Mesh):
    """Check routines to transform meshes."""
    # Centered mesh.
    centered_mesh = mesh.origin_centered()
    assert centered_mesh.size == pytest.approx(mesh.size)
    assert centered_mesh.num_elements == mesh.num_elements

    # Centered mesh of unit volume.
    centered_unit_mesh = mesh.origin_centered_unit()
    assert centered_unit_mesh.size == pytest.approx(1.0)
    assert centered_unit_mesh.num_elements == mesh.num_elements

    # Scaled mesh.
    scale_factors_array = np.random.randint(
        low=1, high=10, size=centered_unit_mesh.dimension
    )
    scale_factors = tuple(scale_factors_array)
    scaled_mesh = centered_unit_mesh.scaled(factors=scale_factors)
    expected_size = float(np.prod(scale_factors_array))
    assert scaled_mesh.size == pytest.approx(expected_size)
    assert scaled_mesh.num_elements == centered_unit_mesh.num_elements

    # Translated mesh.
    translate_offsets_array = np.random.randint(
        low=1, high=10, size=centered_unit_mesh.dimension
    )
    translate_offsets = tuple(translate_offsets_array)
    translated_mesh = mesh.translated(offsets=translate_offsets)
    assert translated_mesh.size == pytest.approx(mesh.size)
    assert translated_mesh.num_elements == mesh.num_elements


@given(
    mesh=stg.builds(
        _make_mesh, stg.lists(partitions_stg, min_size=1, max_size=1)
    ),
    non_unit_factor=stg.floats(
        allow_nan=False, min_value=1, max_value=1e3, exclude_min=True
    ),
)
def test_mesh_equality(mesh: Mesh, non_unit_factor: float):
    """Test a Mesh's instance behavior under equality comparisons."""
    # Scale a mesh by an integer value.
    mesh_scaled_1 = mesh.scaled(1)
    assert mesh_scaled_1 == mesh

    # Scale a mesh by a float value.
    mesh_scaled_2 = mesh.scaled(1.0)
    assert mesh_scaled_2 == mesh

    # Check that several equal meshes collapse into a unique element in a set.
    assert len({mesh, mesh_scaled_1, mesh_scaled_2}) == 1

    # Check a simple inequality.
    assert mesh.scaled(non_unit_factor) != mesh
