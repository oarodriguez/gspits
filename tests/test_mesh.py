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

    with pytest.raises(ValueError):
        # A mesh can have three dimensions at most.
        Mesh((partition_x, partition_y, partition_z, partition_x))

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

    with pytest.raises(ValueError):
        # A mesh can have three dimensions at most.
        Mesh((partition_x, partition_y, partition_z, partition_x))

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
