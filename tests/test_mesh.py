"""Verify the routines in ``gspits.mesh`` module."""

import numpy as np
import pytest
from attr import evolve
from hypothesis import given
from hypothesis import strategies as stg

from gspits import Partition, TimePartition

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
    partition = Partition(lower_bound, upper_bound, num_segments)
    partition_array = partition.array
    assert partition_array.min() == partition.lower_bound
    assert np.allclose(np.diff(partition_array), partition.step_size)
    partition = evolve(partition, endpoint=True)
    partition_array = partition.array
    assert np.allclose(np.diff(partition_array), partition.step_size)


# See https://stackoverflow.com/questions/19141432 for details.
valid_time_step_stg = stg.floats(min_value=np.finfo(float).eps, max_value=128)
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
    time_mesh = TimePartition(
        time_step=time_step, num_steps=num_steps, ini_time=ini_time
    )
    partition_array = time_mesh.array
    assert partition_array.min() == time_mesh.ini_time
    assert np.allclose(np.diff(partition_array), time_mesh.time_step)
    time_mesh = evolve(time_mesh, endpoint=True)
    partition_array = time_mesh.array
    assert np.allclose(np.diff(partition_array), time_mesh.time_step)
