import numpy as np
import pytest
from attr import evolve
from hypothesis import given
from hypothesis import strategies as stg

from gspits import Mesh, TimeMesh

valid_lower_bound_stg = stg.floats(min_value=0, max_value=2)
valid_upper_bound_stg = stg.floats(min_value=3, max_value=4)
valid_num_segments_steps_stg = stg.integers(min_value=1, max_value=512)


@given(lower_bound=stg.floats(min_value=100, max_value=200))
def test_mesh(lower_bound: float):
    """Test spatial meshes with invalid bounds"""
    with pytest.raises(ValueError):
        Mesh(lower_bound, upper_bound=0, num_segments=128)


@given(num_segments=stg.integers(min_value=-1000, max_value=0))
def test_mesh_num_steps(num_segments: int):
    """Test spatial meshes with invalid number of steps."""
    with pytest.raises(ValueError):
        Mesh(lower_bound=0, upper_bound=1, num_segments=num_segments)


@given(
    lower_bound=valid_lower_bound_stg,
    upper_bound=valid_upper_bound_stg,
    num_segments=valid_num_segments_steps_stg,
)
def test_mesh_bounds(
    lower_bound: float, upper_bound: float, num_segments: int
):
    """Test routines for creating spatial meshes."""
    mesh = Mesh(lower_bound, upper_bound, num_segments)
    mesh_as_array = mesh.array
    assert mesh_as_array.min() == mesh.lower_bound
    assert np.allclose(np.diff(mesh_as_array), mesh.step_size)
    mesh = evolve(mesh, endpoint=True)
    mesh_as_array = mesh.array
    assert np.allclose(np.diff(mesh_as_array), mesh.step_size)


# See https://stackoverflow.com/questions/19141432 for details.
valid_time_step_stg = stg.floats(min_value=np.finfo(float).eps, max_value=128)
valid_ini_time_stg = stg.floats(min_value=-128, max_value=128)


@given(
    num_steps=stg.integers(min_value=-1000, max_value=0),
)
def test_time_mesh_num_steps(num_steps: int):
    """Testing temporal meshes with an invalid number of steps."""
    with pytest.raises(ValueError):
        TimeMesh(time_step=1e-2, num_steps=num_steps)


@given(
    time_step=stg.floats(min_value=-1e2, max_value=0),
)
def test_time_mesh_time_step(time_step: float):
    """Testing temporal meshes with an invalid time step."""
    with pytest.raises(ValueError):
        TimeMesh(time_step=time_step, num_steps=128)


@given(
    time_step=valid_time_step_stg,
    num_steps=valid_num_segments_steps_stg,
    ini_time=valid_ini_time_stg,
)
def test_times_mesh(time_step: float, num_steps: int, ini_time: float):
    """Testing routines for creating temporal meshes."""
    time_mesh = TimeMesh(
        time_step=time_step, num_steps=num_steps, ini_time=ini_time
    )
    mesh_as_array = time_mesh.as_array()
    assert mesh_as_array.min() == time_mesh.ini_time
    assert np.allclose(np.diff(mesh_as_array), time_mesh.time_step)
    mesh_as_array = time_mesh.as_array(endpoint=True)
    assert np.allclose(np.diff(mesh_as_array), time_mesh.time_step)
