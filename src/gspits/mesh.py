"""Routines to generate spatial and temporal meshes."""

import numpy as np
from attr import dataclass

__all__ = ["Mesh", "TimeMesh"]


@dataclass(frozen=True)
class Mesh:
    """Spatial mesh specification."""

    # Lower bound.
    lower_bound: float

    # Upper bound.
    upper_bound: float

    # Mesh number of segments.
    num_segments: int

    def __attrs_post_init__(self):
        """Post-initialization procedure."""
        if not self.upper_bound > self.lower_bound:
            raise ValueError
        if not self.num_segments >= 1:
            raise ValueError

    @property
    def step_size(self):
        """Mesh step size."""
        return (self.upper_bound - self.lower_bound) / self.num_segments

    def as_array(self, endpoint: bool = False):
        """Return an array with the mesh points.

        The upper bound can be excluded from the array if needed.

        :param endpoint: Indicate whether to include the upper bound in
            the array. If `False`, the upper bound is excluded. The default
            value is `False`.
        :return: The NumPy array representing the mesh.
        """
        num_segments = self.num_segments + (1 if endpoint else 0)
        return np.linspace(
            self.lower_bound,
            self.upper_bound,
            num=num_segments,
            endpoint=endpoint,
        )


@dataclass(frozen=True)
class TimeMesh:
    """Mesh specification."""

    # Mesh time step.
    time_step: float

    # Mesh number of steps.
    num_steps: int

    # Mesh initial time.
    ini_time: float = 0

    def __attrs_post_init__(self):
        """Post-initialization procedure."""
        if not self.num_steps >= 1:
            raise ValueError
        if not self.time_step > 0:
            raise ValueError

    @property
    def finish_time(self):
        """Mesh step size."""
        return self.ini_time + self.num_steps * self.time_step

    def as_array(self, endpoint: bool = False):
        """Return an array with the mesh points.

        The end time can be excluded from the array if needed.

        :param endpoint: Indicate whether to include the end time in
            the array. If `False`, the end time is excluded. The default
            value is `False`.
        :return: The NumPy array representing the mesh.
        """
        num_steps = self.num_steps + (1 if endpoint else 0)
        return np.linspace(
            self.ini_time, self.finish_time, num=num_steps, endpoint=endpoint
        )
