"""Routines to generate spatial and temporal partitions."""

import numpy as np
from attr import dataclass

__all__ = ["Partition", "TimePartition"]


@dataclass(frozen=True)
class Partition:
    """Construct a spatial partition."""

    # Lower bound.
    lower_bound: float

    # Upper bound.
    upper_bound: float

    # Mesh number of segments.
    num_segments: int

    # Indicate whether to consider the upper bound as part of the partition.
    # If `False`, the upper bound is excluded.
    endpoint: bool = False

    def __attrs_post_init__(self):
        """Post-initialization procedure."""
        if not self.upper_bound > self.lower_bound:
            raise ValueError
        if not self.num_segments >= 1:
            raise ValueError

    @property
    def step_size(self):
        """Partition step size."""
        return (self.upper_bound - self.lower_bound) / self.num_segments

    @property
    def array(self) -> np.ndarray:
        """Return an array with the partition points."""
        endpoint = self.endpoint
        num_segments = self.num_segments + (1 if endpoint else 0)
        return np.linspace(
            self.lower_bound,
            self.upper_bound,
            num=num_segments,
            endpoint=endpoint,
        )


@dataclass(frozen=True)
class TimePartition:
    """Construct a time partition."""

    # Partition time step.
    time_step: float

    # Partition number of steps.
    num_steps: int

    # Partition initial time.
    ini_time: float = 0

    # Indicate whether to include the end time as part of the mesh.
    # If `False`, the end time is excluded.
    endpoint: bool = True

    def __attrs_post_init__(self):
        """Post-initialization procedure."""
        if not self.num_steps >= 1:
            raise ValueError
        if not self.time_step > 0:
            raise ValueError

    @property
    def finish_time(self):
        """Partition finish time."""
        return self.ini_time + self.num_steps * self.time_step

    @property
    def array(self) -> np.ndarray:
        """Return an array with the partition points."""
        endpoint = self.endpoint
        num_steps = self.num_steps + (1 if endpoint else 0)
        return np.linspace(
            self.ini_time,
            self.finish_time,
            num=num_steps,
            endpoint=endpoint,
        )
