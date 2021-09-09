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

    # Indicate whether to consider the upper bound as part of the mesh.
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
        """Mesh step size."""
        return (self.upper_bound - self.lower_bound) / self.num_segments

    @property
    def array(self) -> np.ndarray:
        """Return an array with the mesh points."""
        endpoint = self.endpoint
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
        """Mesh step size."""
        return self.ini_time + self.num_steps * self.time_step

    @property
    def array(self) -> np.ndarray:
        """Return an array with the mesh points."""
        endpoint = self.endpoint
        num_steps = self.num_steps + (1 if endpoint else 0)
        return np.linspace(
            self.ini_time,
            self.finish_time,
            num=num_steps,
            endpoint=endpoint,
        )
