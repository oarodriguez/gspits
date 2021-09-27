"""Routines to generate spatial and temporal partitions."""

import numpy as np
from attr import dataclass, field

__all__ = [
    "Mesh",
    "MeshArrays",
    "MeshPartitions",
    "Partition",
    "TimePartition",
]


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
    def size(self) -> float:
        """Give the partition length."""
        return self.upper_bound - self.lower_bound

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
    def duration(self) -> float:
        """Give the partition duration."""
        return self.finish_time - self.ini_time

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


# Mesh attributes types.
# See bug https://github.com/python/mypy/issues/9980.
MeshPartitions = tuple[Partition, ...]  # type: ignore
MeshArrays = tuple[np.ndarray, ...]  # type: ignore

# Error messages.
MESH_DIMENSION_ERROR = (
    "The mesh maximum allowed dimension is three. Therefore, you should "
    "supply the 'partitions' argument a tuple with at most three elements."
)


@dataclass(frozen=True)
class Mesh:
    """Construct a spatial mesh from several partitions.

    :param tuple[Partition, ...] partitions:
        A tuple of ``Partition`` instances for each dimension of the
        mesh. The tuple must have at most three elements.
    """

    # Partitions that form the mesh.
    partitions: MeshPartitions

    # Mesh sparse arrays.
    _arrays: MeshArrays = field(init=False, default=None, repr=False)

    def __attrs_post_init__(self):
        """Post-initialization tasks."""
        if self.dimension > 3:
            raise ValueError(MESH_DIMENSION_ERROR)
        partition_arrays = [partition.array for partition in self.partitions]
        arrays = np.meshgrid(*partition_arrays, indexing="ij", sparse=True)
        object.__setattr__(self, "_arrays", tuple(arrays))

    @property
    def dimension(self) -> int:
        """Give the mesh dimension.

        It is one for a 1D mesh, two for a 2D mesh, and three for a
        3D mesh.

        :rtype: int
        """
        return len(self.partitions)

    @property
    def arrays(self) -> MeshArrays:
        """Return the NumPy arrays representing the mesh.

        **NOTE**: The returned arrays are sparse.

        :rtype: tuple[numpy.ndarray, ...]
        """
        return self._arrays

    @property
    def shape(self):
        """Shape of the mesh arrays after being broadcast.

        :rtype: tuple[int, ...]
        """
        array_shapes = [array.shape for array in self.arrays]
        return np.broadcast_shapes(*array_shapes)
