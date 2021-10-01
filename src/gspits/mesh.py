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
    """Construct a spatial partition.

    :param float lower_bound:
        The partition lower bound.
    :param float upper_bound:
        The partition upper bound.
    :param int num_segments:
        The partition lower bound.
    :param bool endpoint:
        Indicate whether to consider the upper bound as part of the
        partition. By default, the upper bound is excluded
        (``endpoint = False``).

    :raises ValueError:
        If ``upper_bound`` is less than ``lower_bound``.
    :raises ValueError:
        If ``num_segments`` is a negative integer or zero.
    :raises ValueError:
        If any of ``lower_bound`` or ``upper_bound`` is ``nan``.
    """

    # TODO: Make Partition instances iterable.

    # Lower bound.
    lower_bound: float

    # Upper bound.
    upper_bound: float

    # Mesh number of segments.
    num_segments: int

    # Indicate whether to consider the upper bound as part of the partition.
    # If `False`, the upper bound is excluded.
    endpoint: bool = False

    def __attrs_post_init__(self) -> None:
        """Post-initialization procedure."""
        if np.isnan(self.lower_bound):
            raise ValueError("'nan' is not a valid value for 'lower_bound.")
        if np.isnan(self.upper_bound):
            raise ValueError("'nan' is not a valid value for 'upper_bound'.")
        if not self.upper_bound > self.lower_bound:
            raise ValueError(
                "'upper_bound' must be greater than 'lower_bound'."
            )
        if not self.num_segments >= 1:
            raise ValueError(
                "'num_segments' must be a positive, non-zero integer."
            )

    @classmethod
    def with_size(
        cls: type["Partition"],
        size: float,
        lower_bound: float,
        num_segments: int,
        endpoint: bool = False,
    ) -> "Partition":
        """Create a partition with a given lower bound and size.

        This method is a convenient alternative to construct a ``Partition``
        instance when we want to specify its size and lower bound location.

        :param float size:
            The partition size (length).
        :param float lower_bound:
            Location of the new partition lower bound.
        :param int num_segments:
            The partition number of segments.
        :param bool endpoint:
            Whether or not to include the endpoint in the partition. It is
            ``False`` by default.
        :rtype: Partition
        """
        return cls(
            lower_bound=lower_bound,
            upper_bound=lower_bound + size,
            num_segments=num_segments,
            endpoint=endpoint,
        )

    @classmethod
    def make_origin_centered_unit(
        cls: type["Partition"], num_segments: int, endpoint: bool = False
    ) -> "Partition":
        """Get a partition of unit length centered at the origin.

        :param int num_segments:
            The partition number of segments.
        :param bool endpoint:
            Whether or not to include the endpoint in the partition. It is
            ``False`` by default.
        :rtype: Partition
        """
        return cls(
            lower_bound=-0.5,
            upper_bound=0.5,
            num_segments=num_segments,
            endpoint=endpoint,
        )

    def origin_centered_unit(self) -> "Partition":
        """Get a similar partition of unit length centered at the origin.

        The new ``Partition`` instance shares the same number of segments
        and the ``endpoint`` attribute as the current partition. However,
        its lower and upper bounds are different.

        :rtype: Partition
        """
        return self.make_origin_centered_unit(
            num_segments=self.num_segments,
            endpoint=self.endpoint,
        )

    def origin_centered(self) -> "Partition":
        """Get a similar partition centered at the origin.

        The new ``Partition`` instance shares the same number of segments
        and the ``endpoint`` attribute as the current partition and has the
        same size. However, its lower and upper bounds change.

        :rtype: Partition
        """
        partition_size = self.size
        return Partition(
            lower_bound=-partition_size / 2,
            upper_bound=partition_size / 2,
            num_segments=self.num_segments,
            endpoint=self.endpoint,
        )

    def scaled(self, factor: float) -> "Partition":
        """Make a similar partition under a scaling transformation.

        The new ``Partition`` instance shares the same number of segments
        and the ``endpoint`` attribute as the current partition.

        :param float factor:
            A scale factor. The upper and lower bounds of the new partition
            will be proportional to the bounds of the current one,
            being ``factor`` the proportionality coefficient. Accordingly, the
            size of the new partition will be scaled by the same factor too.
        :rtype: Partition
        """
        return Partition(
            lower_bound=self.lower_bound * factor,
            upper_bound=self.upper_bound * factor,
            num_segments=self.num_segments,
            endpoint=self.endpoint,
        )

    def translated(self, offset: float) -> "Partition":
        """Displace this partition by a fixed number.

        The new ``Partition`` instance shares the same number of segments
        and the ``endpoint`` attribute as the current partition and has the
        same size. However, its lower and upper bounds change due to the
        translation.

        :param float offset:
            The lower and upper bounds of the new partition will be
            displaced by the amount set by ``offset``.
        :rtype: Partition
        """
        return Partition(
            lower_bound=self.lower_bound + offset,
            upper_bound=self.upper_bound + offset,
            num_segments=self.num_segments,
            endpoint=self.endpoint,
        )

    @property
    def size(self) -> float:
        """Give the partition length.

        :rtype: float
        """
        return self.upper_bound - self.lower_bound

    @property
    def step_size(self) -> float:
        """Partition step size.

        :rtype: float
        """
        return (self.upper_bound - self.lower_bound) / self.num_segments

    @property
    def midpoint(self) -> float:
        """Return the partition midpoint.

        :rtype: float
        """
        return (self.lower_bound + self.upper_bound) / 2

    @property
    def array(self) -> np.ndarray:
        """Return an array with the partition points.

        :rtype: numpy.ndarray
        """
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
    """Construct a time partition.

    :param float time_step:
        The partition time step.
    :param int num_steps:
        The partition number of steps.
    :param float ini_time:
        The partition initial time. By default, it is zero.
    :param bool endpoint:
        Indicate whether to consider the upper bound as part of the
        partition. By default, the upper bound is excluded
        (``endpoint = False``).

    :raises ValueError:
        If ``time_step`` is negative or zero.
    :raises ValueError:
        If ``num_steps`` is a negative integer or zero.
    :raises ValueError:
        If any of ``time_step`` or ``ini_time`` is ``nan``.
    """

    # TODO: Make TimePartition instances iterable.

    # Partition time step.
    time_step: float

    # Partition number of steps.
    num_steps: int

    # Partition initial time.
    ini_time: float = 0

    # Indicate whether to include the end time as part of the mesh.
    # If `False`, the end time is excluded.
    endpoint: bool = True

    def __attrs_post_init__(self) -> None:
        """Post-initialization procedure."""
        if np.isnan(self.time_step):
            raise ValueError("'nan' is not a valid value for 'time_step.")
        if np.isnan(self.ini_time):
            raise ValueError("'nan' is not a valid value for 'ini_time'.")
        if not self.time_step > 0:
            raise ValueError(
                "'time_step' must be a positive, non-zero number."
            )
        if not self.num_steps >= 1:
            raise ValueError(
                "'num_steps' must be a positive, non-zero integer."
            )

    @property
    def finish_time(self) -> float:
        """Partition finish time.

        :rtype: float
        """
        return self.ini_time + self.num_steps * self.time_step

    @property
    def duration(self) -> float:
        """Give the partition duration.

        :rtype: float
        """
        return self.finish_time - self.ini_time

    @property
    def array(self) -> np.ndarray:
        """Return an array with the partition points.

        :rtype: numpy.ndarray
        """
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

# Variable types for arguments used in transformation methods.
MeshScalingFactors = tuple[float, ...]  # type: ignore
MeshTranslationOffsets = tuple[float, ...]  # type: ignore

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

    def __attrs_post_init__(self) -> None:
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
    def size(self) -> float:
        """Get the mesh size.

        For a 1D mesh, it is the length of its only partition. For a
        2D mesh, it is the area of the region delimited by its partitions.
        For a 3D mesh, it is the volume.

        :rtype: float
        """
        size = 1.0
        for partition in self.partitions:
            size *= partition.size
        return size

    @property
    def element_size(self) -> float:
        """Size of a mesh partition element.

        :rtype: float
        """
        return float(
            np.prod([partition.step_size for partition in self.partitions])
        )

    @property
    def num_elements(self) -> int:
        """Get the number of elements that compose the mesh.

        :rtype: int
        """
        return int(
            np.prod([partition.num_segments for partition in self.partitions])
        )

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

    def origin_centered_unit(self) -> "Mesh":
        """Get a new mesh of unit volume whose center lies at the origin.

        This method applies a similar transformation to its internal
        partitions to achieve the intended result.

        :rtype: Mesh
        """
        centered_partitions = []
        for partition in self.partitions:
            centered_partition = partition.origin_centered_unit()
            centered_partitions.append(centered_partition)
        return Mesh(MeshPartitions(centered_partitions))

    def origin_centered(self) -> "Mesh":
        """Get a new mesh whose center lies at the origin.

        This method applies a similar transformation to its internal
        partitions to achieve the intended result.

        :rtype: Mesh
        """
        centered_partitions = []
        for partition in self.partitions:
            centered_partition = partition.origin_centered()
            centered_partitions.append(centered_partition)
        return Mesh(MeshPartitions(centered_partitions))

    def scaled(self, factors: MeshScalingFactors) -> "Mesh":
        """Get a new mesh by applying a scaling transformation.

        This method applies a similar transformation to its internal
        partitions to achieve the intended result.

        :param tuple[float, ...] factors:
            A tuple with the same number of elements as this mesh dimension.
        :rtype: Mesh
        """
        scaled_partitions = []
        for partition, factor in zip(self.partitions, factors):
            scaled_partition = partition.scaled(factor=factor)
            scaled_partitions.append(scaled_partition)
        return Mesh(MeshPartitions(scaled_partitions))

    def translated(self, offsets: MeshTranslationOffsets) -> "Mesh":
        """Get a new mesh by applying a translation.

        This method applies a similar transformation to its internal
        partitions to achieve the intended result.

        :param tuple[float, ...] offsets:
            A tuple with the same number of elements as this mesh dimension.
        :rtype: Mesh
        """
        scaled_partitions = []
        for partition, offset in zip(self.partitions, offsets):
            scaled_partition = partition.translated(offset=offset)
            scaled_partitions.append(scaled_partition)
        return Mesh(MeshPartitions(scaled_partitions))
