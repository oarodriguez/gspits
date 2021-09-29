"""gspits - The Gross-Pitaevskii Equation Toolset.

A collection of routines to analyze the ground-state properties and
dynamics of a Bose gas using the Gross-Pitaevskii equation.

Copyright © 2021, Omar Abel Rodríguez-López.
"""

# See https://github.com/python-poetry/poetry/pull/2366#issuecomment-652418094
import importlib.metadata as importlib_metadata

from .mesh import Mesh, Partition, TimePartition
from .system import BlochState, State

metadata = importlib_metadata.metadata("gspits")

# Export package information.
__version__ = metadata["version"]
__author__ = metadata["author"]
__description__ = metadata["description"]
__license__ = metadata["license"]

__all__ = [
    "BlochState",
    "Mesh",
    "Partition",
    "State",
    "TimePartition",
    "__author__",
    "__description__",
    "__license__",
    "__version__",
    "metadata",
]
