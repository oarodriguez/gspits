"""gspits - The Gross-Pitaevskii Equation Toolset.

A collection of routines to analyze the ground-state properties and
dynamics of a Bose gas using the Gross-Pitaevskii equation.

Copyright © 2021, Omar Abel Rodríguez-López.
"""

import importlib.metadata as importlib_metadata

from .mesh import Mesh, TimeMesh

metadata = importlib_metadata.metadata("gspits")

# Export package information.
__version__ = metadata["version"]

__all__ = ["Mesh", "TimeMesh", "__version__", "metadata"]
