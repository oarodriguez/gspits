# Changelog

Versions follow [CalVer](https://calver.org).

## 2021.3.0.dev0 (Not yet released)

### Added

- Fix a term in the `gspits.one_dim.singlespecies.groundstate.BEPSSolver` class.
  This fix involves a term that allows us to find the ground-state when the
  lattice wave-vector is nonzero.
- Implement the harmonic external potential in one, two, and three dimensions:
  `gspits.potentials.harmonic.HarmonicTrap`.
- Implement the one-dimensional harmonic trap external potential:
  `gspits.potentials.onedim.HarmonicTrap`.
- Define new methods to transform a `gspits.system.Mesh` instance:
  `origin_centered_unit`, `origin_centered`, `scaled`, and `translated`.
- Define new methods to transform a `gspits.system.Partition` instance:
  `with_size` (class method), `make_origin_centered_unit` (class method),
  `origin_centered_unit`, `origin_centered`, `scaled`, and `translated`.
- Add protocol `gspits.system.ExternalPotential` to define an external
  potential.
- Add `gspits.system` module to implement protocols and classes used to define
  bosonic systems, such as classes `State` and `BlochState`.
- Add `size`, `element_size`, `num_elements`, and `shape` attributes to
  `gspits.mesh.Mesh` instances.
- Add `size` property to `gspits.mesh.Partition` class.
- Add `duration` property to `gspits.mesh.TimePartition` class.
- Add a `Mesh` class to build multidimensional meshes.
- Add `sphinx-book-theme` theme.
- Install `myst-parser` package so we can write documentation using markdown.

### Changed

- Do not make `gspits.one_dim.system.BlochState` a subclass of
  `gspits.one_dim.system.State`. Simplify the implementation of the former by
  storing the periodic-part wave function instead of the full wave function.
- Rename some classes in `gspits.mesh` module:
  - Rename class `Mesh` to `Partition`.
  - Rename class `TimeMesh` to `TimePartition`.
- Add aliases to new class names in existing code to keep compatibility.

### Deprecated

TODO.

### Removed

TODO.

### Fixed

TODO.

---

## 2021.2.0 (2021-09-25)

### Added

- Add support to calculate Bloch ground-states of periodic Hamiltonians using
  `BEPSSolver` instances.
- Add class `BlochState` to model Bloch states of periodic Hamiltonians.
- Add new 1D Hamiltonians:
  - `OLHTHamiltonian`. Represents a Bose gas within an optical lattice
    superimposed with a harmonic trap.
  - `MRHamiltonian`. Represents a Bose gas within a multi-rod potential
    composed of multiple consecutive wells and barriers.
  - Implement `plane_wave_state` routine to build normalized plane wave states.
- Add task `jupyter-lab`.

### Changed

- Move the routines to analyze one-dimensional, single-species Hamiltonians to
  the `gspits.one_dim.singlespecies` subpackage.
- Retrieve the last state of a `BEPSSolver` instance instead of the Hamiltonian
  ground-state. This change is advantageous since we can retrieve the
  ground-state from the last `BEPSSolverState`, in addition to other essential
  quantities as the energy and chemical potential.
- `BEPSSolver` instances must use the initial state `mesh` attribute. With this
  change, the solver only has one obvious mesh to calculate the ground state.
- Update dependencies and `poetry.lock` file.

### Removed

- Remove `gspits.one_dim.system.SupportsExternalPotential` class.
- Remove `gspits.one_dim.system.external_potential` function.

### Fixed

- Fix the code and equations for the imaginary time evolution. This change
  ensures that the system properties are size-independent if the system density
  remains constant.
- Fix the normalization procedure to construct a Bloch plane wave using the
  `BlochState.plane_wave` class method.

---

## 2021.1.0 (2021-09-01)

### Added

- Add support to identify code quality issues using the `pre-commit` library.
- The Project now uses CalVer to define versions and releases.
- Add routines for creating spatial and temporal meshes/grids.
- Add class `BEPSSolver` to find a 1D Hamiltonian ground-state using the
  _Backward Euler Pseudo-Spectral_ (BEPS) method.
- Define a classes and protocols to define one-dimensional Hamiltonians.
  - The class `HTHamiltonian` represents a 1D Bose gas within a harmonic trap
    external potential.

---

## 0.1.0 (2021-04-29)

Project initialization.
