# Changelog

Versions follow [CalVer](https://calver.org).

## 2021.2.0.dev0 (Not yet released)

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

- Retrieve the last state of a `BEPSSolver` instance instead of the Hamiltonian
  ground-state. This change is advantageous since we can retrieve the
  ground-state from the last `BEPSSolverState`, in addition to other essential
  quantities as the energy and chemical potential.
- `BESPSolver` instances must use the initial state `mesh` attribute. With this
  change, the solver only has one obvious mesh to calculate the ground state.
- Update dependencies and `poetry.lock` file.

### Deprecated

TODO.

### Removed

TODO.

### Fixed

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
