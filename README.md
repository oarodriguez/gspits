# gspits - The Gross-Pitaevskii Equation Toolset

A collection of routines to analyze the ground-state properties and dynamics of a Bose gas using the
Gross-Pitaevskii equation.

## Installation

The project dependencies are managed by [poetry][poetry], so poetry should be installed in the
system. In the project root directory,  execute

```shell
poetry install
```

Poetry will take care of the installation process. Afterward, the project packages and command-line
interface tools should be available in the current shell. It is recommended to create a separate
virtual environment for this project. If you use [conda][conda], it is enough to make a minimal
environment with Python 3.9 or greater, for instance,

```shell
conda create -n gspitsenv python=3.9
```

Naturally, other virtual environment managers can be used.

## Authors

Omar Abel Rodríguez López, [https://github.com/oarodriguez][gh-oarodriguez]

[comment]: <> (---)

[gh-oarodriguez]: https://github.com/oarodriguez
[poetry]: https://python-poetry.org
[conda]: https://docs.conda.io/en/latest/
