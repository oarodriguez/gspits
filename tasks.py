"""
Collection of development tasks.

Usage:
    python -m tasks TASK-NAME
"""
import os
import shutil
from enum import Enum, unique
from pathlib import Path
from subprocess import run

import click
from click.exceptions import Exit

PROJECT_DIR = Path(__file__).parent
SRC_DIR = PROJECT_DIR / "src"
TESTS_DIR = PROJECT_DIR / "tests"
DOCS_DIR = PROJECT_DIR / "docs"
DOCS_SOURCE_DIR = DOCS_DIR / "source"
DOCS_BUILD_DIR = DOCS_DIR / "build"
DATA_DIR = PROJECT_DIR / "data"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
TASKS_FILE = PROJECT_DIR / "tasks.py"
README_FILE = PROJECT_DIR / "README.md"
CHANGELOG_FILE = PROJECT_DIR / "CHANGELOG.md"

# NOTE: See https://stackoverflow.com/a/32799942 to understand why we use
#   shutil.which().
PYTHON_CMD = "python"
POETRY_CMD = shutil.which("poetry")
JUPYTER_CMD = shutil.which("jupyter")
PIP_CMD = "pip"
ISORT_CMD = "isort"
BLACK_CMD = "black"
MDFORMAT_CMD = shutil.which("mdformat")
PYDOCSTYLE_CMD = "pydocstyle"
FLAKE8_CMD = "flake8"
MYPY_CMD = "mypy"
PYTEST_CMD = "pytest"
SPHINX_BUILD_CMD = "sphinx-build"

# Coverage report XML file.
COVERAGE_XML = "coverage.xml"

app = click.Group("tasks")


def _get_package_info():
    """Return the package name and version from pyproject.toml."""
    buffer = run(
        [POETRY_CMD, "version"], capture_output=True, encoding="utf-8"
    )
    if rc := buffer.returncode:
        raise Exit(rc)
    buffer_contents = buffer.stdout
    name: str
    version_: str
    # In principle, the package name should have no spaces.
    name, version_ = buffer_contents.split(" ")
    return name.strip(), version_.strip()


def _get_installed_package_info():
    """Return the name and version of the installed project package."""
    import gspits

    return gspits.metadata["name"], gspits.__version__


@app.command()
def install():
    """Install the current project package.

    Do nothing if the package is already installed.
    """
    try:
        name, version_ = _get_installed_package_info()
    except ModuleNotFoundError:
        install_args = [POETRY_CMD, "install"]
        if rc := run(install_args).returncode:
            raise Exit(rc)
        print("Module installed successfully.")
        verify_message = (
            "Check installed version through "
            """"python -m tasks version" command."""
        )
        print(verify_message)
    else:
        raise click.ClickException(f"{name} {version_} is already installed.")


@app.command()
def uninstall():
    """Uninstall the current project package.

    Returns an error if the project package is not installed.
    """
    try:
        name, version_ = _get_installed_package_info()
        pip_args = [PIP_CMD, "uninstall", "--yes", name]
        if rc := run(pip_args).returncode:
            raise Exit(rc)
        print(f"Package '{name} {version_}' uninstalled successfully.")
    except ModuleNotFoundError:
        name, version_ = _get_package_info()
        raise click.ClickException(
            f"The package '{name} {version_}' has not been installed."
        )


@app.command()
def upgrade():
    """Upgrade the project package installation."""
    name, new_version = _get_package_info()
    try:
        name, old_version = _get_installed_package_info()
        if old_version == new_version:
            print("The installed project package is the latest.")
            raise Exit(1)
        pip_args = [PIP_CMD, "uninstall", "--yes", name]
        if rc := run(pip_args).returncode:
            raise Exit(rc)
        print(f"Package '{name} {old_version}' uninstalled successfully.")
    except ModuleNotFoundError:
        raise click.ClickException(
            f"The package '{name} {new_version}' has not been installed."
        )

    install_args = [POETRY_CMD, "install"]
    if rc := run(install_args).returncode:
        raise Exit(rc)
    print("Package upgraded successfully.")
    verify_message = (
        "Check installed version through "
        """"python -m tasks version" command."""
    )
    print(verify_message)


@app.command()
def version():
    """Show the installed project version."""
    try:
        import gspits

        print(f"{gspits.metadata['name']} {gspits.__version__}")
    except ModuleNotFoundError:
        raise click.ClickException(
            "The package 'gspits' has not been installed."
        )


@app.command()
def tests():
    """Run test suite."""
    pytest_args = [
        PYTEST_CMD,
        "--cov",
        "--cov-report",
        "term-missing",
        "--cov-report",
        f"xml:./{COVERAGE_XML}",
    ]
    raise Exit(run(pytest_args).returncode)


@app.command(name="format")
def format_():
    """Execute formatting tasks.

    Format files using `black` together with `isort` to sort imports.
    """
    format_args = [
        BLACK_CMD,
        str(TASKS_FILE),
        str(SRC_DIR),
        str(TESTS_DIR),
        str(DOCS_DIR),
        str(NOTEBOOKS_DIR),
    ]
    if rc := run(format_args).returncode:
        raise Exit(rc)

    isort_args = [
        ISORT_CMD,
        str(TASKS_FILE),
        str(SRC_DIR),
        str(TESTS_DIR),
        str(DOCS_DIR),
        str(NOTEBOOKS_DIR),
    ]
    if rc := run(isort_args).returncode:
        raise Exit(rc)


@app.command()
def format_docs():
    """Execute formatting tasks for markdown files.

    Format markdown files using the `mdformat` library.
    """
    format_args = [
        MDFORMAT_CMD,
        str(SRC_DIR),
        str(DOCS_DIR),
        str(NOTEBOOKS_DIR),
        str(README_FILE),
        str(CHANGELOG_FILE),
    ]
    raise Exit(run(format_args).returncode)


@app.command()
def typecheck():
    """Execute typechecking tasks.

    Execute `mypy` for static type checking.
    """
    mypy_args = [
        MYPY_CMD,
        str(TASKS_FILE),
        str(SRC_DIR),
        str(TESTS_DIR),
        # str(DOCS_DIR),
        # str(NOTEBOOKS_DIR),
    ]
    raise Exit(run(mypy_args).returncode)


@app.command()
def lint():
    """Execute linting tasks.

    Check code style issues using `flake8` and `pydocstyle` to
    check docstrings.
    """
    pydocstyle_args = [
        PYDOCSTYLE_CMD,
        str(TASKS_FILE),
        str(SRC_DIR),
        str(TESTS_DIR),
        str(DOCS_DIR),
        str(NOTEBOOKS_DIR),
    ]
    if rc := run(pydocstyle_args).returncode:
        raise Exit(rc)

    flake8_args = [
        FLAKE8_CMD,
        str(TASKS_FILE),
        str(SRC_DIR),
        str(TESTS_DIR),
        str(DOCS_DIR),
        str(NOTEBOOKS_DIR),
        "--statistics",
    ]
    if rc := run(flake8_args).returncode:
        raise Exit(rc)


@unique
class DocFormat(str, Enum):
    """Document Formats."""

    HTML = "html"


# Set HTML as the default document format.
default_doc_format = DocFormat.HTML.name

# List of allowed document formats.
doc_formats = list(DocFormat.__members__.keys())


@app.command()
@click.option(
    "--doc-format",
    type=click.Choice(doc_formats),
    default=default_doc_format,
    help=f"Generated documentation format. Defaults to {default_doc_format}.",
)
def build_docs(doc_format: str):
    """Build the documentation."""
    build_docs_args = [
        SPHINX_BUILD_CMD,
        str(DOCS_SOURCE_DIR),
        str(DOCS_BUILD_DIR),
    ]
    doc_format_ = DocFormat[doc_format]
    build_docs_args.extend(["-b", doc_format_])
    raise Exit(run(build_docs_args).returncode)


@unique
class CleaningTask(str, Enum):
    """Cleaning tasks."""

    DOCS = "docs"


# Set DOCS as the default cleaning task.
default_cleaning_task = CleaningTask.DOCS.name

# List of allowed cleaning tasks.
cleaning_tasks = list(CleaningTask.__members__.keys())


@app.command()
@click.option(
    "--task",
    type=click.Choice(cleaning_tasks),
    default=default_cleaning_task,
    help=f"Cleaning task to perform. Defaults to {default_cleaning_task}.",
)
def clean(task: str):
    """Clean project resources."""
    try:
        task_ = None if task is None else CleaningTask[task]
        if task_ is None or task_ is CleaningTask.DOCS:
            shutil.rmtree(DOCS_BUILD_DIR)
    except Exception:
        raise Exit(1)


@app.command()
def jupyter_lab():
    """Execute a jupyter lab server instance in the current directory."""
    environ = {}
    environ.update(os.environ)
    # Set project environment variables.
    environ["DATA_DIR"] = str(DATA_DIR)
    environ["NOTEBOOKS_DIR"] = str(NOTEBOOKS_DIR)
    environ["SCRIPTS_DIR"] = str(SCRIPTS_DIR)
    if JUPYTER_CMD is None:
        raise click.ClickException("'jupyter' library is not installed")
    jupyter_lab_args = [
        JUPYTER_CMD,
        "lab",
        "--no-browser",
    ]
    raise Exit(run(jupyter_lab_args, env=environ).returncode)


if __name__ == "__main__":
    app()
