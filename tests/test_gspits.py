"""Verify the library top-level functionality."""
import gspits


def test_version():
    """Verify we have updated the package version."""
    assert gspits.__version__ == "2022.2.0.dev0"
