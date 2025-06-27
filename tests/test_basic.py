"""
Basic tests for TubeAtlas application.

This file provides initial test coverage to ensure the CI/CD pipeline
works correctly. As the application grows, more comprehensive tests
should be added.
"""

import sys
from pathlib import Path

import pytest


def test_python_version():
    """Test that we're running on Python 3.12+."""
    assert sys.version_info >= (3, 12), f"Python 3.12+ required, got {sys.version}"


def test_project_structure():
    """Test that basic project structure exists."""
    project_root = Path(__file__).parent.parent

    # Check for key directories
    assert (
        project_root / "src" / "tubeatlas"
    ).exists(), "src/tubeatlas directory missing"
    assert (project_root / "pyproject.toml").exists(), "pyproject.toml missing"
    assert (project_root / "Dockerfile").exists(), "Dockerfile missing"
    assert (project_root / "docker-compose.yml").exists(), "docker-compose.yml missing"


def test_imports():
    """Test that we can import the main package."""
    try:
        import tubeatlas

        assert tubeatlas is not None
    except ImportError as e:
        pytest.fail(f"Failed to import tubeatlas package: {e}")


class TestBasicFunctionality:
    """Test suite for basic application functionality."""

    def test_placeholder(self):
        """Placeholder test that always passes."""
        assert True, "This is a placeholder test"

    def test_math_operations(self):
        """Test basic math operations to ensure pytest works."""
        assert 2 + 2 == 4
        assert 5 * 3 == 15
        assert 10 / 2 == 5.0

    def test_string_operations(self):
        """Test string operations."""
        test_string = "TubeAtlas"
        assert test_string.lower() == "tubeatlas"
        assert test_string.upper() == "TUBEATLAS"
        assert len(test_string) == 9


if __name__ == "__main__":
    pytest.main([__file__])
