"""SIAB package."""

try:
    from ._version import version as __version__
except ImportError:
    # Fallback for development installation without setuptools_scm
    __version__ = "3.0.0-dev"

__all__ = ["__version__"]