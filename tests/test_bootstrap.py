"""Bootstrap sanity: package imports and Python version guard."""

import sys


def test_package_imports():
    import tinycomplete

    assert tinycomplete is not None


def test_python_version():
    assert sys.version_info >= (3, 11)
