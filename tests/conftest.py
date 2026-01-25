"""Pytest configuration for molSimplify tests.

Ensures the tests directory is on sys.path so that tests in subdirectories
(tutorials/, examples/) can import helperFuncs.
"""
import sys
from pathlib import Path

# Allow tests in subdirs (e.g. tutorials/, examples/) to import helperFuncs
_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))
