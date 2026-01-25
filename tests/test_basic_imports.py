"""
Test for imports where the packges will be used in molSimplify.
"""

import sys


def test_molsimplify_imported():
    '''
    Sample test, will always pass so long as import statement worked
    '''
    assert "molSimplify" in sys.modules


def test_tf_import():
    '''
    Test whether tensorflow can be imported
    '''
    try:
        import tensorflow  # noqa: F401
        assert "tensorflow" in sys.modules
    except ImportError:
        assert 0


def test_openbabel_import():
    '''
    Test whether openbabel can be imported
    '''
    try:
        try:
            from openbabel import openbabel  # version 3 style import
        except ImportError:  # fallback to version 2
            import openbabel  # noqa: F401
        assert "openbabel" in sys.modules
    except ImportError:
        assert 0
