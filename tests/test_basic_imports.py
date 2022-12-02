"""
Test for imports where the packges will be used in molSimplify.
"""

import sys


def test_molsimplify_imported():
    '''
    Sample test, will always pass so long as import statement worked
    '''
    assert "molSimplify" in sys.modules


def test_psi4_import():
    '''
    Test whether psi4 can be imported
    '''
    try:
        import psi4
        assert "psi4" in sys.modules
    except ImportError:
        assert 0


def test_tf_import():
    '''
    Test whether tensorflow can be imported
    '''
    try:
        import tensorflow
        assert "tensorflow" in sys.modules
    except ImportError:
        assert 0


def test_keras_import():
    '''
    Test whether keras can be imported
    '''
    try:
        import keras
        assert "keras" in sys.modules
    except ImportError:
        assert 0


def test_h5py_import():
    '''
    Test whether h5py can be imported
    '''
    try:
        import h5py
        assert "h5py" in sys.modules
    except ImportError:
        assert 0


def test_openbabel_import():
    '''
    Test whether openbabel can be imported
    '''
    try:
        import openbabel
        assert "openbabel" in sys.modules
    except ImportError:
        assert 0
