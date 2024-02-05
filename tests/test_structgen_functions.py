import numpy as np
from argparse import Namespace
from molSimplify.Classes.atom3D import atom3D
from molSimplify.Classes.rundiag import run_diag
from molSimplify.Scripts.io import loaddata, lig_load, getlicores
from molSimplify.Scripts.structgen import (smartreorderligs,
                                           get_MLdist_database,
                                           get_MLdist,
                                           init_ANN)


def test_smartreorderligs():
    """Expected behavior: First order by denticity, then by number of atom"""
    indices = smartreorderligs(['water']*6, [1]*6)
    assert indices == [0, 1, 2, 3, 4, 5]

    indices = smartreorderligs(['water', 'ammonia', 'water', 'water',
                                'ammonia', 'water'], [1]*6)
    assert indices == [0, 2, 3, 5, 1, 4]

    indices = smartreorderligs(['ammonia']*3 + ['water']*3, [1]*6)
    assert indices == [3, 4, 5, 0, 1, 2]

    # 5 monodentates of different sizes
    indices = smartreorderligs(['furan', 'ammonia', 'pyridine', 'carbonyl',
                                'water'], [1]*5)
    assert indices == [3, 4, 1, 0, 2]

    # Test bidentates
    indices = smartreorderligs(['acac', 'acac', 'en'], [2, 2, 2])
    assert indices == [2, 0, 1]

    indices = smartreorderligs(['en', 'en', 'acac'], [2, 2, 2])
    assert indices == [0, 1, 2]

    indices = smartreorderligs(['water', 'carbonyl', 'acac'], [1, 1, 2])
    assert indices == [2, 1, 0]

    # Tetradentate
    indices = smartreorderligs(['water', 'porphirine', 'misc'], [1, 4, 1])
    assert indices == [1, 0, 2]


def test_get_MLdist_database():
    water, _ = lig_load('water')
    ammonia, _ = lig_load('ammonia')
    connecting_atom = 0
    MLbonds = loaddata('/Data/ML.dat')

    dist, exact_match = get_MLdist_database(
        atom3D(Sym='Fe'), '2', '5', water, connecting_atom, 'water', MLbonds)
    assert exact_match
    assert dist == 2.12

    dist, exact_match = get_MLdist_database(
        atom3D(Sym='Co'), 'III', '5', ammonia,
        connecting_atom, 'ammonia', MLbonds)
    assert exact_match
    assert dist == 2.17

    # Test covariant radii fall back if not in database:
    dist, exact_match = get_MLdist_database(
        atom3D(Sym='Fe'), '2', '5', water, connecting_atom, 'water', {})

    assert exact_match is False
    assert dist == 1.98

    dist, exact_match = get_MLdist_database(
        atom3D(Sym='Cr'), 'II', '5', water, connecting_atom, 'water', {})

    assert exact_match is False
    assert dist == 2.0


def test_get_MLdist():
    water, _ = lig_load('water')
    connecting_atom = 0
    MLbonds = loaddata('/Data/ML.dat')
    this_diag = run_diag()

    # Test user defined BL (take second item from supplied list)
    dist = get_MLdist(atom3D(Sym='Fe'), '2', '5', water, connecting_atom,
                      'water', ['0', '0', '1.2', '0', '0', '0'], 2, False, 0.0,
                      this_diag, MLbonds)
    assert dist == 1.2

    # Test 'c' in bond list
    dist = get_MLdist(atom3D(Sym='Fe'), '2', '5', water, connecting_atom,
                      'water', ['0', '0', 'c', '0', '0', '0'], 2, False, 0.0,
                      this_diag, MLbonds)
    assert dist == 1.98

    # Test DB lookup
    dist = get_MLdist(atom3D(Sym='Fe'), '2', '5', water, connecting_atom,
                      'water', ['False']*6, 2, False, 0.0, this_diag, MLbonds)
    assert this_diag.dict_bondl
    assert dist == 2.12

    # Test covalent fall back
    dist = get_MLdist(atom3D(Sym='Fe'), '2', '5', water, connecting_atom,
                      'water', ['False']*6, 2, False, 0.0, this_diag, {})
    assert this_diag.dict_bondl
    assert dist == 1.98

    # No DB match: use ANN result
    dist = get_MLdist(atom3D(Sym='Fe'), '2', '5', water, connecting_atom,
                      'water', ['False']*6, 2, True, 3.14, this_diag, {})
    assert dist == 3.14


def test_init_ANN():
    licores = getlicores()

    # Test skipping:
    args = Namespace(skipANN=True)
    (ANN_flag, ANN_bondl, _,
     ANN_attributes, catalysis_flag) = init_ANN(
         args, ligands=['water']*6, occs=[1]*6, dents=[1]*6,
         batslist=[[1], [2], [3], [4], [5], [6]], tcats=[0]*6, licores=licores)

    assert ANN_flag is False
    assert ANN_bondl == [False] * 6
    assert ANN_attributes == dict()
    assert catalysis_flag is False

    # Test oldANN
    args = Namespace(skipANN=False, oldANN=True, core='Fe', decoration=False,
                     geometry='oct', oxstate='2', spin='5', debug=False,
                     exchange=0.2)
    (ANN_flag, ANN_bondl, _,
     ANN_attributes, catalysis_flag) = init_ANN(
         args, ligands=['water']*6, occs=[1]*6, dents=[1]*6,
         batslist=[[1], [2], [3], [4], [5], [6]], tcats=[0]*6, licores=licores)

    assert ANN_flag
    assert ANN_bondl == ANN_attributes['ANN_bondl']
    np.testing.assert_allclose(ANN_bondl, [2.0757] * 6, atol=1e-4)
    assert catalysis_flag is False

    # Test default ANN
    args = Namespace(skipANN=False, oldANN=False, core='Fe', decoration=False,
                     geometry='oct', oxstate='2', spin='5', debug=False,
                     exchange=0.2)
    (ANN_flag, ANN_bondl, _,
     ANN_attributes, catalysis_flag) = init_ANN(
         args, ligands=['water']*6, occs=[1]*6, dents=[1]*6,
         batslist=[[1], [2], [3], [4], [5], [6]], tcats=[0]*6, licores=licores)

    assert ANN_flag
    assert ANN_bondl == ANN_attributes['ANN_bondl']
    np.testing.assert_allclose(ANN_bondl, [2.1664] * 4 + [2.1349, 2.1218], atol=1e-4)
    assert catalysis_flag is False
