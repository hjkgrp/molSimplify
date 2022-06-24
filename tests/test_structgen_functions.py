from molSimplify.Classes.atom3D import atom3D
from molSimplify.Scripts.io import loaddata, lig_load
from molSimplify.Scripts.structgen import (smartreorderligs,
                                           get_MLdist_database)


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

    dist, match = get_MLdist_database(
        atom3D(Sym='Fe'), '2', '5', water, connecting_atom, 'water', MLbonds)
    assert match
    assert dist == 2.12

    dist, match = get_MLdist_database(
        atom3D(Sym='Co'), 'III', '5', ammonia,
        connecting_atom, 'ammonia', MLbonds)
    assert match
    assert dist == 2.17

    # Test covariant radii fall back if not in database:
    dist, match = get_MLdist_database(
        atom3D(Sym='Fe'), '2', '5', water, connecting_atom, 'water', {})

    assert ~match
    assert dist == 1.98

    dist, match = get_MLdist_database(
        atom3D(Sym='Cr'), 'II', '5', water, connecting_atom, 'water', {})

    assert ~match
    assert dist == 2.0