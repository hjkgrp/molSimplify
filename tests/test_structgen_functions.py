import pytest
import numpy as np
from argparse import Namespace
from molSimplify.Classes import atom3D, mol3D, run_diag
from molSimplify.Scripts.io import loaddata, lig_load, getlicores
from molSimplify.Scripts.structgen import (smartreorderligs,
                                           getbackbcombsall,
                                           getnupdateb,
                                           get_MLdist_database,
                                           get_MLdist,
                                           init_template,
                                           init_ANN,
                                           align_dent3_lig,
                                           align_dent3_lig_old,
                                           mcomplex)
from molSimplify.Scripts.rmsd import rmsd
from pkg_resources import resource_filename, Requirement


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


@pytest.mark.parametrize(["geometry", "coordination"], [
    ('no', 1),
    ('tpl', 3),
    ('sqp', 4),
    # ('tdh', 4),  TODO: fix tdh (currently not centered correctly)
    ('spy', 5),
    ('tbp', 5),
    ('oct', 6),
    ('tpr', 6),
    ('pbp', 7),
    ('sqap', 8),
    # ('sq', 8),  TODO: Seems to be missing?
])
def test_init_template(geometry, coordination):
    args = Namespace(geometry=geometry, coord=coordination, ccatoms=None, ligloc=True,
                     pangles=False, distort='0', core='Fe', calccharge=True,
                     oxstate='II')
    (m3D, core3D, geom, backbatoms, coord,
     corerefatoms) = init_template(args, cpoints_required=6)

    assert [a.sym for a in m3D.getAtoms()] == ['Fe'] + coordination * ['X']
    assert core3D.getAtoms() == [atom3D('Fe')]
    assert backbatoms == getbackbcombsall(list(range(1, 1 + coordination)))
    assert geom == geometry
    assert coord == coordination
    assert corerefatoms.getAtoms() == [atom3D('Fe') for _ in range(coordination)]


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


def test_getnupdateb():
    backbone_atoms = [[5, 4, 6], [4, 5], [4, 6], [4], [5], [6]]
    batoms, backbone_atoms_modified = getnupdateb(backbone_atoms, 3)
    assert batoms == [5, 4, 6]
    assert backbone_atoms_modified == []

    backbone_atoms = [[5, 4, 6], [4, 5], [4, 6], [4], [5], [6]]
    batoms, backbone_atoms_modified = getnupdateb(backbone_atoms, 2)
    assert batoms == [4, 5]
    assert backbone_atoms_modified == [[6]]


def test_align_dent3_lig():
    args = Namespace(geometry='oct', coord=6, ccatoms=None, ligloc=True,
                     pangles=False, distort='0', core='Fe', calccharge=True,
                     oxstate='II', spin='1', debug=False)
    cpoint = atom3D(Sym="X", xyz=[0.0, -2.077, 0.0])
    batoms = [1, 2, 3]
    m3D, core3D, _, _, _, _ = init_template(args, cpoints_required=6)
    coreref = m3D.getAtom(0)
    ligand = "oxydiacetate_mer"
    lig3D, _ = lig_load(ligand)
    catoms = [11, 0, 12]
    MLb = False
    ANN_flag = False
    ANN_bondl = False
    this_diag = run_diag()
    MLbonds = loaddata('/Data/ML.dat')
    MLoptbds = []
    frozenats = []
    i = 0
    lig3D_aligned, frozenats, MLoptbds = align_dent3_lig(
        args, cpoint, batoms, m3D, core3D, coreref, ligand, lig3D, catoms, MLb,
        ANN_flag, ANN_bondl, this_diag, MLbonds, MLoptbds, frozenats, i)

    np.testing.assert_allclose(
        lig3D_aligned.coordsvect(),
        [[2.080, 0.0, 0.0],
         [2.894,  1.154, -0.012],
         [2.893, -1.154, -0.014],
         [1.832,  2.222, -0.018],
         [3.507,  1.188, -0.941],
         [3.550,  1.178,  0.887],
         [1.831, -2.222, -0.011],
         [3.535, -1.188,  0.896],
         [3.521, -1.178, -0.933],
         [2.173,  3.394, -0.039],
         [2.173, -3.394, -0.021],
         [0.512,  1.926,  0.0],
         [0.512, -1.926,  0.0]], atol=1e-2)
    assert frozenats == [2, 3, 4, 7]  # RM: Actually not sure thats what we want..
    assert MLoptbds == [2.08, 2.08, 2.08]


