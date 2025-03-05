import json
import pytest
import numpy as np
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.atom3D import atom3D
from molSimplify.Classes.globalvars import globalvars


def test_adding_and_deleting_atoms():
    mol = mol3D()
    mol.addAtom(atom3D(Sym='Fe'))

    assert mol.natoms == 1
    assert mol.findMetal() == [0]

    mol.addAtom(atom3D(Sym='Cu'))

    assert mol.natoms == 2
    assert mol.findMetal() == [0, 1]

    mol.deleteatom(0)

    assert mol.natoms == 1
    assert mol.findMetal() == [0]


def test_finding_and_counting_methods():
    mol = mol3D()
    mol.addAtom(atom3D(Sym='Fe'))
    for _ in range(6):
        mol.addAtom(atom3D(Sym='C'))
        mol.addAtom(atom3D(Sym='O'))

    # Test findAtomsbySymbol
    assert mol.findAtomsbySymbol(sym='C') == [1, 3, 5, 7, 9, 11]
    assert mol.findAtomsbySymbol(sym='O') == [2, 4, 6, 8, 10, 12]
    # Test getAtomwithSyms (allows for multiple symbols)
    ref_indices = [0, 2, 4, 6, 8, 10, 12]
    assert (mol.getAtomwithSyms(syms=['Fe', 'O'])
            == [mol.getAtom(i) for i in ref_indices])
    # optional argument allows to return just the indices:
    assert (mol.getAtomwithSyms(syms=['Fe', 'O'], return_index=True)
            == ref_indices)
    # Test mols_symbols
    mol.mols_symbols()
    assert mol.symbols_dict == {'Fe': 1, 'C': 6, 'O': 6}
    # Test count_nonH_atoms
    assert mol.count_nonH_atoms() == 13
    # Test count_atoms (exclude O)
    assert mol.count_atoms(exclude=['H', 'O']) == 7
    # Test count_specific_atoms
    assert mol.count_specific_atoms(atom_types=['C', 'O']) == 12
    # Test count_electrons
    assert mol.count_electrons(charge=2) == 24 + 6*6 + 6*8
    # Test findcloseMetal
    assert mol.findcloseMetal(mol.getAtom(-1)) == 0
    # Test findMetal
    assert mol.findMetal() == [0]
    # Test make_formula (sorted by atomic number)
    assert mol.make_formula(latex=False) == 'Fe1O6C6'
    assert (mol.make_formula(latex=True)
            == r'\textrm{Fe}_{1}\textrm{O}_{6}\textrm{C}_{6}')
    # Test typevect
    np.testing.assert_equal(mol.typevect(), np.array(['Fe'] + ['C', 'O']*6))


def test_add_bond():
    mol = mol3D()
    mol.addAtom(atom3D(Sym='O'))
    mol.addAtom(atom3D(Sym='C'))
    mol.addAtom(atom3D(Sym='H'))
    mol.addAtom(atom3D(Sym='H'))

    # Initialize empty bo_dict and graph
    mol.bo_dict = {}
    mol.graph = np.zeros((4, 4))

    mol.add_bond(0, 1, 2)
    mol.add_bond(1, 2, 1)
    mol.add_bond(1, 3, 1)

    assert mol.bo_dict == {(0, 1): 2, (1, 2): 1, (1, 3): 1}
    np.testing.assert_allclose(mol.graph, [[0, 2, 0, 0],
                                           [2, 0, 1, 1],
                                           [0, 1, 0, 0],
                                           [0, 1, 0, 0]])

    # Assert that bonding an atom to itself fails:
    with pytest.raises(IndexError):
        mol.add_bond(0, 0, 1)

    new_bo_dict = mol.get_bo_dict_from_inds([1, 2, 3])
    assert new_bo_dict == {(0, 1): 1, (0, 2): 1}

    assert mol.get_mol_graph_det(oct=False) == '-154582.1094'
    assert mol.get_mol_graph_det(oct=False, useBOMat=True) == '-154582.1094'


@pytest.mark.skip(reason='Mutating the state of an atom3D can not be detected '
                         ' by the mol3D class')
def test_mutating_atoms():
    mol = mol3D()
    mol.addAtom(atom3D(Sym='Fe'))
    assert mol.findMetal() == [0]

    mol.atoms[0].mutate('C')
    assert mol.findMetal() == []


def test_readfromxyz(resource_path_root):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / "cr3_f6_optimization.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)

    atoms_ref = [
        ("Cr", [-0.060052, -0.000019, -0.000023]),
        ("F", [1.802823, -0.010399, -0.004515]),
        ("F", [-0.070170, 1.865178, 0.0035660]),
        ("F", [-1.922959, 0.010197, 0.0049120]),
        ("F", [-0.049552, -1.865205, -0.0038600]),
        ("F", [-0.064742, 0.003876, 1.8531400]),
        ("F", [-0.055253, -0.003594, -1.8531790]),
    ]

    for atom, ref in zip(mol.atoms, atoms_ref):
        assert (atom.symbol(), atom.coords()) == ref

    # Test read_final_optim_step.
    mol = mol3D()
    mol.readfromxyz(xyz_file, read_final_optim_step=True)

    atoms_ref = [
        ("Cr", [-0.0599865612, 0.0000165451, 0.0000028031]),
        ("F", [1.8820549261, 0.0000076116, 0.0000163815]),
        ("F", [-0.0600064919, 1.9420510001, -0.0000022958]),
        ("F", [-2.0019508544, -0.0000130345, -0.0000067108]),
        ("F", [-0.0599967119, -1.9420284092, 0.0000133671]),
        ("F", [-0.0600235008, 0.0000085354, 1.9418467918]),
        ("F", [-0.0599958059, -0.0000082485, -1.9418293370]),
    ]

    for atom, ref in zip(mol.atoms, atoms_ref):
        assert (atom.symbol(), atom.coords()) == ref

    # Test readstring.
    my_str = '3\n\n\
    O         -5.34667        1.94740        0.01871\n\
    H         -4.35735        1.91881       -0.01321\n\
    H         -5.63604        1.31549       -0.68666\n'
    mol = mol3D()
    mol.readfromxyz(my_str, readstring=True)

    atoms_ref = [
        ("O", [-5.34667, 1.94740, 0.01871]),
        ("H", [-4.35735, 1.91881, -0.01321]),
        ("H", [-5.63604, 1.31549, -0.68666]),
    ]

    for atom, ref in zip(mol.atoms, atoms_ref):
        assert (atom.symbol(), atom.coords()) == ref


@pytest.mark.parametrize('name, geometry_str', [
    ('linear', 'linear'),
    ('trigonal_planar', 'trigonal planar'),
    ('t_shape', 'T shape'),
    ('trigonal_pyramidal', 'trigonal pyramidal'),
    ('tetrahedral', 'tetrahedral'),
    ('square_planar', 'square planar'),
    ('seesaw', 'seesaw'),
    ('trigonal_bipyramidal', 'trigonal bipyramidal'),
    ('square_pyramidal', 'square pyramidal'),
    # ('pentagonal_planar', 'pentagonal planar'),
    ('octahedral', 'octahedral'),
    # ('pentagonal_pyramidal', 'pentagonal pyramidal'),
    ('trigonal_prismatic', 'trigonal prismatic'),
    # ('pentagonal_bipyramidal', 'pentagonal bipyramidal')
    # ('square_antiprismatic', 'square antiprismatic'),
    # ('tricapped_trigonal_prismatic', 'tricapped trigonal prismatic'),
])
def test_get_geometry_type(resource_path_root, name, geometry_str):
    xyz_file = resource_path_root / "inputs" / "geometry_type" / f"{name}.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)

    geo_report = mol.get_geometry_type(debug=True)

    assert geo_report['geometry'] == geometry_str


def test_get_geometry_type_catoms_arr(resource_path_root):
    xyz_file = resource_path_root / "inputs" / "geometry_type" / "octahedral.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)

    with pytest.raises(ValueError):
        mol.get_geometry_type(catoms_arr=[1], debug=True)

    geo_report = mol.get_geometry_type(catoms_arr=[1, 4, 7, 10, 13, 16], debug=True)

    assert geo_report['geometry'] == 'octahedral'


@pytest.mark.parametrize(
    'name, geometry_str, hapticity',
    [
        ('BOWROX_comp_0.mol2', 'tetrahedral', [5, 1, 1, 1]),
        ('BOXTEQ_comp_0.mol2', 'tetrahedral', [6, 1, 1, 1]),
        ('BOXTIU_comp_0.mol2', 'tetrahedral', [6, 1, 1, 1]),
        ('BOZHOQ_comp_2.mol2', 'linear', [5, 5]),
        ('BOZHUW_comp_2.mol2', 'linear', [5, 5]),
        ('BUFLUM_comp_0.mol2', 'T shape', [2, 1, 1]),
        ('BUHMID_comp_0.mol2', 'trigonal planar', [3, 1, 1]),
        ('COYXUM_comp_0.mol2', 'tetrahedral', [5, 1, 1, 1]),
        ('COYYEX_comp_0.mol2', 'trigonal planar', [5, 1, 1]),
        ('COYYIB_comp_0.mol2', 'tetrahedral', [5, 1, 1, 1]),
    ]
)
def test_get_geometry_type_hapticity(resource_path_root, name, geometry_str, hapticity):
    input_file = resource_path_root / "inputs" / "hapticity_compounds" / name
    mol = mol3D()
    mol.readfrommol2(input_file)

    geo_report = mol.get_geometry_type(debug=True)

    print(geo_report)
    assert geo_report["geometry"] == geometry_str
    assert geo_report["hapticity"] == hapticity


@pytest.mark.parametrize(
    'name, con_atoms',
    [
        ('BOWROX_comp_0.mol2', [{3, 4, 5, 6, 7}]),
        ('BOXTEQ_comp_0.mol2', [{4, 5, 6, 7, 8, 9}]),
        ('BOXTIU_comp_0.mol2', [{2, 3, 5, 6, 8, 9}]),
        ('BOZHOQ_comp_2.mol2', [{1, 2, 3, 6, 8}, {4, 5, 7, 9, 10}]),
        ('BOZHUW_comp_2.mol2', [{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}]),
    ]
)
def test_is_sandwich_compound(resource_path_root, name, con_atoms):
    input_file = resource_path_root / "inputs" / "hapticity_compounds" / name
    mol = mol3D()
    mol.readfrommol2(input_file)

    num_sandwich_lig, info_sandwich_lig, aromatic, allconnect, sandwich_lig_atoms = mol.is_sandwich_compound()

    assert num_sandwich_lig == len(con_atoms)
    assert aromatic
    assert allconnect
    for i, (info, lig) in enumerate(zip(info_sandwich_lig, sandwich_lig_atoms)):
        assert info["aromatic"]
        assert info["natoms_connected"] == len(con_atoms[i])
        assert info["natoms_ring"] == len(con_atoms[i])
        assert lig["atom_idxs"] == con_atoms[i]


@pytest.mark.parametrize(
    'name, con_atoms',
    [
        ("BUFLUM_comp_0.mol2", [{2, 4}]),
        ("BUHMID_comp_0.mol2", [{3, 4, 5}]),
    ]
)
def test_is_edge_compound(resource_path_root, name, con_atoms):
    input_file = resource_path_root / "inputs" / "hapticity_compounds" / name
    mol = mol3D()
    mol.readfrommol2(input_file)

    num_edge_lig, info_edge_lig, edge_lig_atoms = mol.is_edge_compound()

    assert num_edge_lig == len(con_atoms)
    for i, (info, lig) in enumerate(zip(info_edge_lig, edge_lig_atoms)):
        assert info["natoms_connected"] == len(con_atoms[i])
        assert lig["atom_idxs"] == con_atoms[i]


def test_mol3D_from_smiles_macrocycles():
    """Uses an examples from Aditya's macrocycles that were previously
    converted wrong.
    """
    smiles = "C9SC(=CCSC(CSC(=NCSC9)))"
    mol = mol3D.from_smiles(smiles)
    assert mol.natoms == 29

    ref_graph = np.zeros([mol.natoms, mol.natoms])
    ref_bo_graph = np.zeros([mol.natoms, mol.natoms])
    bonds = [
        (21, 7, 1.0),
        (29, 14, 1.0),
        (13, 14, 1.0),
        (13, 12, 1.0),
        (9, 10, 1.0),
        (9, 8, 1.0),
        (27, 12, 1.0),
        (6, 7, 1.0),
        (6, 5, 1.0),
        (14, 28, 1.0),
        (14, 1, 1.0),
        (7, 8, 1.0),
        (7, 22, 1.0),
        (2, 1, 1.0),
        (2, 3, 1.0),
        (24, 8, 1.0),
        (12, 11, 1.0),
        (12, 26, 1.0),
        (10, 11, 2.0),
        (10, 25, 1.0),
        (8, 23, 1.0),
        (1, 15, 1.0),
        (1, 16, 1.0),
        (3, 17, 1.0),
        (3, 4, 2.0),
        (5, 19, 1.0),
        (5, 4, 1.0),
        (5, 20, 1.0),
        (4, 18, 1.0),
    ]
    for bond in bonds:
        i, j = bond[0] - 1, bond[1] - 1
        ref_graph[i, j] = ref_graph[j, i] = 1
        ref_bo_graph[i, j] = ref_bo_graph[j, i] = bond[2]

    np.testing.assert_allclose(mol.graph, ref_graph)
    np.testing.assert_allclose(mol.bo_graph, ref_bo_graph)


def test_mol3D_from_smiles_benzene():
    smiles = "c1ccccc1"
    mol = mol3D.from_smiles(smiles)
    assert mol.natoms == 12

    ref_graph = np.zeros([mol.natoms, mol.natoms])
    ref_bo_graph = np.zeros([mol.natoms, mol.natoms])
    bonds = [
        (1, 2, 1.5),
        (2, 3, 1.5),
        (3, 4, 1.5),
        (4, 5, 1.5),
        (5, 6, 1.5),
        (1, 6, 1.5),
        (1, 7, 1.0),
        (2, 8, 1.0),
        (3, 9, 1.0),
        (4, 10, 1.0),
        (5, 11, 1.0),
        (6, 12, 1.0),
    ]
    for bond in bonds:
        i, j = bond[0] - 1, bond[1] - 1
        ref_graph[i, j] = ref_graph[j, i] = 1
        ref_bo_graph[i, j] = ref_bo_graph[j, i] = bond[2]

    np.testing.assert_allclose(mol.graph, ref_graph)
    np.testing.assert_allclose(mol.bo_graph, ref_bo_graph)


@pytest.mark.parametrize(
    "geo_type, key",
    [
        ('linear', 'linear'),
        ('trigonal_planar', 'trigonal planar'),
        ('t_shape', 'T shape'),
        ('trigonal_pyramidal', 'trigonal pyramidal'),
        ('tetrahedral', 'tetrahedral'),
        ('square_planar', 'square planar'),
        ('seesaw', 'seesaw'),
        ('trigonal_bipyramidal', 'trigonal bipyramidal'),
        ('square_pyramidal', 'square pyramidal'),
        ('pentagonal_planar', 'pentagonal planar'),
        ('octahedral', 'octahedral'),
        ('pentagonal_pyramidal', 'pentagonal pyramidal'),
        # ('trigonal_prismatic', 'trigonal prismatic'),
        # ('pentagonal_bipyramidal', 'pentagonal bipyramidal'),
        # ('square_antiprismatic', 'square antiprismatic'),
        # ('tricapped_trigonal_prismatic', 'tricapped trigonal prismatic'),
    ]
)
def test_dev_from_ideal_geometry(resource_path_root, geo_type, key):
    mol = mol3D()
    mol.readfromxyz(resource_path_root / "inputs" / "geometry_type" / f"{geo_type}.xyz")

    globs = globalvars()
    polyhedra = globs.get_all_polyhedra()
    rmsd, max_dev = mol.dev_from_ideal_geometry(polyhedra[key])

    print(polyhedra[key])

    assert rmsd < 1e-3
    assert max_dev < 1e-3


@pytest.mark.parametrize(
    "geo_type, ref",
    [
        ('linear', 'linear'),
        ('trigonal_planar', 'trigonal planar'),
        ('t_shape', 'T shape'),
        ('trigonal_pyramidal', 'trigonal pyramidal'),
        ('tetrahedral', 'tetrahedral'),
        ('square_planar', 'square planar'),
        ('seesaw', 'seesaw'),
        ('trigonal_bipyramidal', 'trigonal bipyramidal'),
        ('square_pyramidal', 'square pyramidal'),
        ('pentagonal_planar', 'pentagonal planar'),
        ('octahedral', 'octahedral'),
        ('pentagonal_pyramidal', 'pentagonal pyramidal'),
        ('trigonal_prismatic', 'trigonal prismatic'),
        # ('pentagonal_bipyramidal', 'pentagonal bipyramidal'),
        # ('square_antiprismatic', 'square antiprismatic'),
        # ('tricapped_trigonal_prismatic', 'tricapped trigonal prismatic'),
    ]
)
def test_geo_geometry_type_distance(resource_path_root, geo_type, ref):
    mol = mol3D()
    mol.readfromxyz(resource_path_root / "inputs" / "geometry_type" / f"{geo_type}.xyz")

    result = mol.get_geometry_type_distance()
    print(result)
    assert result['geometry'] == ref


@pytest.mark.parametrize(
    "geo",
    [
    "benzene",
    "co",
    "cr3_f6_optimization",
    ])
def test_graph_hash(resource_path_root, geo):
    # Note: May fail if a very different version of networkx is used
    # compared to that used for the reference.
    mol = mol3D()
    mol.readfromxyz(resource_path_root / "inputs" / "xyz_files" / f"{geo}.xyz")
    gh = mol.get_graph_hash(attributed_flag=True, oct=False)

    reference_path = str(resource_path_root / "refs" / "graph_hash" / f"{geo}.txt")
    with open(reference_path, 'r') as f:
        reference_gh = f.readline()

    reference_gh = reference_gh.rstrip() # Remove trailing newline.

    assert gh == reference_gh

@pytest.mark.parametrize(
    "name, idx, sym, coords",
    [
        ("caffeine", 6, "N", [-2.27577, 0.41807, 0.00000]),
        ("caffeine", 18, "O", [-7.19749, -1.54201, 0.00000]),
        ("phenanthroline", 20, "H", [-2.89683, 2.18661, -0.00000]),
        ("taurine", 5, "C", [-2.02153, 1.84435, 0.11051]),
    ]
)
def test_getAtom(resource_path_root, name, idx, sym, coords):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)

    atom = mol.getAtom(idx)
    assert atom.symbol() == sym
    assert atom.coords() == coords


@pytest.mark.parametrize(
    "name, transition_metals_only, correct_answer",
    [
    ("benzene", True, []),
    ("benzene", False, []),
    ("fe_complex", True, [6]),
    ("fe_complex", False, [6]),
    ("in_complex", True, []),
    ("in_complex", False, [0]),
    ("bimetallic_al_complex", True, []),
    ("bimetallic_al_complex", False, [0,13]),
    ("UiO-66_sbu", True, [0,1,2,3,4,5]),
    ("UiO-66_sbu", False, [0,1,2,3,4,5]),
    ])
def test_findMetal(resource_path_root, name, transition_metals_only, correct_answer):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)
    metal_list = mol.findMetal(transition_metals_only=transition_metals_only)
    assert metal_list == correct_answer


@pytest.mark.parametrize(
    "name, oct_flag",
    [
    ("fe_complex", True),
    ("fe_complex", False),
    ("caffeine", True),
    ("caffeine", False),
    ("FIrpic", True),
    ("FIrpic", False),
    ])
def test_createMolecularGraph(resource_path_root, name, oct_flag):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)
    mol.createMolecularGraph(oct=oct_flag)

    reference_path = resource_path_root / "refs" / "json" / "createMolecularGraph" / f"{name}_oct_{oct_flag}_graph.json"
    with open(reference_path, 'r') as f:
        reference_graph = json.load(f)

    # For saving np arrays to json, need to cast to list.
    # Convert back for comparison.
    reference_graph = np.array(reference_graph)

    assert np.array_equal(reference_graph, mol.get_graph())


@pytest.mark.parametrize(
    "name, idx, get_graph, correct_answer",
    [
    ("fe_complex", 6, True, [0,2,4,7,9,11]),
    ("fe_complex", 6, False, []),
    ("caffeine", 4, True, [3,5,19]),
    ("caffeine", 4, False, [3,5,19]),
    ("FIrpic", 26, True, [0,23,25]),
    ("FIrpic", 26, False, [0,23,25]),
    ("FIrpic", 0, True, [6,13,26,33,44,52]),
    ("FIrpic", 0, False, [6,13,26,33,44,52]),
    ])
def test_getBondedAtoms(resource_path_root, name, idx, get_graph, correct_answer):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)
    if get_graph:
        mol.createMolecularGraph()

    nats = mol.getBondedAtoms(idx)
    assert nats == correct_answer


@pytest.mark.parametrize(
    "name, idx, oct_flag, correct_answer",
    [
    ("fe_complex", 6, True, [0,2,4,7,9,11]),
    ("fe_complex", 6, False, []),
    ("caffeine", 4, True, [3,5,19]),
    ("caffeine", 4, False, [3,5,19]),
    ("FIrpic", 26, True, [0,23,25]),
    ("FIrpic", 26, False, [0,23,25]),
    ("FIrpic", 0, True, [6,13,26,33,44,52]),
    ("FIrpic", 0, False, [6,13,26,33,44,52]),
    ])
def test_getBondedAtomsSmart(resource_path_root, name, idx, oct_flag, correct_answer):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)

    nats = mol.getBondedAtomsSmart(idx, oct=oct_flag)
    assert nats == correct_answer


@pytest.mark.parametrize(
    "name, idx, correct_answer",
    [
    ("fe_complex", 6, [0,2,4,7,9,11]),
    ("caffeine", 4, [3,5,19]),
    ("caffeine", 7, [6,8]),
    ("caffeine", 9, [6]),
    ])
def test_getBondedAtomsnotH(resource_path_root, name, idx, correct_answer):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)

    nats = mol.getBondedAtomsnotH(idx)
    assert nats == correct_answer


@pytest.mark.parametrize(
    "name, idx, correct_answer",
    [
    ("fe_complex", 6, [0,2,4,7,9,11]),
    ("FIrpic", 0, [6,13,26,33,44,52]),
    ("cr3_f6_optimization", 0, [1,2,3,4,5,6]),
    ])
def test_getBondedAtomsOct(resource_path_root, name, idx, correct_answer):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)

    nats = mol.getBondedAtomsOct(idx)
    assert nats == correct_answer


@pytest.mark.parametrize(
    "name, return_graph",
    [
    ('HKUST-1_linker', True),
    ('HKUST-1_linker', False),
    ('HKUST-1_sbu', True),
    ('HKUST-1_sbu', False),
    ])
def test_assign_graph_from_net(resource_path_root, name, return_graph):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    net_file = resource_path_root / "inputs" / "net_files" / f"{name}.net"
    mol = mol3D()
    mol.readfromxyz(xyz_file)

    if return_graph:
        graph = mol.assign_graph_from_net(net_file, return_graph=return_graph)
    else:
        mol.assign_graph_from_net(net_file, return_graph=return_graph)
        graph = mol.get_graph()

    reference_path = resource_path_root / "refs" / "json" / "assign_graph_from_net" / f"{name}_graph.json"
    with open(reference_path, 'r') as f:
        reference_graph = json.load(f)

    # For saving np arrays to json, need to cast to list.
    # Convert back for comparison.
    reference_graph = np.array(reference_graph)

    assert np.array_equal(reference_graph, graph)


@pytest.mark.parametrize(
    "name, force_clean_flag",
    [
    ('caffeine', True),
    ('caffeine', False),
    ('cr3_f6_optimization', True),
    ('cr3_f6_optimization', False),
    ('taurine', True),
    ('taurine', False),
    ])
def test_convert2OBMol(resource_path_root, name, force_clean_flag):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    reference_path = resource_path_root / "refs" / "json" / "convert2OBMol" / f"{name}_fc_{force_clean_flag}_OB_dict.json"
    with open(reference_path, 'r') as f:
        reference_dict = json.load(f)

    mol = mol3D()
    mol.readfromxyz(xyz_file)
    mol.convert2OBMol(force_clean=force_clean_flag)

    assert reference_dict['NumAtoms'] == mol.OBMol.NumAtoms()
    assert reference_dict['NumBonds'] == mol.OBMol.NumBonds()
    assert reference_dict['NumHvyAtoms'] == mol.OBMol.NumHvyAtoms()
    assert reference_dict['NumRotors'] == mol.OBMol.NumRotors()
    assert reference_dict['Atom4GetType'] == mol.OBMol.GetAtom(4).GetType()
    assert reference_dict['Atom4GetY'] == mol.OBMol.GetAtom(4).GetY()
    # assert reference_dict['Bond4GetBondOrder'] == mol.OBMol.GetBond(4).GetBondOrder()
    # assert reference_dict['Bond4GetBeginAtomIdx'] == mol.OBMol.GetBond(4).GetBeginAtomIdx()
    # assert reference_dict['Bond4GetEndAtomIdx'] == mol.OBMol.GetBond(4).GetEndAtomIdx()

    # Seems version of Open Babel changes the ordering of bonds (e.g., which
    # bond is at index 4).


@pytest.mark.parametrize(
    "name",
    [
    'caffeine',
    'cr3_f6_optimization',
    'taurine',
    ])
def test_convert2OBMol2(resource_path_root, name):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    reference_path = resource_path_root / "refs" / "json" / "convert2OBMol2" /  f"{name}_OB_dict.json"
    with open(reference_path, 'r') as f:
        reference_dict = json.load(f)
    reference_path = resource_path_root / "refs" / "json" / "convert2OBMol2" /  f"{name}_BO_mat.json"
    with open(reference_path, 'r') as f:
        reference_BO_mat = json.load(f)

    # For saving np arrays to json, need to cast to list.
    # Convert back for comparison.
    reference_BO_mat = np.array(reference_BO_mat)

    mol = mol3D()
    mol.readfromxyz(xyz_file)
    mol.convert2OBMol2()

    assert reference_dict['NumAtoms'] == mol.OBMol.NumAtoms()
    assert reference_dict['NumBonds'] == mol.OBMol.NumBonds()
    assert reference_dict['NumHvyAtoms'] == mol.OBMol.NumHvyAtoms()
    assert reference_dict['NumRotors'] == mol.OBMol.NumRotors()
    assert reference_dict['Atom4GetType'] == mol.OBMol.GetAtom(4).GetType()
    assert reference_dict['Atom4GetY'] == mol.OBMol.GetAtom(4).GetY()
    # assert reference_dict['Bond4GetBondOrder'] == mol.OBMol.GetBond(4).GetBondOrder()
    # assert reference_dict['Bond4GetBeginAtomIdx'] == mol.OBMol.GetBond(4).GetBeginAtomIdx()
    # assert reference_dict['Bond4GetEndAtomIdx'] == mol.OBMol.GetBond(4).GetEndAtomIdx()

    assert np.array_equal(reference_BO_mat, mol.BO_mat)


@pytest.mark.parametrize(
    "name, canonicalize, use_mol2, correct_smiles",
    [
    ('phenanthroline', True, True, 'c1ccc2c(n1)c1ncccc1cc2'),
    ('phenanthroline', True, False, 'c1ccc2c(n1)c1ncccc1cc2'),
    ('phenanthroline', False, True, 'c1c2c(ncc1)c1c(cc2)cccn1'),
    ('phenanthroline', False, False, 'c1c2c(ncc1)c1c(cc2)cccn1'),
    ('taurine', True, True, 'NCCS([O])([O])O'),
    ('taurine', True, False, 'NCCS(O)([O])[O]'),
    ('taurine', False, True, 'S([O])([O])(O)CCN'),
    ('taurine', False, False, 'S([O])([O])(O)CCN'),
    ])
def test_get_smiles(resource_path_root, name, canonicalize, use_mol2, correct_smiles):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)
    smiles = mol.get_smiles(canonicalize=canonicalize, use_mol2=use_mol2)
    assert smiles == correct_smiles


@pytest.mark.parametrize(
    "name, correct_answer",
    [
    ('caffeine', [-4.41475,-0.30732,0]),
    ('HKUST-1_sbu', [-4.64738,-2.68238,7.59]),
    ]
    )
def test_centermass(resource_path_root, name, correct_answer):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)
    center_of_mass = mol.centermass()
    assert np.allclose(center_of_mass, correct_answer, atol=1e-5)


@pytest.mark.parametrize(
    "name, bonddict",
    [
    ("benzene", True),
    ("benzene", False),
    ("caffeine", True),
    ("caffeine", False),
    ]
    )
def test_populateBOMatrix(resource_path_root, name, bonddict):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)
    mol.convert2OBMol()
    molBOMat = mol.populateBOMatrix(bonddict=bonddict)

    if bonddict:
        reference_path = resource_path_root / "refs" / "json" / "populateBOMatrix" /  f"{name}_bo_dict_{bonddict}.json"
        with open(reference_path, 'r') as f:
            reference_bo_dict = json.load(f)
        # Needed to adjust the reference dictionary in order
        # to save to json.
        # So, need to adjust this dictionary for the
        # comparison.
        mod_bo_dict = {str(k): v for k, v in mol.bo_dict.items()}
        assert mod_bo_dict == reference_bo_dict

    reference_path = resource_path_root / "refs" / "json" / "populateBOMatrix" /  f"{name}_molBOMat.json"
    with open(reference_path, 'r') as f:
        reference_BOMat = json.load(f)

    # For saving np arrays to json, need to cast to list.
    # Convert back for comparison.
    assert np.array_equal(molBOMat, np.array(reference_BOMat))


@pytest.mark.parametrize(
    "name, idx, bo_dict_flag, graph_flag",
    [
    ("FIrpic", 13, True, True),
    ("FIrpic", 13, True, False),
    ("FIrpic", 13, False, True),
    ("FIrpic", 13, False, False),
    ("penicillin", 16, True, True),
    ("penicillin", 16, True, False),
    ("penicillin", 16, False, True),
    ("penicillin", 16, False, False),
    ]
    )
def test_deleteatom(resource_path_root, name, idx, bo_dict_flag, graph_flag):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)

    reference_path = resource_path_root / "refs" / "json" / "deleteatom" /  f"{name}_bo_dict_{bo_dict_flag}_graph_{graph_flag}.json"
    with open(reference_path, 'r') as f:
        reference_dict = json.load(f)

    if bo_dict_flag:
        mol.convert2OBMol()
        mol.populateBOMatrix(bonddict=True)
    if graph_flag:
        mol.createMolecularGraph(oct=False)

    mol.deleteatom(idx)

    assert mol.natoms == len(mol.atoms)
    assert mol.mass == reference_dict['mass']
    assert mol.natoms == reference_dict['natoms']
    assert len(mol.atoms) == reference_dict['len_atoms']

    if mol.bo_dict:
        # Needed to adjust the reference dictionary in order
        # to save to json.
        # So, need to adjust this dictionary for the
        # comparison.
        mod_bo_dict = {str(k): v for k, v in mol.bo_dict.items()}
    else:
        mod_bo_dict = {}
    assert mod_bo_dict == reference_dict['bo_dict']

    if len(mol.graph):
        # For saving np arrays to json, need to cast to list.
        # Convert back for comparison.
        assert np.array_equal(mol.graph, np.array(reference_dict['graph']))
    else:
        assert mol.graph == reference_dict['graph']


@pytest.mark.parametrize(
    "name, idxs, bo_dict_flag, graph_flag",
    [
    ("FIrpic", [13, 40, 27, 21], True, True),
    ("FIrpic", [13, 21, 27, 40], True, True),
    ("FIrpic", [13, 40, 27, 21], True, False),
    ("FIrpic", [13, 40, 27, 21], False, True),
    ("FIrpic", [13, 40, 27, 21], False, False),
    ("penicillin", [16, 6, 12, 34], True, True),
    ("penicillin", [16, 6, 12, 34], True, False),
    ("penicillin", [16, 6, 12, 34], False, True),
    ("penicillin", [16, 6, 12, 34], False, False),
    ]
    )
def test_deleteatoms(resource_path_root, name, idxs, bo_dict_flag, graph_flag):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"{name}.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)

    reference_path = resource_path_root / "refs" / "json" / "deleteatoms" /  f"{name}_bo_dict_{bo_dict_flag}_graph_{graph_flag}.json"
    with open(reference_path, 'r') as f:
        reference_dict = json.load(f)

    if bo_dict_flag:
        mol.convert2OBMol()
        mol.populateBOMatrix(bonddict=True)
    if graph_flag:
        mol.createMolecularGraph(oct=False)

    mol.deleteatoms(idxs)

    assert mol.natoms == len(mol.atoms)
    assert mol.mass == reference_dict['mass']
    assert mol.natoms == reference_dict['natoms']
    assert len(mol.atoms) == reference_dict['len_atoms']

    if mol.bo_dict:
        # Needed to adjust the reference dictionary in order
        # to save to json.
        # So, need to adjust this dictionary for the
        # comparison.
        mod_bo_dict = {str(k): v for k, v in mol.bo_dict.items()}
    else:
        mod_bo_dict = {}
    assert mod_bo_dict == reference_dict['bo_dict']

    if len(mol.graph):
        # For saving np arrays to json, need to cast to list.
        # Convert back for comparison.
        assert np.array_equal(mol.graph, np.array(reference_dict['graph']))
    else:
        assert mol.graph == reference_dict['graph']


# TODO hapticity_compounds/BOWROX_comp_0.mol2
# TODO io/formaldehyde.mol2
# def test_readfrommol2(resource_path_root):
#     pass
# TODO readstring
# TODO check graph, bo_graph, bo_graph_trunc, bo_dict


# def test_writemol2(resource_path_root, tmp_path):
#     pass
# TODO writestring, ignoreX


# def test_writexyz(resource_path_root, tmp_path):
#     pass
# TODO writestring, withgraph, no_tabs, ignoreX
