from molSimplify.Classes.ligand import ligand_assign_consistent, ligand_breakdown
from molSimplify.Classes.mol3D import mol3D


def test_six_monodentate(resource_path_root):
    xyz_file = (resource_path_root / "inputs" / "ligand_assign_consistent"
                / "fe_water_ammonia_carbonyl_formaldehyde_hydrogensulfide_hydrocyanide.xyz")
    mol = mol3D()
    mol.readfromxyz(xyz_file)

    liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True)
    (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list,
     ax_con_int_list, eq_con_int_list, ax_con_list, eq_con_list,
     built_ligand_list) = ligand_assign_consistent(mol, liglist, ligdents,
                                                   ligcons)
    # Expecting:
    # ax_ligands: ['water', 'carbonyl']
    # eq_ligands: ['hydrogensulfide', 'ammonia', 'hydrocyanide', 'formaldehyde']

    ax_formulas = [lig.mol.make_formula(latex=False) for lig in ax_ligand_list]
    assert ax_formulas == ['O1H2', 'O1C1']
    eq_formulas = [lig.mol.make_formula(latex=False) for lig in eq_ligand_list]
    assert eq_formulas == ['S1H2', 'N1H3', 'N1C1H1', 'O1C1H2']

    assert ax_natoms_list == [3, 2]
    assert eq_natoms_list == [3, 4, 3, 4]

    assert ax_con_int_list == [[0], [0]]
    assert eq_con_int_list == [[0], [0], [1], [1]]

    assert ax_con_list == [[1], [8]]
    assert eq_con_list == [[14], [4], [18], [11]]


def test_triple_bidentate(resource_path_root):
    xyz_file = (resource_path_root / "inputs" / "ligand_assign_consistent"
                / "fe_acac_bipy_bipy.xyz")
    mol = mol3D()
    mol.readfromxyz(xyz_file)

    liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True)
    (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list,
     ax_con_int_list, eq_con_int_list, ax_con_list, eq_con_list,
     built_ligand_list) = ligand_assign_consistent(mol, liglist, ligdents,
                                                   ligcons)

    print(ax_ligand_list, eq_ligand_list)
    ax_formulas = [lig.mol.make_formula(latex=False) for lig in ax_ligand_list]
    assert ax_formulas == ['O2C5H7', 'N2C10H8']
    eq_formulas = [lig.mol.make_formula(latex=False) for lig in eq_ligand_list]
    assert eq_formulas == ['O2C5H7', 'N2C10H8', 'N2C10H8']

    assert ax_natoms_list == [14, 20]
    assert eq_natoms_list == [14, 20, 20]

    assert ax_con_int_list == [[5], [0]]
    assert eq_con_int_list == [[0], [1, 0], [1]]

    assert ax_con_list == [[6], [35]]
    assert eq_con_list == [[1], [16, 15], [36]]
