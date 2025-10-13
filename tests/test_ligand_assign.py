from molSimplify.Classes.ligand import ligand_assign_original, ligand_breakdown
from molSimplify.Classes.mol3D import mol3D


def test_six_monodentate(resource_path_root):
    xyz_file = (resource_path_root / "inputs" / "ligand_assign_consistent"
                / "fe_water_ammonia_carbonyl_formaldehyde_hydrogensulfide_hydrocyanide.xyz")
    mol = mol3D()
    mol.readfromxyz(xyz_file)

    liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True)
    (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list,
     ax_con_int_list, eq_con_int_list, ax_con_list, eq_con_list,
     built_ligand_list) = ligand_assign_original(mol, liglist, ligdents, ligcons)
    # Expecting:
    # ax_ligands: ['water', 'carbonyl']
    # eq_ligands: ['hydrogensulfide', 'ammonia', 'hydrocyanide', 'formaldehyde']

    ax_formulas = [lig.mol.make_formula(latex=False) for lig in ax_ligand_list]
    assert ax_formulas == ['H2S', 'CHN']
    eq_formulas = [lig.mol.make_formula(latex=False) for lig in eq_ligand_list]
    assert eq_formulas == ['H2O', 'H3N', 'CO', 'CH2O']

    assert ax_natoms_list == [3, 3]
    assert eq_natoms_list == [3, 4, 2, 4]

    assert ax_con_int_list == [[0], [1]]
    assert eq_con_int_list == [[0], [0], [0], [1]]

    assert ax_con_list == [[14], [18]]
    assert eq_con_list == [[1], [4], [8], [11]]


def test_triple_bidentate(resource_path_root):
    xyz_file = (resource_path_root / "inputs" / "ligand_assign_consistent"
                / "fe_acac_bipy_bipy.xyz")
    mol = mol3D()
    mol.readfromxyz(xyz_file)

    liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True)
    (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list,
     ax_con_int_list, eq_con_int_list, ax_con_list, eq_con_list,
     built_ligand_list) = ligand_assign_original(mol, liglist, ligdents, ligcons)

    print(ax_ligand_list, eq_ligand_list)
    ax_formulas = [lig.mol.make_formula(latex=False) for lig in ax_ligand_list]
    assert ax_formulas == ['C5H7O2']
    eq_formulas = [lig.mol.make_formula(latex=False) for lig in eq_ligand_list]
    assert eq_formulas == ['C10H8N2', 'C10H8N2']

    assert ax_natoms_list == [14]
    assert eq_natoms_list == [20, 20]

    assert ax_con_int_list == [[0, 5]]
    assert eq_con_int_list == [[0, 1], [0, 1]]

    assert ax_con_list == [[1, 6]]
    assert eq_con_list == [[15, 16], [35, 36]]
