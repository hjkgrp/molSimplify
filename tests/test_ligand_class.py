from molSimplify.Classes.ligand import ligand


def test_ligand_class(resource_path_root):
    mol2_file = str(
        resource_path_root / "inputs" / "ligand_class" / "fe_acac.mol2"
    )
    lig = ligand(
        master_mol=None,
        index_list=None,
        dent=None,
        read_lig=mol2_file,
    )
    assert lig.dent == 2
    assert lig.master_mol.make_formula(latex=False) == "FeC5H7O2"

    lig_graph_det, mol2_str = lig.get_lig_mol2()
    assert lig_graph_det == '-2.023585127e+17'

    with open(resource_path_root / "inputs" / "ligand_class" / "acac_ref.mol2") as fin:
        ref_str = fin.read()
    # Remove last new line (because of our autoformatting)
    assert mol2_str[:-1] == ref_str

    buried_vol = lig.percent_buried_vol()
    assert abs(buried_vol - 30.034) < 1e-3
