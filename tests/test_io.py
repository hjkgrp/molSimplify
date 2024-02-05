import numpy as np
from molSimplify.Scripts.io import lig_load


def test_lig_load(resource_path_root):
    lig_file = str(resource_path_root / "inputs" / "io" / "acac.mol2")
    mol, emsg = lig_load(lig_file)
    # Assert that the error message is empty
    assert not emsg
    # Convert to mol3D after loading the OBmol
    mol.convert2mol3D()
    # Load the reference from the ligand folder
    ref, _ = lig_load('acac')

    assert mol.natoms == ref.natoms
    assert all(mol.symvect() == ref.symvect())
    print(mol.coordsvect())
    print(ref.coordsvect())
    np.testing.assert_allclose(mol.coordsvect(), ref.coordsvect())
    assert mol.charge == ref.charge
