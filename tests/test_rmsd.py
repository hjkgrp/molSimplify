import pytest
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Scripts.rmsd import rigorous_rmsd


@pytest.mark.parametrize(
    'path1,path2,ref_hungarian,ref_none',
    [['example_1_noff', 'example_1', 0.3991, 0.7749],
     ['BUWGOQ', 'BUWGOQ_final', 2.43958, 0.49113],
     ['BUWGOQ_noH', 'BUWGOQ_noH_final', 1.74814, 0.11826],
     ['BUWGOQ', 'BUWGOQ_final_reordered', 2.43958, 3.02598]]
    )
def test_rigorous_rmsd(resource_path_root, path1, path2, ref_hungarian, ref_none, atol=1e-3):
    # Reference values calculated using https://pypi.org/project/rmsd/
    # >>> calculate_rmsd --reorder path1.xyz path2.xyz
    # >>> calculate_rmsd path1.xyz path2.xyz
    xyz_1 = resource_path_root / "inputs" / "rmsd" / f"{path1}.xyz"
    mol1 = mol3D()
    mol1.readfromxyz(xyz_1)

    xyz_2 = resource_path_root / "inputs" / "rmsd" / f"{path2}.xyz"
    mol2 = mol3D()
    mol2.readfromxyz(xyz_2)

    r = rigorous_rmsd(mol1, mol2, reorder='hungarian')
    assert abs(r - ref_hungarian) < atol

    r = rigorous_rmsd(mol1, mol2, reorder='none')
    assert abs(r - ref_none) < atol
