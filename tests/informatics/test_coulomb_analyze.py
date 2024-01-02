import numpy as np
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Informatics.coulomb_analyze import create_coulomb_matrix


def test_create_coulomb_matrix(resource_path_root):
    xyz_file = resource_path_root / "inputs" / "cr3_f6_optimization.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)
    cm = create_coulomb_matrix(mol)
    print(cm)
    nuclear_charges = np.array([24., 9., 9., 9., 9., 9., 9.])
    xyzs = mol.coordsvect()
    r = np.sqrt(np.sum((xyzs[:, np.newaxis, :] - xyzs[np.newaxis, :, :])**2, axis=-1))
    ref = np.outer(nuclear_charges, nuclear_charges)
    ref[r > 0.] /= r[r > 0.]
    ref[np.diag_indices_from(ref)] = 0.5 * nuclear_charges ** 2.4
    # Finally sort by L2 norm
    norms = np.linalg.norm(ref, axis=0)
    inds = np.argsort(norms)[::-1]
    ref = ref[inds, :]
    ref = ref[:, inds]
    print(ref)
    np.testing.assert_allclose(cm, ref, atol=1e-8)
