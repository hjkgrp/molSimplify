import numpy as np
from argparse import Namespace
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Scripts.distgeom import (GetCMDists, CosRule,
                                          Metrize, findshape)


def test_CosRule(atol=1e-6):
    # Test simple "unit" triangle
    assert abs(CosRule(1., 1., 90) - np.sqrt(2)) < atol
    assert abs(CosRule(1, np.sqrt(2), 45) - 1.) < atol
    assert abs(CosRule(np.sqrt(2), 1., 45) - 1.) < atol
    # Test equilateral triangle
    a = 3.456
    assert abs(CosRule(a, a, 60) - a) < atol
    # Test Pythagoras
    assert abs(CosRule(3., 4., 90) - 5.) < atol
    # Test 30-60-90 triangle
    a = 2.345
    assert abs(CosRule(a, 0.5*a*np.sqrt(3), 30) - 0.5*a) < atol
    assert abs(CosRule(a, 0.5*a, 60) - 0.5*a*np.sqrt(3)) < atol
    assert abs(CosRule(0.5*a, 0.5*a*np.sqrt(3), 90) - a) < atol
    # Test symmetry on 30-60-90
    assert abs(CosRule(0.5*a*np.sqrt(3), a, 30) - 0.5*a) < atol
    assert abs(CosRule(0.5*a, a, 60) - 0.5*a*np.sqrt(3)) < atol
    assert abs(CosRule(0.5*a*np.sqrt(3), 0.5*a, 90) - a) < atol


def test_GetCMDists():
    # Dummy points in 3d
    xyzs = np.array([[0., 0., 0.],
                     [1., 0., 0.],
                     [0., 1., 0.],
                     [0., 1., 1.],
                     [0., 0., 0.]])

    dist_mat = np.sqrt(np.sum(
        (xyzs[np.newaxis, :, :] - xyzs[:, np.newaxis, :])**2, axis=-1))
    # Calculate geometric center
    center = np.mean(xyzs, axis=0)
    # Distances to the center:
    dists_ref = np.sqrt(np.sum((xyzs - center)**2, axis=-1))

    dists, status = GetCMDists(dist_mat, len(xyzs))
    assert status
    np.testing.assert_allclose(dists, dists_ref)


def test_Metrize():
    """"A few tests are commented out because they would only be
    satisfied after introducing breaking changes to Metrize()
    (see TODO tags)"""
    # Setup
    natoms = 9
    LB = np.zeros((natoms, natoms))
    UB = 100*np.ones((natoms, natoms))
    # Build distance matrix
    D = Metrize(LB, UB, natoms)
    # Test bounds
    np.testing.assert_array_compare(np.greater_equal, D, LB)
    np.testing.assert_array_compare(np.less_equal, D, UB)
    # Test symmetry
    np.testing.assert_equal(D, D.T)
    # Diagonal should be zero
    # np.testing.assert_equal(np.diag(D), np.zeros(natoms))
    # Test triangle inequality. For each triangle of atoms i, j, and k:
    # r_ij <= r_ik + r_kj. TODO the current implementation occasionally
    # breaks the constraint on purpose for a potential speed up.
    # for i in range(natoms):
    #     for j in range(i, natoms):  # Test only lower triangular matrix
    #         for k in range(natoms):
    #             assert D[i, j] <= D[i, k] + D[k, j]
    # Test elements for specific seed
    # D = Metrize(LB, UB, natoms, seed=1234)
    # Reference for the seed
    # ref = np.array([[0.,        19.151945, 62.210877, 100.],
    #                 [19.151945, 0.,        43.772774, 100.],
    #                 [62.210877, 43.772774, 0.,        100.],
    #                 [100.,      100.,      100.,      0.]])
    # np.testing.assert_allclose(D, ref)


def test_findshape():
    # First trivial case of monodentate ligands:
    args = Namespace(geometry='oct', lig=['carbonyl', 'water'],
                     ligocc=[5, 1])

    angles_dict = findshape(args, mol3D(name='carbonyl'))
    assert angles_dict == {'0-0': 0.0}

    # More interesting bidentate case:
    args = Namespace(geometry='oct', lig=['acac', 'water'],
                     ligocc=[2, 2])

    angles_dict = findshape(args, mol3D(name='acac'))
    assert angles_dict == {'0-0': 0.0, '0-1': 90.0,
                           '1-0': 90.0, '1-1': 0.0}

    args = Namespace(geometry='oct', lig=['porphyrine', 'water'],
                     ligocc=[1, 2])

    angles_dict = findshape(args, mol3D(name='porphyrine'))
    assert angles_dict == {'0-0': 0.0, '0-1': 90.0, '0-2': 180.0, '0-3': 90.0,
                           '1-0': 90.0, '1-1': 0.0, '1-2': 90.0, '1-3': 180.0,
                           '2-0': 180.0, '2-1': 90.0, '2-2': 0.0, '2-3': 90.0,
                           '3-0': 90.0, '3-1': 180.0, '3-2': 90.0, '3-3': 0.0}
