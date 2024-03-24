import pytest
import numpy as np
from molSimplify.Informatics.active_learning.expected_improvement import (
    get_2D_pareto_indices,
    get_2D_PI_and_centroid,
    get_2D_EI,
    get_2D_EHVI,
)


def test_get_pareto_indices():
    points = np.random.default_rng(0).normal(size=(500, 2))

    pareto_indices = get_2D_pareto_indices(points)
    np.testing.assert_equal(pareto_indices, [239, 119, 308, 95, 49, 230, 79, 206, 151])

    # Also test the 3 other quadrants:
    # Maximize dimension 0, minimize dimension 1
    points[:, 0] *= -1
    pareto_indices = get_2D_pareto_indices(points)
    np.testing.assert_equal(pareto_indices, [135, 331, 376, 247, 34, 151])

    # Maximize dimension 0, maximize dimension 1
    points[:, 1] *= -1
    pareto_indices = get_2D_pareto_indices(points)
    np.testing.assert_equal(pareto_indices, [135, 258, 23, 489, 466, 123, 109])

    # Minimize dimension 0, maximize dimension 1
    points[:, 0] *= -1
    pareto_indices = get_2D_pareto_indices(points)
    np.testing.assert_equal(pareto_indices, [239, 69, 175, 201, 389, 109])


@pytest.mark.parametrize(
    "method, ref_PI",
    [
        ("aug", [0.331885, 0.75219, 0.853196]),
        ("dom", [0.062788, 0.185605, 0.640678]),
        ("mix", [0.197337, 0.468897, 0.746937]),
    ]
)
def test_get_2D_PI_and_centroid(method, ref_PI, atol=1e-6):
    pred_mean = np.array([[5.0, 5.0], [3.0, 6.0], [3.0, 3.0]])
    pred_std = np.array([[1.2, 1.4], [1.1, 0.8], [2.6, 1.1]])
    pareto_points = np.array(
        [[2.0, 7.0], [4.0, 4.0], [6.0, 2.0], [8.0, 1.0]])

    PI_xy, centroid_xy = get_2D_PI_and_centroid(
        pred_mean, pred_std, pareto_points, method=method)
    # Now swap axis and assert that the PI is the same and that the centroids
    # also swapped axis.
    # NOTE: the pareto_points must also be reversed to satisfy the ordering constraints
    PI_yx, centroid_yx = get_2D_PI_and_centroid(
        pred_mean[:, ::-1], pred_std[:, ::-1], pareto_points[::-1, ::-1],
        method=method)

    np.testing.assert_allclose(PI_xy, PI_yx, atol=atol)
    np.testing.assert_allclose(centroid_xy, centroid_yx[:, ::-1], atol=atol)
    np.testing.assert_allclose(PI_xy, ref_PI, atol=atol)


@pytest.mark.parametrize(
    "method, ref_EI",
    [
        ("aug", [0.033024, 0.966226, 1.684631]),
        ("dom", [0.066208, 0.220915, 1.708273]),
        ("mix", [0.017599, 0.551167, 1.693937]),
    ]
)
def test_get_2D_EI(method, ref_EI, atol=1e-6):
    pred_mean = np.array([[5.0, 5.0], [3.0, 6.0], [3.0, 3.0]])
    pred_std = np.array([[1.2, 1.4], [1.1, 0.8], [2.6, 1.1]])
    pareto_points = np.array(
        [[2.0, 7.0], [4.0, 4.0], [6.0, 2.0], [8.0, 1.0]])

    EI_xy = get_2D_EI(pred_mean, pred_std, pareto_points, method=method)
    # Now swap axis and assert that the EI is the same
    # NOTE: the pareto_points must also be reversed to satisfy the ordering constraints
    EI_yx = get_2D_EI(pred_mean[:, ::-1], pred_std[:, ::-1], pareto_points[::-1, ::-1],
                      method=method)

    np.testing.assert_allclose(EI_xy, EI_yx, atol=atol)
    np.testing.assert_allclose(EI_xy, ref_EI, atol=atol)


@pytest.mark.parametrize(
    "pred_mean, pred_std, pareto_points, r, ref_EHVI",
    [
        (np.array([[0.0, 0.0]]),
         np.array([[0.1, 0.1]]),
         np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]),
         np.array([3.0, 3.0]),
         [3.07978846]),
        (np.array([[10.0, 10.0]]),
         np.array([[4.0, 4.0]]),
         np.array([[1.0, 2.0], [2.0, 1.0]]),
         np.array([11.0, 11.0]),
         [0.0726813]),
        (np.array([[5.0, 5.0], [3.0, 6.0], [3.0, 3.0]]),
         np.array([[1.2, 1.4], [1.1, 0.8], [2.6, 1.1]]),
         np.array([[2.0, 7.0], [4.0, 4.0], [6.0, 2.0], [8.0, 1.0]]),
         np.array([10.0, 10.0]),
         [0.4975774, 1.4780866, 10.2390386]),
    ]
)
def test_get_2D_EHVI(pred_mean, pred_std, pareto_points, r, ref_EHVI, atol=1e-6):
    EHVI_xy = get_2D_EHVI(pred_mean, pred_std, pareto_points, r)
    # Now swap axis and assert that the EHVI is the same
    # NOTE: the pareto_points must also be reversed to satisfy the ordering constraints
    EHVI_yx = get_2D_EHVI(pred_mean[:, ::-1], pred_std[:, ::-1], pareto_points[::-1, ::-1], r[::-1])

    np.testing.assert_allclose(EHVI_xy, EHVI_yx, atol=atol)
    np.testing.assert_allclose(EHVI_xy, ref_EHVI, atol=atol)
