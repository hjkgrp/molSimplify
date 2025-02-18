import pytest
from molSimplify.Informatics.MOF.PBC_functions import (
    compute_adj_matrix,
    compute_distance_matrix,
    fractional2cart,
    mkcell,
    readcif,
    solvent_removal,
    )
import numpy as np
import json

@pytest.mark.parametrize(
    "name",
    [
        "FOKYIP_clean",
        "SETDUS_clean",
        "UXUPEK_clean",
        "NEXXIZ_clean",
        "YICDAR_clean",
        "VONBIK_clean",
    ])
def test_cif_reading(resource_path_root, name):
    cpar, allatomtypes, fcoords = readcif(str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"))

    reference_cpar = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / "txt" / f"{name}_cpar.txt"))
    reference_allatomtypes = str(resource_path_root / "refs" / "informatics" / "mof" / "json" / f"{name}_allatomtypes.json")
    reference_fcoords = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / "txt" / f"{name}_fcoords.txt"))

    with open(reference_allatomtypes, 'r') as f:
        reference_allatomtypes = json.load(f)

    assert np.array_equal(cpar, reference_cpar)
    assert allatomtypes == reference_allatomtypes
    assert np.array_equal(fcoords, reference_fcoords)

@pytest.mark.parametrize(
    "name",
    [
        "FOKYIP_clean",
        "SETDUS_clean",
        "UXUPEK_clean",
        "NEXXIZ_clean",
        "YICDAR_clean",
        "VONBIK_clean",
    ])
def test_pairwise_distance_calc(resource_path_root, name):
    cpar, allatomtypes, fcoords = readcif(str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"))
    cell_v = mkcell(cpar)
    cart_coords = fractional2cart(fcoords, cell_v)
    distance_mat = compute_distance_matrix(cell_v, cart_coords)

    reference_mat = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / "txt" / f"{name}_distance_mat.txt"))
    assert np.allclose(distance_mat, reference_mat)

@pytest.mark.parametrize(
    "name",
    [
        "FOKYIP_clean",
        "SETDUS_clean",
        "UXUPEK_clean",
        "NEXXIZ_clean",
        "YICDAR_clean",
        "VONBIK_clean",
    ])
def test_adjacency_matrix_calc(resource_path_root, name):
    cpar, allatomtypes, fcoords = readcif(str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"))
    distance_mat = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / "txt" / f"{name}_distance_mat.txt"))

    adj_mat, _ = compute_adj_matrix(distance_mat, allatomtypes)
    adj_mat = adj_mat.todense()

    reference_mat = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / "txt" / f"{name}_adj_mat.txt"))
    assert np.array_equal(adj_mat, reference_mat)

@pytest.mark.parametrize(
    "name",
    [
        "Zn_MOF",
        "Co_MOF",
    ])
def test_solvent_removal(resource_path_root, tmp_path, name):
    input_geo = str(resource_path_root / "inputs" / "cif_files" / f"{name}_with_solvent.cif")
    output_path = str(tmp_path / f"{name}.cif")
    solvent_removal(input_geo, output_path)

    # Comparing two CIF files for equality
    reference_cif_path = str(resource_path_root / "refs" / "informatics" / "mof" / "cif" / f"{name}.cif")
    cpar1, allatomtypes1, fcoords1 = readcif(output_path)
    cpar2, allatomtypes2, fcoords2 = readcif(reference_cif_path)

    assert np.array_equal(cpar1, cpar2)
    assert allatomtypes1 == allatomtypes2
    assert np.array_equal(fcoords1, fcoords2)
