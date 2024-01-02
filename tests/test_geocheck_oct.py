import pytest
import helperFuncs as hp


@pytest.mark.parametrize("testName", [
    "all_flying_away",
    "broken_ligands",
    "catom_change",
    "H_transfer",
    "ligand_assemble",
    "ligand_bent",
    "linear_broken",
    "methane_trans",
    "rotational_group",
    "switch_test",
    "compact_bonding",
    "triplebond_linear_broken",
    "iodine_sulfur",
    "oct_comp_greedy",
    "atom_ordering_mismatch",
    "iodide_radius"
])
def test_geocheck_oct(tmpdir, resource_path_root, testName):
    thresh = 0.01
    passGeo = hp.runtestgeo(tmpdir, resource_path_root, testName, thresh)
    assert passGeo
