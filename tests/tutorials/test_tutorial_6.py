import os

import pytest

import helperFuncs as hp

# Skip slow test in CI (takes >10s); still runs when pytest is invoked locally.
_IN_CI = os.environ.get("CI", "").lower() in ("true", "1")


# Tutorial 6: place_on_slab — single CO on Pd slab (1co case).
def test_tutorial_6_1co(tmp_path, resource_path_root):
    testName = "tutorial_6"
    threshOG = 2.0
    [passNumAtoms, passOG] = hp.runtest_molecule_on_slab(
        tmp_path, resource_path_root, testName, threshOG)
    assert passNumAtoms
    assert passOG


def test_tutorial_6_stag_3co(tmp_path, resource_path_root):
    """Tutorial 6 variant: staggered 3× CO on Pd slab (num_placements=3). Uses placement_seed for reproducibility."""
    testName = "tutorial_6_stag_3co"
    threshOG = 2.0
    [passNumAtoms, passOG] = hp.runtest_molecule_on_slab(
        tmp_path, resource_path_root, testName, threshOG)
    assert passNumAtoms
    assert passOG


def test_tutorial_6_mno4(tmp_path, resource_path_root):
    """Tutorial 6 variant: MnO4-like fragment (mno5.xyz) on Pd slab, 4 surface atoms."""
    testName = "tutorial_6_mno4"
    threshOG = 2.0
    [passNumAtoms, passOG] = hp.runtest_molecule_on_slab(
        tmp_path, resource_path_root, testName, threshOG,
        xyz_relative_paths={'-target_molecule': "../xyz_files/mno5.xyz"})
    assert passNumAtoms
    assert passOG


@pytest.mark.skipif(_IN_CI, reason="takes >10s, skipped in CI")
def test_tutorial_6_fepo(tmp_path, resource_path_root):
    """Tutorial 6 variant: FePO fragment (fepo.xyz) on Pd slab, object_align Fe."""
    testName = "tutorial_6_fepo"
    threshOG = 2.0
    [passNumAtoms, passOG] = hp.runtest_molecule_on_slab(
        tmp_path, resource_path_root, testName, threshOG,
        xyz_relative_paths={'-target_molecule': "../xyz_files/fepo.xyz"})
    assert passNumAtoms
    assert passOG
