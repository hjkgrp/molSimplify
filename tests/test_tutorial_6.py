import helperFuncs as hp


def test_tutorial_6(tmpdir, resource_path_root):
    testName = "tutorial_6"
    threshOG = 2.0
    [passNumAtoms, passOG] = hp.runtest_molecule_on_slab(
        tmpdir, resource_path_root, testName, threshOG)
    assert passNumAtoms
    assert passOG
