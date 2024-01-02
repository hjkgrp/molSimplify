import helperFuncs as hp


def test_tutorial_2(tmpdir, resource_path_root):
    testName = "tutorial_2"
    threshOG = 2.0
    [passNumAtoms, passOG] = hp.runtest_slab(
        tmpdir, resource_path_root, testName, threshOG,
        extra_files=['pd_test_tutorial_2.cif'])
    assert passNumAtoms
    assert passOG
