import helperFuncs as hp


def test_tutorial_qm9_part_one(tmpdir, resource_path_root):
    testName = "tutorial_qm9_part_one"
    hp.runtest_num_atoms_in_xyz(tmpdir, resource_path_root, testName)
