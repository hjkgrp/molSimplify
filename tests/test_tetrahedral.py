import helperFuncs as hp


def test_example_1(tmpdir, resource_path_root):
    testName = "tetrahedral_1"
    threshMLBL = 0.1  # Change this value for your need
    threshLG = 0.1  # Change this value for your need
    threshOG = 2.0  # Change this value for you need
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtestNoFF(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report


def test_example_2(tmpdir, resource_path_root):
    testName = "tetrahedral_2"
    threshMLBL = 0.1  # Change this value for your need
    threshLG = 0.1  # Change this value for your need
    threshOG = 2.0  # Change this value for you need
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtestNoFF(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
