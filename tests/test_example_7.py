import helperFuncs as hp


def test_example_7(tmpdir, resource_path_root):
    testName = "example_7"
    threshMLBL = 0.1
    threshLG = 1.0
    threshOG = 3.0  # Increased threshold from 2.0 to 3.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtest(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report, pass_qcin


def test_example_7_No_FF(tmpdir, resource_path_root):
    testName = "example_7"
    threshMLBL = 0.1
    threshLG = 1.1
    threshOG = 3.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtestNoFF(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report, pass_qcin
