import helperFuncs as hp


def test_example_1(tmpdir, resource_path_root):
    testName = "example_1"
    threshMLBL = 0.1
    threshLG = 1.0
    threshOG = 2.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtest(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert hp.runtest_reportonly(tmpdir, resource_path_root, testName)


def test_example_1_No_FF(tmpdir, resource_path_root):
    testName = "example_1"
    threshMLBL = 0.1
    threshLG = 1.0
    threshOG = 2.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtestNoFF(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
