import helperFuncs as hp


def test_tridentate_mer(tmpdir):
    testName = "tridentate_mer"
    threshMLBL = 0.1
    threshLG = 1.0
    threshOG = 1.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtest(
        tmpdir, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert hp.runtest_reportonly(tmpdir, testName)


def test_tridentate_fac(tmpdir):
    testName = "tridentate_fac"
    threshMLBL = 0.1
    threshLG = 1.0
    threshOG = 1.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtest(
        tmpdir, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert hp.runtest_reportonly(tmpdir, testName)
