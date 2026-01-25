import helperFuncs as hp


def test_tutorial_7(tmp_path, resource_path_root):
    """Generate Fe + 4× pyridine + 2× chloride for spin 1 and 5; assess like runtest."""
    testName = "tutorial_7"
    threshMLBL = 0.1
    threshLG = 0.5
    threshOG = 1.0
    expected_spins = ['1', '5']
    out = hp.runtest_multispin(
        tmp_path, resource_path_root, testName, expected_spins,
        threshMLBL, threshLG, threshOG)
    passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin = out
    assert passNumAtoms, "multispin run should produce two structures with same atom count"
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin
