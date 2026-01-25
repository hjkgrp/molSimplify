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


def test_tutorial_7_decoration(tmp_path, resource_path_root):
    """Generate Fe + 4× pyridine + 2× chloride with Cl decoration at index 7; assess like runtest."""
    testName = "tutorial_7_decoration"
    threshMLBL = 0.1
    threshLG = 0.5
    threshOG = 1.0
    out = hp.runtest(
        tmp_path, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin = out
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin


def test_tutorial_7_decoration_multi(tmp_path, resource_path_root):
    """Generate Fe + 4× pyridine + 2× chloride with Cl at 7, CO at 9; assess like runtest."""
    testName = "tutorial_7_decoration_multi"
    threshMLBL = 0.1
    threshLG = 1.0   # looser (cf. tutorial_8) for CI cross-platform
    threshOG = 3.0   # looser (cf. tutorial_3, 8, 9, 10) for CI cross-platform
    out = hp.runtest(
        tmp_path, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin = out
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin


def test_tutorial_7_decoration_4lig(tmp_path, resource_path_root):
    """Generate Fe + pyridine×3 + chloride (ligocc 1 1 2 2), Cl at 7 and CO at 9 on 1st/2nd type; assess like runtest."""
    testName = "tutorial_7_decoration_4lig"
    threshMLBL = 0.1
    threshLG = 1.0   # looser (cf. tutorial_8) for CI cross-platform
    threshOG = 3.0   # looser for 4-lig CI variability (cf. test_example_7)
    out = hp.runtest(
        tmp_path, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin = out
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin
