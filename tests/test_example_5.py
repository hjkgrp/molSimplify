import helperFuncs as hp


def test_example_5(tmpdir, resource_path_root):
    # There are two versions of this test depending on the openbabel version.
    # This is necessary because openbabel changed the numbering of atoms for v3.
    try:
        # This is the recommended method to import openbabel for v3
        from openbabel import openbabel  # noqa: F401
        testName = "example_5_v3"
    except ImportError:
        testName = "example_5"
    threshMLBL = 0.1
    threshLG = 0.5
    threshOG = 1.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtest(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report, pass_qcin


def test_example_5_No_FF(tmpdir, resource_path_root):
    # There are two versions of this test depending on the openbabel version.
    # This is necessary because openbabel changed the numbering of atoms for v3.
    try:
        # This is the recommended method to import openbabel for v3
        from openbabel import openbabel  # noqa: F401
        testName = "example_5_v3"
    except ImportError:
        testName = "example_5"
    threshMLBL = 0.1
    threshLG = 0.5
    threshOG = 1.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtestNoFF(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report, pass_qcin
