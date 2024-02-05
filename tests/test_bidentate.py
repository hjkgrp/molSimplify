import helperFuncs as hp


def test_bidentate(tmpdir, resource_path_root):
    # There are two versions of this test depending on the openbabel version.
    # This is necessary because openbabel changed the numbering of atoms for v3.
    try:
        # This is the recommended method to import openbabel for v3
        from openbabel import openbabel  # noqa: F401
        testName = "bidentate_v3"
    except ImportError:
        testName = "bidentate"
    threshMLBL = 0.1
    threshLG = 1.0
    threshOG = 1.5
    [passMultiFileCheck, pass_structures] = hp.runtestMulti(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    assert passMultiFileCheck
    for f, passNumAtoms, passMLBL, passLG, passOG, pass_report in pass_structures:
        print(f)
        assert passNumAtoms
        assert passMLBL
        assert passLG
        assert passOG
        assert pass_report
