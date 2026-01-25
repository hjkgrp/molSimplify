import helperFuncs as hp
import pytest


@pytest.mark.skip(reason="Randomly fails.")
def test_tutorial_10_part_two(tmp_path, resource_path_root):
    testName = "tutorial_10_part_two"
    threshMLBL = 0.1
    threshLG = 7.0
    threshOG = 7.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtest(
        tmp_path, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
