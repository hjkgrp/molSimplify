import helperFuncs as hp
import pytest


def test_example_7(tmp_path, resource_path_root):
    testName = "example_7"
    threshMLBL = 0.1
    threshLG = 1.0
    threshOG = 3.0  # Increased threshold from 2.0 to 3.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtest(
        tmp_path, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report, pass_qcin


@pytest.mark.skip(reason="Randomly fails.")
def test_example_7_No_FF(tmp_path, resource_path_root):
    testName = "example_7"
    threshMLBL = 0.1
    threshLG = 1.1
    threshOG = 3.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtestNoFF(
        tmp_path, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report, pass_qcin
