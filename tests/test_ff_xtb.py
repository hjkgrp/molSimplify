import pytest
import shutil
import helperFuncs as hp


# Decorator to skip test if xtb is not installed
xtb_installed = pytest.mark.skipif(shutil.which('xtb') is None,
                                   reason='Could not find xtb installation')


@xtb_installed
def test_xtb_before(tmp_path, resource_path_root):
    testName = "xtb_H2O_before"
    threshMLBL = 0.01
    threshLG = 0.01
    threshOG = 2.0
    (passNumAtoms, passMLBL, passLG,
     passOG, pass_report, pass_qcin) = hp.runtest(
        tmp_path, resource_path_root, testName, threshMLBL, threshLG, threshOG, seed=31415)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin


@xtb_installed
def test_xtb_ANC_fail(tmp_path, resource_path_root):
    testName = "xtb_ANC_fail"
    threshMLBL = 0.01
    threshLG = 0.01
    threshOG = 0.01
    (passNumAtoms, passMLBL, passLG,
     passOG, pass_report, pass_qcin) = hp.runtest(
        tmp_path, resource_path_root, testName, threshMLBL, threshLG, threshOG, seed=31415)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin


@xtb_installed
def test_xtb_high_spin(tmp_path, resource_path_root):
    testName = "xtb_bipy_spin5"
    threshMLBL = 0.01
    threshLG = 0.05
    threshOG = 0.5
    (passNumAtoms, passMLBL, passLG,
     passOG, pass_report, pass_qcin) = hp.runtest(
        tmp_path, resource_path_root, testName, threshMLBL, threshLG, threshOG, seed=31415)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin


@xtb_installed
def test_xtb_final_opt(tmp_path, resource_path_root):
    testName = "xtb_final_opt"
    threshMLBL = 0.01
    threshLG = 0.01
    threshOG = 0.05
    (passNumAtoms, passMLBL, passLG,
     passOG, pass_report, pass_qcin) = hp.runtest(
        tmp_path, resource_path_root, testName, threshMLBL, threshLG, threshOG, seed=31415)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin


@xtb_installed
def test_gfnff(tmp_path, resource_path_root):
    testName = "gfnff_NH3_BA"
    threshMLBL = 0.01
    threshLG = 0.01
    threshOG = 2.0
    (passNumAtoms, passMLBL, passLG,
     passOG, pass_report, pass_qcin) = hp.runtest(
        tmp_path, resource_path_root, testName, threshMLBL, threshLG, threshOG, seed=31415)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin
