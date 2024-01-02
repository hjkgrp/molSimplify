import pytest
import shutil
import helperFuncs as hp


# Decorator to skip test if xtb is not installed
xtb_installed = pytest.mark.skipif(shutil.which('xtb') is None,
                                   reason='Could not find xtb installation')


@xtb_installed
def test_xtb_before(tmpdir, resource_path_root):
    testName = "xtb_H2O_before"
    threshMLBL = 0.01
    threshLG = 0.01
    threshOG = 2.0
    (passNumAtoms, passMLBL, passLG,
     passOG, pass_report, pass_qcin) = hp.runtest(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG, seed=31415)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin


@pytest.mark.skip(reason="TODO: figure out why this test started failing suddenly")
@xtb_installed
def test_xtb_before_after(tmpdir, resource_path_root):
    testName = "xtb_imidazole_BA"
    threshMLBL = 0.01
    threshLG = 0.01
    threshOG = 0.05
    (passNumAtoms, passMLBL, passLG,
     passOG, pass_report, pass_qcin) = hp.runtest(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG, seed=31415)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin


@xtb_installed
def test_xtb_ANC_fail(tmpdir, resource_path_root):
    testName = "xtb_ANC_fail"
    threshMLBL = 0.01
    threshLG = 0.01
    threshOG = 0.01
    (passNumAtoms, passMLBL, passLG,
     passOG, pass_report, pass_qcin) = hp.runtest(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG, seed=31415)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin


@xtb_installed
def test_xtb_high_spin(tmpdir, resource_path_root):
    testName = "xtb_bipy_spin5"
    threshMLBL = 0.01
    threshLG = 0.05
    threshOG = 0.5
    (passNumAtoms, passMLBL, passLG,
     passOG, pass_report, pass_qcin) = hp.runtest(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG, seed=31415)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin


@xtb_installed
def test_xtb_final_opt(tmpdir, resource_path_root):
    testName = "xtb_final_opt"
    threshMLBL = 0.01
    threshLG = 0.01
    threshOG = 0.05
    (passNumAtoms, passMLBL, passLG,
     passOG, pass_report, pass_qcin) = hp.runtest(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG, seed=31415)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin


@xtb_installed
def test_gfnff(tmpdir, resource_path_root):
    testName = "gfnff_NH3_BA"
    threshMLBL = 0.01
    threshLG = 0.01
    threshOG = 2.0
    (passNumAtoms, passMLBL, passLG,
     passOG, pass_report, pass_qcin) = hp.runtest(
        tmpdir, resource_path_root, testName, threshMLBL, threshLG, threshOG, seed=31415)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
    assert pass_qcin
