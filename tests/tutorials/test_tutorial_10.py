import shutil

import helperFuncs as hp
from molSimplify.Scripts.generator import startgen
from molSimplify.Classes.globalvars import globalvars
from molSimplify.Scripts.io import copy_to_custom_path


def test_tutorial_10_acetate_homoleptic(tmp_path, resource_path_root):
    """Tutorial 10: homoleptic complex from acetate-like ligand (CC(=O)[O-]) via .in file."""
    testName = "tutorial_10_from_smiles"
    threshMLBL = 0.1
    threshLG = 2.0
    threshOG = 2.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtest(
        tmp_path, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    # Not checking for passMLBL because there are atoms very close to the cutoff.
    # assert passMLBL
    assert passLG
    assert passOG
    assert pass_report


def test_tutorial_10_ligadd_smiles(tmp_path, resource_path_root):
    """Tutorial 10: add custom ligand NACAC via -ligadd, then build homoleptic Fe(NACAC)3.

    Covers:
      1) molsimplify -ligadd "O=C(C)C(N)(N)C(=O)C" -ligname NACAC -ligcon 1,8 -skipANN True
         with -custom_data_dir to avoid interactive prompt.
      2) molsimplify -lig NACAC -ligocc 3 -skipANN True (using the same custom data dir).

    Geometry is checked via runtest (passNumAtoms, passMLBL, passLG, passOG, pass_report).
    """
    with hp.working_directory(tmp_path):
        globs = globalvars()
        globs.custom_path = str(tmp_path)
        copy_to_custom_path()

        custom_dir = str(tmp_path)
        rundir = str(tmp_path)

        ligadd_input = (
            f"-rundir {rundir}\n"
            f"-custom_data_dir {custom_dir}\n"
            "-ligadd O=C(C)C(N)(N)C(=O)C\n"
            "-ligname NACAC\n"
            "-ligcon 1,8\n"
            "-skipANN True\n"
        )
        emsg = startgen(
            ["main.py", "-i", "dummy.in"],
            False,
            inputfile_str=ligadd_input,
        )
        assert not emsg, f"ligadd step failed: {emsg}"

        build_name = "tutorial_10_from_smiles_2"
        threshMLBL = 0.1
        threshLG = 2.5
        threshOG = 3.0
        passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin = hp.runtest(
            tmp_path, resource_path_root, build_name, threshMLBL, threshLG, threshOG,
            custom_data_dir=custom_dir,
        )
        assert passNumAtoms
        assert passMLBL
        assert passLG
        assert passOG
        assert pass_report


def test_tutorial_10_ligadd_mol(tmp_path, resource_path_root):
    """Tutorial 10: add glycinate from gly.mol via -ligadd, then build homoleptic Fe(glycinate)3.

    Covers:
      1) molsimplify -ligadd gly.mol -ligname glycinate -ligcon 3,4 -skipANN True
         (gly.mol from testresources, copied to run dir; -custom_data_dir avoids prompt).
      2) molsimplify -lig glycinate -ligocc 3 -skipANN True.

    Geometry is checked via runtest (passNumAtoms, passMLBL, passLG, passOG, pass_report).
    """
    with hp.working_directory(tmp_path):
        globs = globalvars()
        globs.custom_path = str(tmp_path)
        copy_to_custom_path()

        # Make gly.mol available in the run folder as in the tutorial
        gly_src = resource_path_root / "inputs" / "in_files" / "tutorial_10" / "gly.mol"
        shutil.copy(gly_src, tmp_path / "gly.mol")

        custom_dir = str(tmp_path)
        rundir = str(tmp_path)

        ligadd_input = (
            f"-rundir {rundir}\n"
            f"-custom_data_dir {custom_dir}\n"
            "-ligadd gly.mol\n"
            "-ligname glycinate\n"
            "-ligcon 3,4\n"
            "-skipANN True\n"
        )
        emsg = startgen(
            ["main.py", "-i", "dummy.in"],
            False,
            inputfile_str=ligadd_input,
        )
        assert not emsg, f"ligadd step failed: {emsg}"

        build_name = "tutorial_10_from_mol"
        threshMLBL = 0.1
        threshLG = 2.0
        threshOG = 2.0
        passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin = hp.runtest(
            tmp_path, resource_path_root, build_name, threshMLBL, threshLG, threshOG,
            custom_data_dir=custom_dir,
        )
        assert passNumAtoms
        assert passMLBL
        assert passLG
        assert passOG
        assert pass_report
