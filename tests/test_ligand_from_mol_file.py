import pytest
from molSimplify.Scripts.generator import startgen
from helperFuncs import working_directory, compareGeo, compare_report_new
import shutil


@pytest.mark.skip("Loading multidentate ligands from files is currently not supported")
def test_ligand_from_mol_file(tmpdir, resource_path_root):
    input_file = resource_path_root / "inputs" / "ligand_from_mol_file.in"
    shutil.copyfile(input_file, tmpdir / "ligand_from_mol_file.in")
    mol_file = resource_path_root / "inputs" / "pdp.mol"
    shutil.copyfile(mol_file, tmpdir / "pdp.mol")

    ref_xyz = resource_path_root / "refs" / "ligand_from_mol_file.xyz"
    ref_report = resource_path_root / "refs" / "ligand_from_mol_file.report"

    threshMLBL = 0.1
    threshLG = 0.1
    threshOG = 0.1

    with working_directory(tmpdir):
        startgen(['main.py', '-i', 'ligand_from_mol_file.in'], flag=False, gui=False)

        jobdir = tmpdir / 'Runs' / 'ligand_from_mol_file'
        output_xyz = str(jobdir / 'ligand_from_mol_file.xyz')
        output_report = str(jobdir / 'ligand_from_mol_file.report')

        passNumAtoms, passMLBL, passLG, passOG = compareGeo(
            output_xyz, ref_xyz, threshMLBL, threshLG, threshOG)
        assert passNumAtoms
        assert passMLBL
        assert passLG
        assert passOG
        pass_report = compare_report_new(output_report, ref_report)
        assert pass_report
