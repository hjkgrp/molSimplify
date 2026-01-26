import json
import os
import re
import random
import shutil
import numpy as np
from typing import List, Tuple
from molSimplify.Scripts.geometry import kabsch, distance
from molSimplify.Scripts.generator import startgen
from molSimplify.Classes.globalvars import (dict_oneempty_check_st,
                                            oneempty_angle_ref)
from molSimplify.Classes.mol3D import mol3D
from typing import Dict
from contextlib import contextmanager
from pathlib import Path
import posixpath


def is_number(s: str) -> bool:
    """
    Check whether the string is a integral/float/scientific.

    Parameters
    ----------
        s : str
            The string to be assessed.

    Returns
    -------
        is_num : bool
            Flag indicating whether s is a number.
    """
    try:
        float(s)
        is_num = True
    except ValueError:
        is_num = False
    return is_num


def get_parent_folder(test_name):
    """
    Determine the parent folder name based on the test name.

    Parameters
    ----------
        test_name : str
            The name of the test.
            E.g., "example_1".

    Returns
    -------
        f : str
            The name of the parent folder.
    """
    # Getting the name of the parent folder for reference results.
    # Parent folders are named in line with test names.
    possible_folders = [
    'tutorial',
    'example',
    'xtb',
    'molcas',
    'old_ann',
    'orca',
    'gfnff',
    'tetrahedral',
    ]

    job_name_l = test_name.lower()
    for f in possible_folders:
        if f in job_name_l:
            return f # parent folder

    # If get to this point in the code,
    # no parent folder was identified.
    raise Exception('No parent folder identified.')


def get_in_files_subfolder(name: str) -> str:
    """
    Return the in_files subfolder for a given test name (e.g. tutorial_5, example_1_noff).
    Used to organize inputs/in_files into tutorial_2/, tutorial_5/, example_1/, etc.
    """
    stem = name.split(".", 1)[0] if "." in name else name
    lower = stem.lower()
    if lower.startswith("tutorial_"):
        m = re.match(r"tutorial_(\d+)(?:_|$)", lower)
        if m:
            return "tutorial_" + m.group(1)
        return "tutorial"
    if lower.startswith("example_"):
        parts = lower.split("_", 2)
        if len(parts) >= 2:
            return "example_" + parts[1]
        return "example"
    return get_parent_folder(stem)


def get_ref_subfolder(parent_folder: str, name: str) -> str:
    """
    Return the ref subfolder under refs/<parent_folder>/ for a given test name.
    For tutorial/example this is e.g. tutorial_5, example_1; for others it is ''.
    """
    stem = name.split(".", 1)[0] if "." in name else name
    if parent_folder == "tutorial":
        return get_in_files_subfolder(stem)
    if parent_folder == "example":
        return get_in_files_subfolder(stem)
    return ""


@contextmanager
def working_directory(path: Path):
    prev_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev_cwd)


def fuzzy_equal(x1, x2, thresh: float) -> bool:
    return np.fabs(float(x1) - float(x2)) < thresh


def fuzzy_compare_xyz(xyz1, xyz2, thresh: float) -> bool:
    """
    Compare that two geometries are roughly equal.

    Parameters
    ----------
        xyz1 : str
            Path to the first xyz file.
        xyz2 : str
            Path to the second xyz file.
        thresh : float
            The RMSD tolerance used for comparison.

    Returns
    -------
        fuzzyEqual : bool
            Verdict on whether the two geometries are roughly equal.
    """
    fuzzyEqual = False
    mol1 = mol3D()
    mol1.readfromxyz(xyz1)
    mol2 = mol3D()
    mol2.readfromxyz(xyz2)
    mol1, U, d0, d1 = kabsch(mol1, mol2)
    rmsd12 = mol1.rmsd(mol2)
    print(f'rmsd is {rmsd12:.2f}')
    if rmsd12 < thresh:
        fuzzyEqual = True
    return fuzzyEqual


def getAllLigands(xyz, transition_metals_only=True):
    """
    Get all the ligands of a transition metal complex.

    Parameters
    ----------
        xyz : str
            Path to the xyz file.
            Assumed to be a transition metal complex.
        transition_metals_only : bool
            Whether to only consider transition metals as metals.

    Returns
    -------
        ligands : list of mol3D
            mol3D objects of all ligands.
    """
    mymol3d = mol3D()
    mymol3d.readfromxyz(xyz)
    # Assumes only one metal in the molecule,
    # so [0] is used.
    mm = mymol3d.findMetal(transition_metals_only=transition_metals_only)[0]
    mbonded = mymol3d.getBondedAtoms(mm)
    ligands = []  # mol3D objects of ligands.
    ligAtoms = [] # Atom indices identified to be part of a ligand.

    # Get the 1st atom of one ligand.
    for iatom in mbonded:
        if iatom not in ligAtoms:
            lig = [iatom]
            oldlig = []

            # Keep branching out by getting bonded atoms
            # until ligand does not grow any more.
            while len(lig) > len(oldlig):
                # Make a copy of lig.
                oldlig = lig[:]
                for i in oldlig:
                    lbonded = mymol3d.getBondedAtoms(i)
                    for j in lbonded:
                        if (j != mm) and (j not in lig):
                            lig.append(j)

            newlig = mol3D()
            for i in lig:
                newlig.addAtom(mymol3d.atoms[i])
                ligAtoms.append(i)
            ligands.append(newlig)

    print(f"Ligand analysis of xyz file: {xyz}")
    print(f"There are {len(ligands)} ligand(s) bonded with metal center\
            {mm} in the complex")
    for i in range(0, len(ligands)):
        print(f"Number of atoms in ligand # {i} : {ligands[i].natoms}")
    return ligands


def getMetalLigBondLength(mymol3d: mol3D, transition_metals_only=True) -> List[float]:
    """
    Get all the metal-ligand bond lengths of a transition metal complex.

    Parameters
    ----------
        mymol3d : mol3D
            The molecule to be analyzed.
            Assumed to be a transition metal complex.
        transition_metals_only : bool
            Whether to only consider transition metals as metals.

    Returns
    -------
        blength : list of float
            The bond lengths between the metal and the ligands.
    """
    # Assumes only one metal in the molecule,
    # so [0] is used.
    mm = mymol3d.findMetal(transition_metals_only=transition_metals_only)[0]
    bonded = mymol3d.getBondedAtoms(mm)
    blength = []
    for i in bonded:
        blength.append(
            distance(mymol3d.atoms[mm].coords(), mymol3d.atoms[i].coords()))
    return blength


def compareNumAtoms(xyz1, xyz2) -> bool:
    """
    Compare number of atoms of two geometries.

    Parameters
    ----------
        xyz1 : str
            Path to the first xyz file.
        xyz2 : str
            Path to the second xyz file.

    Returns
    -------
        passNumAtoms : bool
            Whether the number of atoms is the same
            between the two structures.
    """
    print("Checking total number of atoms.")
    mol1 = mol3D()
    mol1.readfromxyz(xyz1)
    mol2 = mol3D()
    mol2.readfromxyz(xyz2)
    # Compare number of atoms
    passNumAtoms = (mol1.natoms == mol2.natoms)
    print(f"Pass total number of atoms check: {passNumAtoms}")
    return passNumAtoms


def compareMLBL(xyz1, xyz2, thresh: float, transition_metals_only=True) -> bool:
    """
    Compare metal-ligand bond lengths between two transition metal complexes.

    Parameters
    ----------
        xyz1 : str
            Path to the first xyz file.
            Assumed to be a transition metal complex.
        xyz2 : str
            Path to the second xyz file.
            Assumed to be a transition metal complex.
        thresh : float
            The tolerance used for bond length comparison.
        transition_metals_only : bool
            Whether to only consider transition metals as metals.

    Returns
    -------
        passMLBL : bool
            Verdict on whether M-L bond lengths between
            the two geometries are equal.
    """
    print("Checking metal-ligand bond length.")
    mol1 = mol3D()
    mol1.readfromxyz(xyz1)
    mol2 = mol3D()
    mol2.readfromxyz(xyz2)
    bl1 = getMetalLigBondLength(mol1, transition_metals_only=transition_metals_only)
    bl2 = getMetalLigBondLength(mol2, transition_metals_only=transition_metals_only)
    passMLBL = True

    if len(bl1) != len(bl2):
        print("Error! Number of metal-ligand bonds is different")
        passMLBL = False
    else:
        # Check each metal-ligand bond length.
        # Failure on any leads to overall test failure.
        for i in range(0, len(bl1)):
            if not fuzzy_equal(bl1[i], bl2[i], thresh):
                print(f"Error! Metal-ligand bond length mismatch for bond # {i}")
                passMLBL = False

    print(f"Pass metal-ligand bond length check: {passMLBL}")
    print(f"Threshold for bond length difference: {thresh}")

    return passMLBL


def compareLG(xyz1, xyz2, thresh: float, transition_metals_only=True) -> bool:
    """
    Compare ligand geometries between two transition metal complexes.

    Parameters
    ----------
        xyz1 : str
            Path to the first xyz file.
            Assumed to be a transition metal complex.
        xyz2 : str
            Path to the second xyz file.
            Assumed to be a transition metal complex.
        thresh : float
            The RMSD tolerance used for ligand comparison.
        transition_metals_only : bool
            Whether to only consider transition metals as metals.

    Returns
    -------
        passLG : bool
            Verdict on whether ligand geometries between
            the two complexes are equal.
    """
    passLG = True
    ligs1 = getAllLigands(xyz1, transition_metals_only=transition_metals_only)
    ligs2 = getAllLigands(xyz2, transition_metals_only=transition_metals_only)
    if len(ligs1) != len(ligs2):
        passLG = False
        return passLG
    for i in range(0, len(ligs1)):  # Iterate over the ligands.
        print(f"Checking geometry for ligand # {i}")
        ligs1[i], U, d0, d1 = kabsch(ligs1[i], ligs2[i])
        rmsd12 = ligs1[i].rmsd(ligs2[i])
        print(f'rmsd is {rmsd12:.2f}')
        if rmsd12 > thresh:
            passLG = False
            return passLG
    print(f"Pass ligand geometry check: {passLG}")
    print(f"Threshold for ligand geometry RMSD difference: {thresh}")
    return passLG


def compareOG(xyz1, xyz2, thresh: float) -> bool:
    """
    Comparison of overall geometries of two structures.

    Parameters
    ----------
        xyz1 : str
            Path to the first xyz file.
        xyz2 : str
            Path to the second xyz file.
        thresh : float
            The RMSD tolerance used for overall geometry comparison.

    Returns
    -------
        passOG : bool
            Verdict on whether the two geometries are roughly equal overall.
    """
    print("Checking the overall geometry.")
    passOG = fuzzy_compare_xyz(xyz1, xyz2, thresh)
    print("Pass overall geometry check: ", passOG)
    print("Threshold for overall geometry check: ", thresh)
    return passOG


def compareGeo(xyz1, xyz2, threshMLBL, threshLG, threshOG, slab=False, transition_metals_only=True):
    """
    Thorough comparison of two geometries, considering number of atoms and overall geometries.
    For non-slab geometries, also considering metal-ligand bond lengths and ligand geometries.

    Parameters
    ----------
        xyz1 : str
            Path to the first xyz file.
        xyz2 : str
            Path to the second xyz file.
        threshMLBL : float
            The tolerance used for bond length comparison.
        threshLG : float
            The RMSD tolerance used for ligand comparison.
        threshOG : float
            The RMSD tolerance used for overall geometry comparison.
        slab : bool
            Flag indicating whether the geometries are slabs.
            Reduces the scope fo the checks.
        transition_metals_only : bool
            Whether to only consider transition metals as metals.

    Returns
    -------
        passNumAtoms : bool
            Whether the number of atoms is the same
            between the two structures.
        passMLBL : bool
            Verdict on whether M-L bond lengths between
            the two geometries are equal.
        passLG : bool
            Verdict on whether ligand geometries between
            the two complexes are equal.
        passOG : bool
            Verdict on whether the two geometries are roughly equal overall.
    """
    # Compare number of atoms.
    passNumAtoms = compareNumAtoms(xyz1, xyz2)
    # Compare metal-ligand bond length.
    if not slab:
        passMLBL = compareMLBL(xyz1, xyz2, threshMLBL, transition_metals_only=transition_metals_only)
        # Compare single ligand geometry.
        passLG = compareLG(xyz1, xyz2, threshLG, transition_metals_only=transition_metals_only)
    # Compare gross match of overall complex.
    passOG = compareOG(xyz1, xyz2, threshOG)
    # FF free test
    # ANN set bond length test
    # covalent radii test
    if not slab:
        return passNumAtoms, passMLBL, passLG, passOG
    else:
        return passNumAtoms, passOG


def comparedict(ref, gen, thresh):
    """
    Compares the two dictionaries given as inputs
    to see if they are equivalent.

    Parameters
    ----------
        ref : dict
            The reference dictionary.
        gen : dict
            The newly generated dictionary.
        thresh : float
            The threshold for float comparison,
            for numerical dictionary items.

    Returns
    -------
        passComp : bool
            Whether the dictionaries are found to be equivalent.
    """
    passComp = True
    if not set(ref.keys()) <= set(gen.keys()):
        raise KeyError("Keys in the dictionary are not equivalent.")
    for key in ref:
        if is_number(ref[key]):
            valref, valgen = float(ref[key]), float(gen[key])
            if not abs(valref - valgen) < thresh:
                passComp = False
        else:
            # Items for this key are strings, not numbers.
            valref, valgen = str(ref[key]), str(gen[key])
            if not valgen == valref:
                passComp = False
    return passComp


def jobname(infile: str) -> str:
    name = os.path.basename(infile)
    name = name.replace(".in", "")
    return name


def jobdir(infile):
    name = jobname(infile)
    homedir = os.getcwd()
    mydir = homedir + '/Runs/' + name
    return mydir


def parse4test(infile, tmp_path: Path, extra_args: Dict[str, str] = {},
               skip_jobdir: bool = False) -> str:
    """
    Parse the in file and rewrite it,
    taking into account extra_args.

    Parameters
    ----------
        infile : str
            The path to the in file.
        tmp_path : pathlib.PosixPath
            Pre-defined pytest fixture. Temporary folder path to run the test.
        extra_args : dict
            Extra arguments to be written to the new job in file.
        skip_jobdir : bool
            If True, do not append -jobdir or -name (e.g. for multispin runs
            that need spin subdirs S1, S5, ... under rundir).

    Returns
    -------
        newname : str
            The path to the new job in file.
        jobdir : str
            The job directory.
    """
    name = jobname(infile)
    f = posixpath.join(tmp_path, os.path.basename(infile))
    newname = str(tmp_path) + "/" + os.path.basename(infile)
    print(newname)
    print('&&&&&&&&&')
    with open(infile, 'r') as f_in:
        data = f_in.readlines()
    newdata = ""
    for line in data:
        if line.split()[0] in extra_args.keys():
            newdata += f'{line.split()[0]} \
            {os.path.dirname(infile)}/{extra_args[line.split()[0]]}\n'
            continue
        if not (("-jobdir" in line) or ("-name" in line)):
            newdata += line
        # Check if we need to parse the dir of smi file.
        if ("-lig " in line) and (".smi" in line):
            smi = line.strip('\n').split()[1]
            abs_smi = os.path.dirname(infile) + '/' + smi
            newdata += "-lig " + abs_smi + "\n"
    newdata += f"-rundir {tmp_path}\n"
    if not skip_jobdir:
        newdata += "-jobdir " + name + "\n"
        newdata += "-name " + name + "\n"
    print('=====')
    print(newdata)
    with open(newname, 'w') as fi:
        fi.write(newdata)
    print(f"Input file parsed for test is located: {newname}")
    jobdir = str(tmp_path) if skip_jobdir else str(tmp_path / name)
    return newname, jobdir


def parse4testNoFF(infile, tmp_path: Path) -> str:
    """
    Parse the in file and rewrite it.
    Similar to parse4test, but with
    ffoption set to no.

    Parameters
    ----------
        infile : str
            The path to the in file.
        tmp_path : pathlib.PosixPath
            Pre-defined pytest fixture. Temporary folder path to run the test.

    Returns
    -------
        newname : str
            The path to the new job in file.
        jobdir : str
            The job directory.
    """
    name = jobname(infile)
    newinfile = str(tmp_path / (name + "_noff.in"))
    shutil.copyfile(infile, newinfile)
    newname, jobdir = parse4test(newinfile, tmp_path, extra_args={"-ffoption": "N"})
    return newname, jobdir


def report_to_dict(lines):
    """
    Create a dictionary from comma-separated files.

    Parameters
    ----------
        lines : list of str
            Contents of a file, obtained with readlines().

    Returns
    -------
        d : dict
            Dictionary reflecting the contents of lines.
    """
    d = dict()
    for line in lines:
        key, val = line.strip().split(',')[0:2]
        try:
            d[key] = float(val.strip('[]'))
        except ValueError:
            d[key] = str(val.strip('[]'))
    # Extra step for ANN_bond list:
    if 'ANN_bondl' in d.keys():
        d['ANN_bondl'] = [float(i.strip('[]')) for i in d['ANN_bondl'].split()]
    return d


def compare_report_new(report1, report2):
    """
    Compare the reports, split key and values, do
    fuzzy comparison on the values.

    Parameters
    ----------
        report1 : str
            Path to first report file.
        report2 : str
            Path to second report file.

    Returns
    -------
        are_equal : bool
            Verdict on whether the reports are equal.
    """
    with open(report1, 'r') as f_in:
        data1 = f_in.readlines()
    with open(report2, 'r') as f_in:
        data2 = f_in.readlines()

    if data1 and data2:
        are_equal = True
        dict1 = report_to_dict(data1)
        dict2 = report_to_dict(data2)
    else:
        are_equal = False
        print('File not found.')
        if not data1:
            print(f'Missing: {report1}')
        if not data2:
            print(f'Missing: {report2}')

    if are_equal:
        for k in dict1.keys():
            if are_equal:
                val1 = dict1[k]
                if k not in dict2.keys():
                    are_equal = False
                    print("Report compare failed for ", report1, report2)
                    print(f"keys {k} not present in {report2}")
                else:
                    val2 = dict2[k]

                    if not k == "ANN_bondl":
                        # See whether the values are numbers or text.
                        if is_number(val1) and is_number(val2):
                            are_equal = fuzzy_equal(val1, val2, 1e-4)
                        else:
                            are_equal = (val1 == val2)
                        if not are_equal:
                            print("Report compare failed for ",
                                  report1, report2)
                            print("Values don't match for key", k)
                            print([val1, val2])
                    else:
                        # Loop over ANN bonds?
                        # See whether the values are numbers or text.
                        for v1, v2 in zip(val1, val2):
                            are_equal = fuzzy_equal(v1, v2, 1e-4)
                        if not are_equal:
                            print("Report compare failed for ",
                                  report1, report2)
                            print("Values don't match for key", k)
                            print([val1, val2])
            else:
                break

    return are_equal


def compare_qc_input(inp, inp_ref):
    """
    Compare two quantum chemistry code input files
    to determine if they are equal.

    Parameters
    ----------
        inp : str
            The path to the quantum chemistry (qc) code input file.
        inp_ref : pathlib.PosixPath
            The path to the reference qc code input file.

    Returns
    -------
        passQcInputCheck : bool
            Verdict on whether qc input files are equal.
    """
    passQcInputCheck = True
    if not os.path.exists(inp_ref):
        return passQcInputCheck
    elif os.path.exists(inp_ref) and (not os.path.exists(inp)):
        passQcInputCheck = False
        print(f"{inp} not found.")
        return passQcInputCheck

    with open(inp, 'r') as f_in:
        data1 = f_in.read()
    with open(inp_ref, 'r') as f_in:
        data_ref = f_in.read()
    if len(data1) != len(data_ref):
        passQcInputCheck = False
        return passQcInputCheck
    for i in range(0, len(data1)):
        if data1[i] != data_ref[i]:
            passQcInputCheck = False
            break
    return passQcInputCheck


def runtest(tmp_path, resource_path_root, name, threshMLBL, threshLG, threshOG, seed=31415,
            custom_data_dir=None):
    """
    Performs test for specified test name.

    Parameters
    ----------
        tmp_path : pathlib.PosixPath
            Pre-defined pytest fixture. Temporary folder path to run the test.
        resource_path_root : pathlib.PosixPath
            Variable from pytest-resource-path.
            Points to molSimplify/tests/testresources.
        name : str
            The name of the test.
        threshMLBL : float
            The tolerance used for bond length comparison.
        threshLG : float
            The RMSD tolerance used for ligand comparison.
        threshOG : float
            The RMSD tolerance used for overall geometry comparison.
        seed : int
            The random seed.
        custom_data_dir : str or pathlib.Path, optional
            If set, appends -custom_data_dir to the run input so the job uses
            this path (e.g. when the test adds ligands to a custom dir first).

    Returns
    -------
        passNumAtoms : bool
            Whether the number of atoms is the same
            between the output and reference structures.
        passMLBL : bool
            Verdict on whether M-L bond lengths between
            the output and reference structures are equal.
        passLG : bool
            Verdict on whether ligand geometries between
            the output and reference structures are equal.
        passOG : bool
            Verdict on whether the geometries of the
            output and reference are roughly equal overall.
        pass_report : bool
            Verdict on whether the output and reference
            reports are equal.
        pass_qcin : bool
            Verdict on whether the output and reference
            qc input files are equal.
    """
    # Set seeds to eliminate randomness from test results.
    random.seed(seed)
    np.random.seed(seed)
    in_sub = get_in_files_subfolder(name)
    infile = resource_path_root / "inputs" / "in_files" / in_sub / f"{name}.in"
    newinfile, myjobdir = parse4test(infile, tmp_path)
    if custom_data_dir is not None:
        with open(newinfile, 'a') as f:
            f.write(f"-custom_data_dir {custom_data_dir}\n")
    args = ['main.py', '-i', newinfile]
    with working_directory(tmp_path):
        startgen(args, False, False)
    output_xyz = myjobdir + '/' + name + '.xyz'
    output_report = myjobdir + '/' + name + '.report'
    output_qcin = myjobdir + '/terachem_input'
    with open(newinfile, 'r') as f_in:
        molsim_data = f_in.read()
    if 'orca' in molsim_data.lower():
        output_qcin = myjobdir + '/orca.in'

    if 'molcas' in molsim_data.lower():
        output_qcin = myjobdir + '/molcas.input'

    # qcgen writes name.in when -name is set (parse4test sets it); fallback so QC comparison works
    if (
        output_qcin == myjobdir + '/terachem_input'
        and not os.path.exists(output_qcin)
    ):
        alt_qcin = myjobdir + '/' + name + '.in'
        if os.path.exists(alt_qcin):
            output_qcin = alt_qcin

    parent_folder = get_parent_folder(name)
    ref_sub = get_ref_subfolder(parent_folder, name)
    ref_base = resource_path_root / "refs" / parent_folder
    ref_root = (ref_base / ref_sub) if ref_sub else ref_base

    def _ref_path(fname):
        p = ref_root / fname
        return p if p.exists() else ref_base / fname

    ref_xyz = _ref_path(f"{name}.xyz")
    ref_report = _ref_path(f"{name}.report")
    ref_qcin = _ref_path(f"{name}.qcin")

    print("Test input file: ", newinfile)
    print("Test output files are generated in ", myjobdir)
    print("Output xyz file: ", output_xyz)
    pass_xyz = compareGeo(output_xyz, ref_xyz, threshMLBL, threshLG, threshOG)
    [passNumAtoms, passMLBL, passLG, passOG] = pass_xyz
    pass_report = compare_report_new(output_report, ref_report)
    print("Reference xyz file: ", ref_xyz)
    print("Test report file: ", output_report)
    print("Reference report file: ", ref_report)
    print("Reference xyz status: ", pass_xyz)
    print("Reference report status: ", pass_report)
    pass_qcin = compare_qc_input(output_qcin, ref_qcin)
    print("Reference qc input file: ", ref_qcin)
    print("Test qc input file:", output_qcin)
    print("Qc input status:", pass_qcin)
    return passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin


def runtest_multispin(
    tmp_path: Path,
    resource_path_root: Path,
    name: str,
    expected_spins: List[str],
    threshMLBL: float,
    threshLG: float,
    threshOG: float,
) -> Tuple[bool, bool, bool, bool, bool, bool]:
    """
    Run molSimplify with multiple spin states and assess outputs like runtest.

    Runs structure generation from the input file {name}.in (with -spin listing
    several multiplicities, e.g. 1 5). The jobdir then contains one subdir per
    spin (S1, S5, ...), each with generated .xyz and .report files. Each spin's
    output is compared to references in resource_path_root / "refs" /
    parent_folder, using the same geometry and report checks as runtest. Pass
    flags are combined across all spins (all must pass).

    Parameters
    ----------
        tmp_path : pathlib.PosixPath
            Pre-defined pytest fixture. Temporary folder path to run the test.
        resource_path_root : pathlib.PosixPath
            Variable from pytest-resource-path. Points to
            molSimplify/tests/testresources.
        name : str
            Test name; input file is inputs/in_files/{name}.in, refs live under
            refs / get_parent_folder(name).
        expected_spins : list of str
            Spin multiplicities that were requested (e.g. ['1', '5']). Used to
            locate subdirs S1, S5, ... under the jobdir.
        threshMLBL : float
            Tolerance for metal–ligand bond length comparison.
        threshLG : float
            RMSD tolerance for ligand geometry comparison.
        threshOG : float
            RMSD tolerance for overall geometry comparison.

    Returns
    -------
        passNumAtoms : bool
            True if each spin's structure has the same number of atoms as its
            reference.
        passMLBL : bool
            True if each spin's M–L bond lengths match the reference within
            threshMLBL.
        passLG : bool
            True if each spin's ligand geometries match the references within
            threshLG.
        passOG : bool
            True if each spin's overall geometry matches the reference within
            threshOG.
        pass_report : bool
            True if each spin's report matches its reference.
        pass_qcin : bool
            True if each spin's QC input file matches its reference.
    """
    in_sub = get_in_files_subfolder(name)
    infile = resource_path_root / "inputs" / "in_files" / in_sub / f"{name}.in"
    newinfile, _ = parse4test(infile, tmp_path, skip_jobdir=True)
    args = ['main.py', '-i', newinfile]
    with working_directory(tmp_path):
        startgen(args, False, False)

    parent_folder = get_parent_folder(name)
    ref_sub = get_ref_subfolder(parent_folder, name)
    ref_base = resource_path_root / "refs" / parent_folder
    ref_root = (ref_base / ref_sub) if ref_sub else ref_base

    def _ref_path(fname):
        p = ref_root / fname
        return p if p.exists() else ref_base / fname

    search_root = Path(tmp_path)

    for i, sp in enumerate(expected_spins):
        spin_subdir = 'S' + sp
        # xyz may be in S1/.../file.xyz or S1/file.xyz (conf subdirs)
        out_xyz = next(p for p in search_root.rglob('*.xyz') if spin_subdir in p.parts)
        out_report = out_xyz.parent / (out_xyz.stem + '.report')
        out_qcin = out_xyz.parent / 'terachem_input'
        ref_xyz = _ref_path(f"{name}_s{sp}.xyz")
        ref_report = _ref_path(f"{name}_s{sp}.report")
        ref_qcin = _ref_path(f"{name}_s{sp}.qcin")
        pass_xyz = compareGeo(str(out_xyz), str(ref_xyz), threshMLBL, threshLG, threshOG)
        pr = compare_report_new(str(out_report), str(ref_report))
        pq = compare_qc_input(str(out_qcin), str(ref_qcin))
        if i == 0:
            passNumAtoms, passMLBL, passLG, passOG = pass_xyz
            pass_report = pr
            pass_qcin = pq
        else:
            passNumAtoms = passNumAtoms and pass_xyz[0]
            passMLBL = passMLBL and pass_xyz[1]
            passLG = passLG and pass_xyz[2]
            passOG = passOG and pass_xyz[3]
            pass_report = pass_report and pr
            pass_qcin = pass_qcin and pq
    return passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin


def runtest_slab(tmp_path, resource_path_root, name, threshOG, extra_files=None):
    """
    Performs test for slab builder.

    Parameters
    ----------
        tmp_path : pathlib.PosixPath
            Pre-defined pytest fixture. Temporary folder path to run the test.
        resource_path_root : pathlib.PosixPath
            Variable from pytest-resource-path.
            Points to molSimplify/tests/testresources.
        name : str
            The name of the test.
        threshOG : float
            Tolerance for RMSD comparison of overall geometries.
        extra_files : list of str
            Extra files to be copied to the test directory.

    Returns
    -------
        passNumAtoms : bool
            Whether the number of atoms is the same
            between the output and reference structures.
        passOG : bool
            Verdict on whether the geometries of the
            output and reference are roughly equal overall.
    """
    in_sub = get_in_files_subfolder(name)
    infile = resource_path_root / "inputs" / "in_files" / in_sub / f"{name}.in"
    newinfile, _ = parse4test(infile, tmp_path)
    if extra_files is not None:
        for file_name in extra_files:
            file_path = resource_path_root / "inputs" / "cif_files" / f"{file_name}"
            shutil.copyfile(file_path, tmp_path / file_name)
    args = ['main.py', '-i', newinfile]
    with working_directory(tmp_path):
        startgen(args, False, False)
    output_xyz = tmp_path / 'slab' / 'super332.xyz'
    parent_folder = get_parent_folder(name)
    ref_base = resource_path_root / "refs" / parent_folder
    ref_sub = get_ref_subfolder(parent_folder, name)
    ref_root = (ref_base / ref_sub) if ref_sub else ref_base
    _ref_path = lambda fname: ref_root / fname if (ref_root / fname).exists() else ref_base / fname
    ref_xyz = _ref_path(f"{name}.xyz")
    print(f"Output xyz file: {output_xyz}")
    pass_xyz = compareGeo(output_xyz, ref_xyz, threshMLBL=0, threshLG=0,
                          threshOG=threshOG, slab=True)
    [passNumAtoms, passOG] = pass_xyz
    return passNumAtoms, passOG


def runtest_molecule_on_slab(tmp_path, resource_path_root, name, threshOG, extra_files=None,
                             xyz_relative_paths=None):
    """
    Performs test for slab builder with a molecule adsorbed on a slab.

    Parameters
    ----------
        tmp_path : pathlib.PosixPath
            Pre-defined pytest fixture. Temporary folder path to run the test.
        resource_path_root : pathlib.PosixPath
            Variable from pytest-resource-path.
            Points to molSimplify/tests/testresources.
        name : str
            The name of the test.
        threshOG : float
            Tolerance for RMSD comparison of overall geometries.
        extra_files : list of str
            Extra files to be copied to the test directory.
        xyz_relative_paths : dict, optional
            Override paths for -unit_cell and -target_molecule, relative to
            the in file's directory (e.g. "../../xyz_files/slab.xyz" from
            in_files/tutorial_6/).
            Default: slab.xyz and co.xyz.

    Returns
    -------
        passNumAtoms : bool
            Whether the number of atoms is the same
            between the output and reference structures.
        passOG : bool
            Verdict on whether the geometries of the
            output and reference are roughly equal overall.
    """
    default_paths = {
        '-unit_cell': "../../xyz_files/slab.xyz",
        '-target_molecule': "../../xyz_files/co.xyz",
    }
    extra_args = default_paths if xyz_relative_paths is None else {
        **default_paths, **xyz_relative_paths
    }
    in_sub = get_in_files_subfolder(name)
    infile = resource_path_root / "inputs" / "in_files" / in_sub / f"{name}.in"
    newinfile, _ = parse4test(infile, tmp_path, extra_args=extra_args)
    if extra_files is not None:
        for file_name in extra_files:
            file_path = resource_path_root / "inputs" / f"{file_name}"
            shutil.copyfile(file_path, tmp_path / file_name)
    args = ["main.py", "-i", newinfile]
    with working_directory(tmp_path):
        startgen(args, False, False)
    output_xyz = tmp_path / "loaded_slab" / "loaded.xyz"
    parent_folder = get_parent_folder(name)
    ref_base = resource_path_root / "refs" / parent_folder
    ref_sub = get_ref_subfolder(parent_folder, name)
    ref_root = (ref_base / ref_sub) if ref_sub else ref_base
    _ref_path = lambda fname: ref_root / fname if (ref_root / fname).exists() else ref_base / fname
    ref_xyz = _ref_path(f"{name}.xyz")
    print(f"Output xyz file: {output_xyz}")
    pass_xyz = compareGeo(output_xyz, ref_xyz, threshMLBL=0, threshLG=0,
                          threshOG=threshOG, slab=True)
    [passNumAtoms, passOG] = pass_xyz
    return passNumAtoms, passOG


def runtestgeo(tmp_path, resource_path_root, name, thresh, deleteH=True, geo_type="oct"):
    """
    Performs test comparing dictionary of measurements of geometry
    generated by IsOct or IsStructure against a reference.

    Parameters
    ----------
        tmp_path : pathlib.PosixPath
            Pre-defined pytest fixture. Temporary folder path to run the test.
        resource_path_root : pathlib.PosixPath
            Variable from pytest-resource-path.
            Points to molSimplify/tests/testresources.
        name : str
            The name of the test.
        thresh : float
            The threshold for float comparison,
            for numerical dictionary items.
        deleteH : bool, optional
            Flag to delete Hs in ligand comparison. Default is True.
        geo_type : str
            The geometry type.

    Returns
    -------
        passGeo : bool
            Whether the dictionaries of the reference and the output
            are found to be equivalent.
    """
    initgeo = resource_path_root / "inputs" / "geocheck" / name / "init.xyz"
    optgeo = resource_path_root / "inputs" / "geocheck" / name / "opt.xyz"
    refjson = resource_path_root / "refs" / "geocheck" / name / "ref.json"
    mymol = mol3D()
    mymol.readfromxyz(optgeo)
    init_mol = mol3D()
    init_mol.readfromxyz(initgeo)
    with working_directory(tmp_path):
        if geo_type == "oct":
            _, _, dict_struct_info = mymol.IsOct(
                init_mol=init_mol, debug=False, flag_deleteH=deleteH)
        elif geo_type == "one_empty":
            _, _, dict_struct_info = mymol.IsStructure(
                init_mol=init_mol, dict_check=dict_oneempty_check_st,
                angle_ref=oneempty_angle_ref, num_coord=5, debug=False,
                flag_deleteH=deleteH)
        else:
            raise ValueError(f"Invalid geo_type {geo_type}")
    with open(refjson, "r") as fo:
        dict_ref = json.load(fo)
    print(f"ref: {dict_ref}")
    print(f"now: {dict_struct_info}")
    passGeo = comparedict(dict_ref, dict_struct_info, thresh)
    return passGeo


def runtestNoFF(tmp_path, resource_path_root, name, threshMLBL, threshLG, threshOG):
    """
    Performs test for specified test name, with no force field applied.

    Parameters
    ----------
        tmp_path : pathlib.PosixPath
            Pre-defined pytest fixture. Temporary folder path to run the test.
        resource_path_root : pathlib.PosixPath
            Variable from pytest-resource-path.
            Points to molSimplify/tests/testresources.
        name : str
            The name of the test.
        threshMLBL : float
            The tolerance used for bond length comparison.
        threshLG : float
            The RMSD tolerance used for ligand comparison.
        threshOG : float
            The RMSD tolerance used for overall geometry comparison.

    Returns
    -------
        passNumAtoms : bool
            Whether the number of atoms is the same
            between the output and reference structures.
        passMLBL : bool
            Verdict on whether M-L bond lengths between
            the output and reference structures are equal.
        passLG : bool
            Verdict on whether ligand geometries between
            the output and reference structures are equal.
        passOG : bool
            Verdict on whether the geometries of the
            output and reference are roughly equal overall.
        pass_report : bool
            Verdict on whether the output and reference
            reports are equal.
        pass_qcin : bool
            Verdict on whether the output and reference
            qc input files are equal.
    """
    in_sub = get_in_files_subfolder(name)
    infile = resource_path_root / "inputs" / "in_files" / in_sub / f"{name}.in"
    newinfile, myjobdir = parse4testNoFF(infile, tmp_path)
    [passNumAtoms, passMLBL, passLG, passOG, pass_report,
     pass_qcin] = [True, True, True, True, True, True]
    if newinfile != "":
        newname = jobname(newinfile)
        args = ['main.py', '-i', newinfile]
        with working_directory(tmp_path):
            startgen(args, False, False)
        output_xyz = myjobdir + '/' + newname + '.xyz'
        output_report = myjobdir + '/' + newname + '.report'
        with open(newinfile, 'r') as f_in:
            molsim_data = f_in.read()
        output_qcin = myjobdir + '/terachem_input'
        if 'orca' in molsim_data.lower():
            output_qcin = myjobdir + '/orca.in'
        if 'molcas' in molsim_data.lower():
            output_qcin = myjobdir + '/molcas.input'
        parent_folder = get_parent_folder(name)
        ref_base = resource_path_root / "refs" / parent_folder
        ref_sub = get_ref_subfolder(parent_folder, name)
        ref_root = (ref_base / ref_sub) if ref_sub else ref_base
        _ref_path = lambda fname: ref_root / fname if (ref_root / fname).exists() else ref_base / fname
        ref_xyz = _ref_path(f"{newname}.xyz")
        ref_report = _ref_path(f"{newname}.report")
        ref_qcin = _ref_path(f"{name}.qcin")
        print("Test input file: ", newinfile)
        print("Test output files are generated in ", myjobdir)
        print("Output xyz file: ", output_xyz)
        pass_xyz = compareGeo(output_xyz, ref_xyz,
                              threshMLBL, threshLG, threshOG)
        [passNumAtoms, passMLBL, passLG, passOG] = pass_xyz
        pass_report = compare_report_new(output_report, ref_report)
        print("Reference xyz file: ", ref_xyz)
        print("Test report file: ", output_report)
        print("Reference report file: ", ref_report)
        print("Reference xyz status: ", pass_xyz)
        print("Reference report status: ", pass_report)
        pass_qcin = compare_qc_input(output_qcin, ref_qcin)
        print("Reference qc input file: ", ref_qcin)
        print("Test qc input file: ", output_qcin)
        print("Qc input status: ", pass_qcin)
    return passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin


def runtest_reportonly(tmp_path, resource_path_root, name, seed=31415):
    """
    Performs test for specified test name, only looking at report files.

    Parameters
    ----------
        tmp_path : pathlib.PosixPath
            Pre-defined pytest fixture. Temporary folder path to run the test.
        resource_path_root : pathlib.PosixPath
            Variable from pytest-resource-path.
            Points to molSimplify/tests/testresources.
        name : str
            The name of the test.
        seed : int
            The random seed.

    Returns
    -------
        pass_report : bool
            Verdict on whether the output and reference
            reports are equal.
    """
    # Set seeds to eliminate randomness from test results.
    random.seed(seed)
    np.random.seed(seed)
    in_sub = get_in_files_subfolder(name)
    infile = resource_path_root / "inputs" / "in_files" / in_sub / f"{name}.in"
    # Copy the input file to the temporary folder.
    shutil.copy(infile, tmp_path/f'{name}_reportonly.in')
    # Add the report only flag.
    with open(tmp_path/f'{name}_reportonly.in', 'a') as f:
        f.write('-reportonly True\n')
    newinfile, myjobdir = parse4test(tmp_path/f'{name}_reportonly.in', tmp_path)
    args = ['main.py', '-i', newinfile]
    with open(newinfile, 'r') as f:
        print(f.readlines())
    with working_directory(tmp_path):
        startgen(args, False, False)
    output_report = myjobdir + '/' + name + '_reportonly.report'
    parent_folder = get_parent_folder(name)
    ref_base = resource_path_root / "refs" / parent_folder
    ref_sub = get_ref_subfolder(parent_folder, name)
    ref_root = (ref_base / ref_sub) if ref_sub else ref_base
    _ref_path = lambda fname: ref_root / fname if (ref_root / fname).exists() else ref_base / fname
    ref_report = _ref_path(f"{name}.report")
    # Copy the reference report to the temporary folder.
    shutil.copy(ref_report, tmp_path/f'{name}_ref.report')
    with open(tmp_path/f'{name}_ref.report', 'r') as f:
        lines = f.read()
    lines = lines.replace('Min_dist (A), 1000', 'Min_dist (A), graph')
    with open(tmp_path/f'{name}_ref.report', 'w') as f:
        f.write(lines)

    print("Test input file: ", newinfile)
    print("Test output files are generated in ", myjobdir)
    pass_report = compare_report_new(output_report, tmp_path/f'{name}_ref.report')
    print("Test report file: ", output_report)
    print("Reference report file: ", ref_report)
    print("Reference report status: ", pass_report)
    return pass_report
