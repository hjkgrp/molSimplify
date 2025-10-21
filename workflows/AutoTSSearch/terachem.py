import sys
import os
import subprocess
import time
import re
import shutil

sys.path.append(os.path.expanduser("~/molSimplify/molSimplify/Classes/"))
sys.path.append(os.path.expanduser("~/molSimplify/molSimplify/Scripts/"))
sys.path.append(os.path.expanduser("~/ts/core/"))

from mol3D import mol3D
from io_utils import *
from structure_gen import *

def write_terachem_opt(
    name_xyz,
    filename,
    charge,
    spinmult,
    basis_set="lacvps_ecp",
    functional="b3lyp",
    lvlshift="yes",
    lvla="0.25",
    lvlb="0.25",
    unrestricted=False,
    list_constraints=None
):
    """
    Write Terachem input file for geometry optimization.
    Parameters
    ----
    name_xyz : str
        Name of .xyz file to be optimized.
    charge: str
        Charge of the complex to be optimized
    spinmult: str
        Spin multiplicity of the complex to be optimized (2S+1).
    basis_set: str, optional
        Name of the basis set used. Default is lacvps_ecp.
    functional: str, optional
        Functional used. Default is b3lyp.
    lvlshift: str, optional
        Yes or no, whether the calculation will use levelshift or not. Default is yes.
    lvla: str, optional
        Value for levelshifta. Default is 0.25.
    lvlb: str, optional
        Value for levelshiftb. Default is 0.25
    unrestricted: bool, optional
        Whether the calculation is unrestricted or not. Default is False.
    list_constraints: list of int, optional
        List of atom indices to be frozen during optimization(0-indexed). Default is None.
    Return
    ----
    terachem_in : str
        text for terachem input file.
    """
    # Write terachem input file
    if float(spinmult) > 1:
        unrestricted_str = "u"
    elif unrestricted:
        unrestricted_str = "u"
    else:
        unrestricted_str = ""
    charge_str = str(charge)
    spinmult_str = str(spinmult)
    if list_constraints is None:
        constraints = ""
    else:
        constraints = "$constraints_freeze\n"
        for atom in list_constraints:
            constraints += f"xyz {atom+1}\n"
        constraints += "$end"
    terachem_in = f"""run minimize
coordinates {name_xyz}
basis {basis_set}
method {unrestricted_str}{functional}
charge {charge_str}
spinmult {spinmult_str}
levelshift {lvlshift}
levelshiftvala {lvla}
levelshiftvalb {lvlb}
scf diis+a
timings yes
dftd d3
new_minimizer yes
maxit 500
ml_prop yes
poptype mulliken
bon_order_list yes
scrdir ./{filename}
end
{constraints}
"""
    return terachem_in

def troubleshoot_terachem_opt(path_to_terachem_out, filename, input_xyz, cluster, unrestricted=False, list_constraints=None, custom_cluster_path=None):
    """
    Troubleshoot Terachem optimization output file.
    Parameters
    ----------
    path_to_terachem_out : str
        Path to the Terachem output file.
    filename : str
        Name of the Terachem input file used for the bond length scan.
    input_xyz : str
        Path to the .xyz file containing the initial geometry.
    unrestricted : bool, optional
        Whether the calculation is unrestricted or not. Default is False.
    list_constraints : list of int, optional
        List of atom indices to be frozen during optimization (0-indexed). Default is None.
    Returns
    -------
    bool, int
        True if troubleshooting was successful, False otherwise, and rerun number.
    """
    # Check for successful optimization
    job_status = findtext("Job finished", path_to_terachem_out, return_type='boolean')
    if job_status:
        print("Terachem optimization completed successfully.")
        return True, 0 
    if not job_status:
        if findtext("Job terminated", path_to_terachem_out, return_type='boolean'):
            if findtext("Incorrect molecular charge or spin multiplicity", path_to_terachem_out, return_type='boolean'):
                # Rerun the job with correct charge or spin multiplicity
                print("Job terminated due to incorrect molecular charge or spin multiplicity, rerunning with corrected values.")
                charges = findtext("charge", path_to_terachem_out, first_instance=True, return_type='full_line')
                spinmults = findtext("spinmult", path_to_terachem_out, first_instance=True, return_type='full_line')
                if charges and spinmults:
                    charge = float(charges[0].split()[-1])
                    spinmult = float(spinmults[0].split()[-1])+1
                    # Clean up previous files
                    file_list = [f"{filename}.in", f"{filename}_job", f"{filename}/"]
                    for file in file_list:
                        if os.path.isfile(file):
                            os.remove(file)
                        if os.path.isdir(file):
                            shutil.rmtree(file)
                    # Run Terachem optimization with correct charge and spin multiplicity
                    parameters = {
                        "name_xyz": input_xyz,
                        "charge": charge,
                        "spinmult": spinmult,
                        "list_constraints": list_constraints,
                        "unrestricted": unrestricted
                    }
                    terachem_opt_id = submit_job("terachem", "opt", parameters, filename, cluster=cluster, custom_script_path=custom_cluster_path)
                    wait_job(terachem_opt_id, cluster, "terachem", f"{filename}.out", wait_time=1, custom_script_path=custom_cluster_path)
                    return False, terachem_opt_id
        else:
            print("Terachem optimization did not finish successfully, terminating.")
            return False, 0

def check_terachem_structure(filename, terachem_input_geo):
    """
    Check if the Terachem optimization structure is correct.
    Parameters
    ----------
    filename : str
        Name of the Terachem input file used for optimization.
    terachem_input_geo : str
        Path to the Terachem input geometry file for structure comparison.
    Returns
    -------
    bool
        True if structure check passed, False otherwise.
    """
    check3D = mol3D()
    check3D.readfromxyz(filename, read_final_optim_step=True)
    if terachem_input_geo is not None:
        input_geo = mol3D()
        input_geo.readfromxyz(terachem_input_geo, read_final_optim_step=True)
    else:
        input_geo=None
        main_check,*_=check3D.Oct_inspection(init_mol=input_geo)
    if main_check == 1:
        print("Structure check passed.")
        return True
    else:
        print("Structure check failed.")
        return False

def check_terachem_opt(path_to_terachem_out, filename, input_xyz, cluster, structure_check, terachem_input_geo=None, atoms_w_unpaired_e=None, custom_cluster_path=None):
    """
    Check if Terachem optimization was successful.
    Parameters
    ----------
    path_to_terachem_out : str
        Path to the Terachem output file.
    filename : str
        Name of the Terachem input file used for optimization.
    input_xyz : str
        Path to the .xyz file containing the initial geometry.
    opt_filename : str
        Path to the .xyz file containing the final optimization step.
    structure_check : bool, optional
        If True, perform a structure check on the final optimization step.
        Default is False.
    terachem_input_geo : str, optional
        Path to the Terachem input geometry file for structure comparison.
        Default is None.
    atoms_w_unpaired_e : list, optional
        List of atom indices with unpaired electrons for spin distribution check.
        Default is None.
    Returns
    -------
    bool
        True if optimization was successful, False otherwise.
    """
    # Check for successful optimization
    job_status = findtext("Job finished", path_to_terachem_out, return_type='boolean')
    if not job_status:
        check, rerun_id = troubleshoot_terachem_opt(path_to_terachem_out, filename, input_xyz, cluster, custom_cluster_path=custom_cluster_path)
        while not check:
            print("Terachem optimization did not finish successfully, rerunning...")
            wait_job(rerun_id, cluster, "terachem", f"{filename}.out", custom_script_path=custom_cluster_path)
            check, rerun_id = troubleshoot_terachem_opt(path_to_terachem_out, filename, input_xyz, cluster, custom_cluster_path=custom_cluster_path)
            if check:
                print("Terachem optimization completed successfully.")
                break
            else:
                print("Terachem optimization failed again, rerunning...")
    # Structure check
    current_dir = os.getcwd()
    extract_final_opt_step(f"{current_dir}/{filename}/optim.xyz", f"{filename}_opt.xyz")
    time.sleep(10)  # Wait for the file to be written
    if structure_check:
        structure = check_terachem_structure(f"{current_dir}/{filename}_opt.xyz", terachem_input_geo)
        if not structure:
            print("Structure check failed, terminating.")
            sys.exit(1)    
    # Check if calculation was unrestricted
    spinmults = findtext("spinmult", path_to_terachem_out,first_instance=True, return_type='full_line')
    spinmult = re.search(r'\d+', spinmults[0])
    if spinmult:
        spinmult = int(spinmult.group())
    else:
        print("Spin multiplicity not found in Terachem output, terminating.")
        sys.exit(1)
    # Check mulliken spin distribution and spin deviation for unrestricted calculations
    if spinmult >1:
        # Check for spin contamination
        spin_contamination = findtext("SPIN S-SQUARED",path_to_terachem_out,last_instance=True, return_type='full_line')
        spin_check = re.search(r"SPIN S-SQUARED:\s*([0-9.]+)\s*\(exact:\s*([0-9.]+)\)", spin_contamination[0])
        if spin_check:
            spin_squared = float(spin_check.group(1))
            exact_spin_squared = float(spin_check.group(2))
            if abs(spin_squared - exact_spin_squared) > 1:
                print(f"Spin contamination detected: {spin_squared} (exact: {exact_spin_squared})")
                sys.exit(1)
            else:
                print(f"Spin contamination check passed: {spin_squared} (exact: {exact_spin_squared})")
        # Check for mulliken spin distribution
        if atoms_w_unpaired_e is not None:
            unpaired_electrons = spinmult - 1
            mullpop = findtext("Atom",f"{current_dir}/{filename}/mullpop", last_instance=True, return_type='line_no')
            if not mullpop:
                print("Mulliken population analysis not found, terminating.")
                sys.exit(1)
            with open(f"{current_dir}/{filename}/mullpop", 'r') as f:
                mullpop_lines = [line.split() for line in f.readlines()[mullpop[0]-1:]]
                densities = []
                for a in atoms_w_unpaired_e:
                    for line in mullpop_lines:
                        if line[0] == str(a+1):
                            densities.append(float(line[-1]))
                density = sum(abs(x) for x in densities)
                density_check = density - unpaired_electrons
                if abs(density_check) > 1:
                    print(f"Mulliken spin distribution check failed: {density_check} (expected: {unpaired_electrons})")
                    print(f"Spin density in atom indices {atoms_w_unpaired_e}: {densities}")
                    sys.exit(1)
                else:
                    print(f"Mulliken spin distribution check passed: {density_check} (expected: {unpaired_electrons})")
    else:
        print("Spin multiplicity is 1, no spin contamination or mulliken spin distribution check needed.")
    print("Terachem optimization completed successfully.")
    return True

def terachem_geo_optimize(name_xyz, filename, cluster, charge, spinmult, ind, options):
    """
    Optimize geometry using terachem, wait for completion, and check output.
    Parameters
    ----------
    name_xyz : str
        Name of the .xyz file containing the initial geometry.
    filename : str
        Name of the output file for the optimized geometry.
    cluster : str
        Type of cluster to submit the job to (e.g., 'slurm', 'sge', 'custom').
    charge : int
        Charge of the tmc.
    spinmult : int
        Spin multiplicity of the tmc (2S+1).
    ind : list of int
        Indices of the atoms with unpaired electrons in the tmc.
    options: dict
        Dictionary of optional parameters:
        {
            "specifications": (basis, functional, lvlshft, lvla, lvlb, unrestricted),
            "structure_check": bool,
            "list_constraints": ...,
            "mech_step": "oxo" or "hat",
            "custom_cluster_path": script path for custom cluster,
            "running_directory": where the job will be run (if not provided, current directory is used)
        }
    Returns
    -------
    None
    """
    base_dir = os.getcwd()
    if options and options.get("running_directory"):
        base_dir = options["running_directory"]
    specs = options.get("specifications", {}) if options else {}
    parameters = {
        "name_xyz": name_xyz,
        "charge": charge,
        "spinmult": spinmult,
        "basis_set": specs.get("basis", "lacvps_ecp"),
        "functional": specs.get("functional", "b3lyp"),
        "lvlshift": specs.get("lvlshft", 0),
        "lvla": specs.get("lvla", 0),
        "lvlb": specs.get("lvlb", 0),
        "unrestricted": specs.get("unrestricted", False),
        "list_constraints": options.get("list_constraints") if options else None
    }
    print("Custom parameters provided for Terachem optimization.")
    tera_opt_id = submit_job("terachem", "opt", parameters, filename, cluster, custom_script_path=options.get("custom_cluster_path"))
    # Wait for terachem job to finish
    wait_job(tera_opt_id, cluster, "terachem", f"{filename}.out", custom_script_path=options.get("custom_cluster_path"))
    # Check terachem output file and extract optimized geometry
    if options.get("mech_step") == "oxo":
        atoms_w_unpaired_e = [ind[0], ind[1]]
    elif options.get("mech_step") == "hat":
        atoms_w_unpaired_e = [ind[0], ind[1], ind[2], ind[3]]
    else:
        atoms_w_unpaired_e = ind
    bool = check_terachem_opt(
        f"{base_dir}/{filename}.out", 
        filename,
        input_xyz=f"{base_dir}/{name_xyz}.xyz",
        cluster=cluster,
        structure_check=options.get("structure_check"),
        terachem_input_geo=f"{base_dir}/{name_xyz}",
        atoms_w_unpaired_e=atoms_w_unpaired_e,
        custom_cluster_path=options.get("custom_cluster_path"),
    )
    while not bool:
        print("Terachem optimization failed")
    return True