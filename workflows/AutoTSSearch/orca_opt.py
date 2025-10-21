import sys
import os
import subprocess
import re

sys.path.append(os.path.expanduser("~/molSimplify/molSimplify/Classes/"))
sys.path.append(os.path.expanduser("~/molSimplify/molSimplify/Scripts/"))
sys.path.append(os.path.expanduser("~/ts/core/"))

from io_utils import *
from terachem import *
from structure_gen import *

def write_orca_opt(name_xyz, charge, spinmult, involved_atoms, calculation_type, unrestricted=False, list_constraints=None, rerun=False):
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
    involved_atoms: list
        List of atom indices involved in the transition state optimization.
    calculation_type : str
        Type of calculation to be performed ('TS', 'opt', or 'sp').
        'TS' for transition state optimization,
        'opt' for geometry optimization,
        'sp' for single point energy calculation.
    unrestricted: bool, optional
        If True, use unrestricted calculations, only relevant if spinmult = 1 and unrestricted calculation is desired.
        Default is False.
    list_constraints: list of int, optional
        List of atom indices to be frozen during optimization (0-indexed). Default is None.
    rerun: bool, optional
        If True, rerun the ORCA calculation from a previous SCF calculation. Default is False.
    Return
    ----
    terachem_in : str
        text for terachem input file.
    """
    if float(spinmult) >1:
        unrestricted_str="UKS"
    elif unrestricted:
        unrestricted_str="UKS"
    else:
        unrestricted_str=""
    if list_constraints is None:
        constraints = ""
    else:
        constraints = "Constraints\n"
        for atom in list_constraints:
            constraints += f"{{C {atom} C}}\n"
        constraints += "end"
    if rerun:
        print("Rerunning ORCA TS optimization from previous SCF calulation.")
        read_gbw = f"\n! moread\n%moinp {name_xyz}.gbw\nend\n"
    else:
        read_gbw = ""
    if calculation_type == 'TS':
        ts_string = f"""TS_Active_Atoms {"{"}{involved_atoms[0]} {involved_atoms[1]} {involved_atoms[2]}{"}"} end
TS_Active_Atoms_Factor 1.5
"""
        calc_type = "OptTS"
    else:
        ts_string = ""
    if calculation_type == 'sp':
        calc_type = ""
    elif calculation_type == 'opt':
        calc_type = "Opt"
    # Write orca input file
    orca_in = f"""!{unrestricted_str} B3LYP D3BJ def2-svp tightscf SlowConv {calc_type} Freq

%pal
nprocs 16
end
{read_gbw}
%scf
maxiter 5000
end

%geom
Recalc_Hess 20
Calc_Hess true
{ts_string}
{constraints}
end

*xyzfile {charge} {spinmult} {name_xyz}
"""
    return orca_in

def check_orca_TS_modes(path_to_orca_out,involved_atoms,scan_type='breaking'):
    """
    Check if ORCA transition state optimization was successful.
    Parameters
    ----------
    path_to_orca_out : str
        Path to the ORCA output file.
    involved_atoms : list
        List of atom indices involved in the transition state optimization.
        Contains 3 indices: [bonding atom in substrate, binding atom site in tmc/mof, substrate atom bound to bonding atom]
    scan_type : str, optional
        Type of scan, either 'breaking' or 'bonding'. Default is 'breaking'.
    Returns
    -------
    bool
        True if the transition state optimization was successful, False otherwise.
    """
    # TS modes extraction
    # Find the start and end of the Redundant Internal Coordinates section
    ric_start = findtext("Redundant Internal Coordinates", path_to_orca_out,last_instance=True, return_type='line_no')
    ric_end = findtext("Geometry step timings", path_to_orca_out,last_instance=True, return_type='line_no')
    # Extract and clean lines
    red_int_coords_lines_str = extract_lines(path_to_orca_out, ric_start[0]+7, ric_end[0]-3)
    red_int_coords_lines = red_int_coords_lines_str.splitlines()
    red_int_coords = []
    for line in red_int_coords_lines:
        cleaned_line = re.sub(r'\([^)]*\)','', line)
        cleaned_line_matrix = re.sub(r'\s+', ' ', cleaned_line).strip().split()
        red_int_coords.append(cleaned_line_matrix)
    # Collect TS modes (lines with 7 columns)
    ts_modes = [i for i in red_int_coords if len(i) == 7]
    line_ts_modes = []
    for mode in ts_modes:
        id = int(mode[0].replace('.',''))-1
        if 0 <= id < len(red_int_coords_lines):
            line_ts_modes.append(red_int_coords_lines[id])
        else:
            print(f"Warning: mode id {id} out of range for red_int_coords_lines (length {len(red_int_coords_lines)})")
    # Write to file
    print("Transition State Modes:")
    for l in line_ts_modes:
        print(' '.join(l))
    # TS mode check
    if not ts_modes:
        print("No transition state modes found.")
        return False
    numeric_ts_modes = []
    for i in ts_modes:
        try:
            float(i[6])
            numeric_ts_modes.append(i)
        except ValueError:
            continue
    if not numeric_ts_modes:
        print("No valid numeric transition state modes found.")
        return False
    # Check if the dominant mode is the one we expect
    max_mode_index = ts_modes.index(max(numeric_ts_modes, key=lambda x: float(x[6])))
    max_mode = line_ts_modes[max_mode_index]
    bonding_atom = involved_atoms[0]
    if scan_type == 'breaking':
        involved_atom = involved_atoms[2]
    elif scan_type == 'bonding':
        involved_atom = involved_atoms[1]
    if "B" not in max_mode or f"{bonding_atom}" not in max_mode or f"{involved_atom}" not in max_mode:
        print(f"Expected {scan_type} TS mode not dominant between {bonding_atom} and {involved_atom}.")
        print("Dominant mode:", max_mode)
        return False
    else:
        print(f"Expected {scan_type} TS mode is dominant between {bonding_atom} and {involved_atom}.")
        print("Dominant mode:", max_mode)
        return True

def check_orca_TS_vib_freqs(path_to_orca_out):
    """
    Check if ORCA transition state optimization has imaginary frequencies.
    Parameters
    ----------
    path_to_orca_out : str
        Path to the ORCA output file.
    Returns
    -------
    bool
        True if imaginary frequencies are found, False otherwise.
    """
    base_dir = os.getcwd()
    vib_freq_start = findtext("VIBRATIONAL FREQUENCIES", path_to_orca_out,last_instance=True, return_type='line_no',case_sensitive=True)
    vib_freq_end = findtext("NORMAL MODES", path_to_orca_out, last_instance=True, return_type='line_no', case_sensitive=True)
    vib_freq_lines = extract_lines(path_to_orca_out, vib_freq_start[0]+5, vib_freq_end[0]-4)
    vib_freq_split = vib_freq_lines.splitlines()
    vib_freqs = []
    freqs = []
    for line in vib_freq_split:
        if "***imaginary mode***" in line.lower():
            vib_freqs.append(line)
            idx = vib_freq_split.index(line)
            split = line.split()
            intensity = abs(float(split[1]))
            freqs.append([idx, intensity])
    print("Imaginary Frequencies:")
    if vib_freqs:
        if len(vib_freqs) > 1:
            print("Multiple imaginary frequencies found")
            print("This may indicate a problem with the transition state optimization.")
            dominant_freq = max(freqs, key=lambda x: abs(x[1]))
            print(f"Dominant imaginary frequency: {vib_freq_split[dominant_freq[0]]} with intensity {dominant_freq[1]}")
            cmd = f"/orcd/pool/004/hjkulik_shared/src/orca_6_0_1_linux_x86-64_shared_openmpi416/orca_pltvib {path_to_orca_out} {dominant_freq[0]}"
            subprocess.run(cmd, shell=True)
            return False
        for freq in vib_freqs:
            print(freq)
            cmd = f"/orcd/pool/004/hjkulik_shared/src/orca_6_0_1_linux_x86-64_shared_openmpi416/orca_pltvib {path_to_orca_out} {vib_freq_split.index(freq)}"
            subprocess.run(cmd, shell=True)
    else:
        print("No imaginary frequencies found.")
        print("Vibrational Frequencies:")
        print(vib_freq_lines)
        return False
    return True

def check_orca_gibbs_free_energy(path_to_orca_out):
    """
    Check if ORCA Gibbs free energy calculation was successful.
    Parameters
    ----------
    path_to_orca_out : str
        Path to the ORCA output file.
    Returns
    -------
    bool
        True if Gibbs free energy is found, False otherwise.
    """
    gibbs_line = findtext("Final Gibbs free energy", path_to_orca_out, last_instance=True, return_type='full_line', case_sensitive=True)
    if gibbs_line:
        gibbs_split = gibbs_line[0].split()
        gibbs_free_energy = float(gibbs_split[-2])*627.509  # Convert Hartree to kcal/mol
        return gibbs_free_energy
    return None

def check_orca_spin_contamination(path_to_orca_out, atoms_w_unpaired_electrons):
    """
    Check if ORCA spin contamination is present.
    Parameters
    ----------
    path_to_orca_out : str
        Path to the ORCA output file.
    Returns
    -------
    bool
        False if spin contamination is found, True otherwise.
    """
    spin_calculation = findtext("Deviation", path_to_orca_out, last_instance=True, return_type='boolean', case_sensitive=True)
    if not spin_calculation:
        print("Spin-restricted calculation.")
        return True
    spin_line = findtext("Deviation", path_to_orca_out, last_instance=True, return_type='full_line', case_sensitive=True)
    if spin_line:
        spin_split = spin_line[0].split()
        spin_deviation = float(spin_split[-1])
        if spin_deviation > 1.0:
            print(f"Spin contamination detected: {spin_deviation}")
            return False
    spin_population_start = findtext("MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS", path_to_orca_out, last_instance=True, return_type='line_no', case_sensitive=True)
    spin_population_end = findtext("Sum of atomic spin populations", path_to_orca_out, last_instance=True, return_type='line_no', case_sensitive=True)
    spin_population_lines = extract_lines(path_to_orca_out, spin_population_start[0]+2, spin_population_end[0])
    spin_population_split = spin_population_lines.splitlines()
    spin_populations = []
    densities = []
    for line in spin_population_split:
        if spin_population_split.index(line) < len(spin_population_split)-1:
            if line.strip():
                split_line = line.split()
                atom_index = int(split_line[0])
                spin_population = float(split_line[-1])
                spin_populations.append({"atom_idx": atom_index, "spin_population": spin_population})
    if atoms_w_unpaired_electrons is None:
        print("No atoms with unpaired electrons specified for Mulliken spin distribution check, ignoring check.")
    else:
        for atom in atoms_w_unpaired_electrons:
            atom_spin = next((item["spin_population"] for item in spin_populations if item["atom_idx"] == atom), None)
            densities.append(atom_spin)
        sum_densities = sum(densities) if densities else 0
        unpaired_electrons = spin_population_split[-1].split()[-1]
        density_check = sum_densities - unpaired_electrons
        if abs(density_check) > 1:
            print(f"Mulliken spin distribution check failed: {density_check} unpaired electrons in expected atoms (expected: {unpaired_electrons})")
            print(f"Spin density in atom indices {atoms_w_unpaired_electrons}: {densities}")
            return False
    return True

def check_orca_output(path_to_orca_out, calculation_type, atoms_w_unpaired_electrons=None, involved_atoms=None, scan_type=None):
    """
    Check if ORCA transition state optimization was successful.
    Parameters
    ----------
    path_to_orca_out : str
        Path to the ORCA output file.
    involved_atoms : list
        List of atom indices involved in the transition state optimization.
        Contains 3 indices: [bonding atom in substrate, binding atom site in tmc/mof, substrate atom bound to bonding atom]
    scan_type : str
        Type of scan, either 'breaking' or 'bonding'. Default is 'breaking'.
    calculation_type : str
        Type of calculation performed, either 'TS', 'opt', or 'sp'.
        'TS' for transition state optimization,
        'opt' for geometry optimization,
        'sp' for single point energy calculation.
    atoms_w_unpaired_electrons : list, optional
        List of atom indices with unpaired electrons for Mulliken spin distribution check.
    Returns
    -------
    bool
        True if the transition state optimization was successful, False otherwise.
    """
    # Check if the ORCA job finished successfully
    job_status = findtext("ORCA TERMINATED NORMALLY", path_to_orca_out, return_type='boolean')
    if not job_status:
        print("ORCA job did not finish successfully, terminating.")
    if calculation_type == 'TS':
        # TS modes check
        ts_modes_check = check_orca_TS_modes(path_to_orca_out, involved_atoms, scan_type=scan_type)
        if not ts_modes_check:
            print("Transition state modes check failed.")
        # Check for imaginary frequencies
        vib_freq_check = check_orca_TS_vib_freqs(path_to_orca_out)
        if not vib_freq_check:
            print("Imaginary frequencies check failed.")
    # Check spin contamination and mulliken spin distribution
    spin_check = check_orca_spin_contamination(path_to_orca_out, atoms_w_unpaired_electrons=atoms_w_unpaired_electrons)
    if not spin_check:
        print("Spin contamination or Mulliken spin distribution check failed.")
    # Extract Gibbs free energy
    gibbs_free_energy = check_orca_gibbs_free_energy(path_to_orca_out)
    if gibbs_free_energy:
        print(f"\nGibbs free energy: {gibbs_free_energy} kcal/mol")
    return True

def orca_transition_state_optimization(ts_guess_xyz, orca_filename, charge, spinmult, involved_atoms, cluster, options):
    """
    Run ORCA transition state optimization.
    Parameters
    ----------
    ts_guess_xyz : str
        Name of the .xyz file containing the transition state guess geometry.
    orca_ts_name : str
        Name of the ORCA job to be run.
    charge : int
        Charge of the complex to be optimized.
    spinmult : int
        Spin multiplicity of the complex to be optimized (2S+1).
    involved_atoms : list
        List of atom indices involved in the transition state optimization.
        Contains 3 indices: [bonding atom in substrate, binding atom site in tmc/mof, substrate atom bound to bonding atom]
    options : dict
        Dictionary of optional parameters:
        {
            "unrestricted": bool,
            "scan_type": "breaking" or "bonding",
            "custom_cluster_path": script path for custom cluster,
            "atoms_w_unpaired_electrons": list of int,
            "list_constraints": list of int,
            "wait_time": int,
        }
    wait_time : int, optional
        Time in minutes to wait between checks for job completion. Default is 3 minutes.
    list_constraints : list of int, optional
        List of atom indices to be frozen during optimization (0-indexed). Default is None.
    """
    # Run ORCA TS optimization
    parameters = {
        "name_xyz": ts_guess_xyz,
        "charge": charge,
        "spinmult": spinmult,
        "involved_atoms": involved_atoms,
        "unrestricted": options.get("unrestricted", None),
        "list_constraints": options.get("list_constraints", None),
    }
    orca_TS_id = submit_job("orca","opt", parameters, orca_filename, cluster, custom_script_path=options["custom_cluster_path"])
    # Wait for ORCA TS optimization to finish
    wait_job(orca_TS_id, cluster, "orca", f"{orca_filename}.out", wait_time=options.get("wait_time", 1), custom_script_path=options["custom_cluster_path"])
    # Check ORCA TS optimization output file and extract TS geometry and gibbs free energy
    base_dir = os.getcwd()
    check_orca_output(f"{base_dir}/{orca_filename}.out", calculation_type='TS', atoms_w_unpaired_electrons=options["atoms_w_unpaired_electrons"], involved_atoms=involved_atoms, scan_type=options.get("scan_type", 'breaking'))

def orca_geometry_optimization(ts_guess_xyz, orca_filename, charge, spinmult, cluster, options):
    """
    Run ORCA optimization.
    Parameters
    ----------
    ts_guess_xyz : str
        Name of the .xyz file containing the initial geometry.
    orca_ts_name : str
        Name of the ORCA job to be run.
    charge : int
        Charge of the complex to be optimized.
    spinmult : int
        Spin multiplicity of the complex to be optimized (2S+1).
    cluster : str
        Cluster to run the job on.
    options : dict
        Dictionary of optional parameters:
        {
            "unrestricted": bool,
            "custom_cluster_path": script path for custom cluster,
            "atoms_w_unpaired_electrons": list of int,
            "wait_time": int,
            "list_constraints": list of int,
        }
    """
    # Run ORCA optimization
    parameters = {
        "name_xyz": ts_guess_xyz,
        "charge": charge,
        "spinmult": spinmult,
        "unrestricted": options.get("unrestricted", None),
        "list_constraints": options.get("list_constraints", None),
    }
    orca_id = submit_job("orca","opt", parameters, orca_filename, cluster, custom_script_path=options.get("custom_cluster_path"))
    # Wait for ORCA optimization to finish
    wait_job(orca_id, cluster, "orca", f"{orca_filename}.out", wait_time=options.get("wait_time", 1), custom_script_path=options.get("custom_cluster_path"))
    # Check ORCA optimization output file and extract optimized geometry and gibbs free energy
    base_dir = os.getcwd()
    check_orca_output(f"{base_dir}/{orca_filename}.out", calculation_type='opt', atoms_w_unpaired_electrons=options.get("atoms_w_unpaired_electrons", None))

def orca_single_point_energy(ts_guess_xyz, orca_filename, charge, spinmult, cluster, options):
    """
    Run ORCA single point energy calculation.
    Parameters
    ----------
    ts_guess_xyz : str
        Name of the .xyz file containing the initial geometry.
    orca_ts_name : str
        Name of the ORCA job to be run.
    charge : int
        Charge of the complex to be optimized.
    spinmult : int
        Spin multiplicity of the complex to be optimized (2S+1).
    cluster : str
        Cluster to run the job on.
    options : dict
        Dictionary of optional parameters:
        {
            "unrestricted": bool,
            "custom_cluster_path": script path for custom cluster,
            "atoms_w_unpaired_electrons": list of int,
            "wait_time": int,
            "list_constraints": list of int,
        }
    """
    # Run ORCA single point energy calculation
    parameters = {
        "name_xyz": ts_guess_xyz,
        "charge": charge,
        "spinmult": spinmult,
        "list_constraints": options.get("list_constraints", None),
    }
    orca_id = submit_job("orca","sp", parameters, orca_filename, cluster, custom_script_path=options.get("custom_cluster_path"))
    # Wait for ORCA single point energy calculation to finish
    wait_job(orca_id, cluster, "orca", f"{orca_filename}.out", wait_time=options.get("wait_time", 1), custom_script_path=options.get("custom_cluster_path"))
    # Check ORCA single point energy calculation output file and extract energy
    base_dir = os.getcwd()
    check_orca_output(f"{base_dir}/{orca_filename}.out", calculation_type='sp', atoms_w_unpaired_electrons=options.get("atoms_w_unpaired_electrons", None))
