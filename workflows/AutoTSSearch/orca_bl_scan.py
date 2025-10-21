import sys
import os
import subprocess
import time
import re
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
import shutil

sys.path.append(os.path.expanduser("~/molSimplify/molSimplify/Classes/"))
sys.path.append(os.path.expanduser("~/molSimplify/molSimplify/Scripts/"))

from mol3D import mol3D
from geometry import *
from io_utils import *
from terachem import *
from structure_gen import *

def write_orca_bl_scan(name_xyz,charge,spinmult,mech_step,step_no,step_change=None,scan_type='breaking', unrestricted=False, custom=False, list_constraints=None):
    """
    Write ORCA input file for bond length scan in a mechanistic step.
    Parameters
    ----------
    name_xyz : str
        Name of the .xyz file to be scanned.
    charge : int
        Charge of the complex to be scanned.
    spinmult : int
        Spin multiplicity of the complex to be scanned (2S+1).
    mech_step : str
        Mechanistic step to be scanned, either 'oxo' or 'hydroxo'.
    step_no : int
        Number of steps for the bond length scan.
    step_change : float
        Step change for the bond length scan.
    scan_type : str, optional
        Type of scan, either 'breaking' or 'formation'. Default is 'breaking'.
    unrestricted : bool, optional
        If True, use unrestricted calculations (UKS). Default is False.
    custom : float or None, optional
        If provided, use this value for the final bond length instead of the default values.
    list_constraints : list of int, optional
        List of atom indices to be frozen during optimization (0-indexed). Default is None.
    Returns
    -------
    orca_bl_scan : str
        Text for ORCA input file for bond length scan.
    """
    if float(spinmult) >1:
        unrestricted_str="UKS"
    elif unrestricted:
        unrestricted_str="UKS"
    else:
        unrestricted_str=""
    if not custom:
        final_bl_oxo_breaking = 2.3  # Default final O-N bond length for oxo-N₂ bond breaking
        final_bl_oxo_formation = 1.5  # Default final metal-O bond length for oxo formation
        final_bl_hat_breaking = 2.3  # Default final O-H bond length for hydroxo-Methane bond breaking
        final_bl_hat_formation = 0.85  # Default final O-H bond length
    elif custom:
        final_bl_oxo_breaking = custom
        final_bl_oxo_formation = custom
        final_bl_hat_breaking = custom
        final_bl_hat_formation = custom
    # Determine the starting and final bond lengths based on the mechanistic step
    if mech_step == "oxo":
        o_idx, n_idx, metal_idx, no_distance, metal_distance = oxo_n2o_bond_breaking_distance(name_xyz)
        if scan_type == 'breaking':
            start_bl = no_distance
            final_bl = final_bl_oxo_breaking # Final O-N bond length for oxo-N₂ bond breaking
            atom1_to_scan = o_idx
            atom2_to_scan = n_idx
        elif scan_type == 'formation':
            start_bl = metal_distance
            final_bl = final_bl_oxo_formation # Final metal-O bond length for oxo formation
            atom1_to_scan = metal_idx
            atom2_to_scan = o_idx
        else:
            raise ValueError("scan_type must be 'breaking' or 'formation'")
    elif mech_step == "hat":
        o_idx, h_idx, c_idx, oh_distance, ch_distance= hydroxo_ch4_bond_breaking_distance(name_xyz)
        if scan_type == 'breaking':
            start_bl = ch_distance
            final_bl = final_bl_hat_breaking
            atom1_to_scan = c_idx
            atom2_to_scan = h_idx
        elif scan_type == 'formation':
            start_bl = oh_distance
            final_bl = final_bl_hat_formation
            atom1_to_scan = h_idx
            atom2_to_scan = o_idx
        else:
            raise ValueError("scan_type must be 'breaking' or 'formation'")
    else:
        raise ValueError("mech_step must be 'oxo' or 'hydroxo' will implement 'methanol' later")
    if step_change:
        num_steps = abs(round((float(start_bl) - float(final_bl))/float(step_change)))
    else:
        num_steps = step_no
    if list_constraints is None:
        constraints = ""
    else:
        constraints = "Constraints\n"
        for atom in list_constraints:
            constraints += f"{{C {atom} C}}\n"
        constraints += "end"

    orca_bl_scan = f"""!{unrestricted_str} B3LYP D3BJ def2-svp tightscf SlowConv Opt

%pal
nprocs 16
end

%scf
maxiter 5000
end

%geom
maxIter 5000
scan B {atom1_to_scan} {atom2_to_scan} = {start_bl}, {final_bl}, {num_steps} end
{constraints}
end

*xyzfile {charge} {spinmult} {name_xyz}
"""
    return orca_bl_scan

def plot_energy_scan(energies, bond_lengths, ids, save_path, fit=True,spline_s=0):
    """
    Plot energy scan results with iteration index and bond length axes.
    Optionally fit a polynomial or spline to the data.

    Parameters
    ----------
    energies : list of float
        Energy values (y-axis).
    bond_lengths : list of float
        Bond length values for each point.
    ids : list of int
        Iteration indices (x-axis).
    save_path : str
        Path to save the plot image.
    fit : str or None, optional
        'poly' for polynomial fit, 'spline' for spline fit, None for no fit.
    degree : int, optional
        Degree of the polynomial fit (default: 2).
    spline_s : float, optional
        Smoothing factor for spline fit (default: 0 for interpolation).
    """
    fig, ax1 = plt.subplots()
    ax1.scatter(ids, energies, label='Energy Results', color='blue')

    if fit:
        x = np.array(ids)
        y = np.array(energies)
        spline = UnivariateSpline(x, y, s=spline_s)
        x_fit = np.linspace(min(x), max(x), 200)
        y_fit = spline(x_fit)
        result = minimize_scalar(lambda x: -spline(x), bounds=(min(ids), max(ids)), method='bounded')
        max_x = result.x
        max_y = spline(max_x)
        bond_max = np.interp(max_x, ids, bond_lengths)
        max_fitted = [max_x, max_y, bond_max]
        ax1.plot(x_fit, y_fit, label='Spline fit', color='green')
        ax1.legend()
        local_maxima_indices = []
        # Check first point
        if len(energies) > 1 and energies[0] > energies[1]:
            local_maxima_indices.append(0)
        # Check interior points
        for i in range(1, len(energies)-1):
            if energies[i] > energies[i-1] and energies[i] > energies[i+1]:
                local_maxima_indices.append(i)
        # Check last point
        if len(energies) > 1 and energies[-1] > energies[-2]:
            local_maxima_indices.append(len(energies)-1)
        local_maxima = [(ids[i], energies[i], bond_lengths[i]) for i in local_maxima_indices]
    ax1.set_xlabel('Iteration Index')
    ax1.set_ylabel('Relative Energy (kcal/mol)')
    # Add secondary x-axis for bond lengths
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ids)
    ax2.set_xticklabels([f"{bl:.2f}" for bl in bond_lengths], rotation=45)
    ax2.set_xlabel('Bond Length (Å)')
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return max_fitted, local_maxima if fit else None

def identify_bl_scan_max(max_fitted,local_maxima, energy_results):
    """
    Identify the maximum energy guess from the ORCA bond length scan output.
    Parameters
    ----------
    max_fitted : list(float)
        List containing the maximum fitted values [max_x, max_y, bond_max] - output from the fitting process.
    energy_results : list of list
        List of energy results from the ORCA bond length scan output, where each entry is a list containing [bond_length, energy].
    """
    # Check if the ORCA job finished successfully
    sorted_energy_results = sorted(energy_results, key=lambda x :abs(float(x[1]) - max_fitted[1]))[:2]
    closest = sorted_energy_results[0]
    closest_idx = energy_results.index(closest)+1
    print(f"Closest ORCA output file: orca_bl_scan.{str(closest_idx).zfill(3)}.xyz")
    # Check if max fitted is at either ends
    print("max_fitted[0] type:", type(max_fitted[0]))
    if float(max_fitted[0]) == 1:
        if len(local_maxima) == 1:
            print("Maximum fitted is at the start, rerunning scan before this point.")
            return [False, "max at start", closest_idx, closest[0]]
        else:
            if len(local_maxima) == 2:
                if local_maxima[1][0] == len(energy_results):
                    print("Maximum fitted is at the start, and another local maxima was found at the end.")
                    print("Rerunning scan before this point.")
                    return [False, "max at start", closest_idx, closest[0]]
                print("Maximum fitted is at the start, but another local maxima was found, ignoring boundary maxima.")
                local_maxima = [lm for lm in local_maxima if lm[0] != 1]
                print(f"Local maxima after removing boundary maxima: {local_maxima}")
                max_fitted = local_maxima[0]
            if len(local_maxima) > 3:
                if local_maxima[2][0] == len(energy_results):
                    print("Maximum fitted is at the start, and another local maxima was found at the end.")
                    print("Ignoring boundary maxima.")
                    local_maxima = [lm for lm in local_maxima if lm[0] != 1 and lm[0] != len(energy_results)]
                    print(f"Local maxima after removing boundary maxima: {local_maxima}")
                    max_fitted = local_maxima[0]
                print("Maximum fitted is at the start, but other local maxima were found, ignoring boundary maxima.")
                local_maxima = [lm for lm in local_maxima if lm[0] != 1]
                print(f"Local maxima after removing boundary maxima: {local_maxima}")
                return [False, "multiple maxima", closest_idx, closest[0], local_maxima]
    elif float(max_fitted[0]) == float(len(energy_results)):
        if len(local_maxima) == 1:
            print("Maximum fitted is at the end, rerunning scan after this point.")
            return [False, "max at end", closest_idx, closest[0]]
        else:
            if len(local_maxima) == 2:
                if local_maxima[0][0] == 1:
                    print("Maximum fitted is at the end, and another local maxima was found at the start.")
                    print("Rerunning scan after this point.")
                    return [False, "max at end", closest_idx, closest[0]]
                print("Maximum fitted is at the end, but another local maxima was found, ignoring boundary maxima.")
                local_maxima = [lm for lm in local_maxima if lm[0] != len(energy_results)]
                print(f"Local maxima after removing boundary maxima: {local_maxima}")
                max_fitted = local_maxima[0]
            if len(local_maxima) > 3:
                if local_maxima[2][0] == len(energy_results):
                    print("Maximum fitted is at the end, and another local maxima was found at the start.")
                    print("Ignoring boundary maxima.")
                    local_maxima = [lm for lm in local_maxima if lm[0] != 1 and lm[0] != len(energy_results)]
                    print(f"Local maxima after removing boundary maxima: {local_maxima}")
                    max_fitted = local_maxima[0]
                print("Maximum fitted is at the end, but other local maxima were found, ignoring boundary maxima.")
                local_maxima = [lm for lm in local_maxima if lm[0] != 1]
                print(f"Local maxima after removing boundary maxima: {local_maxima}")
                return [False, "multiple maxima", closest_idx, closest[0], local_maxima]
    # Check if max fitted is close enough to the closest structure
    if abs(max_fitted[2]-float(closest[0])) < 0.1:
        print("Found close structure for transition state guess.")
        return [True, "scan successful", closest_idx, float(closest[0])]
    else: 
        print("No close structure found for transition state guess, finding two closest structures.")
        # Find the two closest structures
        if len(sorted_energy_results) > 1:
            second_closest = sorted_energy_results[1]
            second_closest_idx = energy_results.index(second_closest)+1
            if abs(closest_idx-second_closest_idx) == 1:
                print(f"Closest structures are adjacent with bond lengths {closest[0]} Å and {second_closest[0]} Å, using them for rerun.")
                # Define guesses for rerun
                if closest_idx == 1:
                    second_guess = second_closest_idx + 2
                    first_guess = closest_idx
                elif closest_idx == len(energy_results):
                    second_guess = second_closest_idx - 2
                    first_guess = closest_idx
                else:
                    if second_closest_idx == 1:
                        second_guess = closest_idx + 2
                        first_guess = second_closest_idx
                    elif second_closest_idx == len(energy_results):
                        second_guess = closest_idx - 2
                        first_guess = second_closest_idx
                first_guess_bl = energy_results[first_guess - 1][0]
                second_guess_bl = energy_results[second_guess - 1][0]
                print(f"Using {str(first_guess).zfill(3)} and {str(second_guess).zfill(3)} as guesses for rerun.")
                result = [False, "max energy not close enough", [first_guess, first_guess_bl], [second_guess, second_guess_bl]]
                return result
            else:
                print("Closest structures are not adjacent")
                if closest_idx == 1:
                    second_guess = closest_idx + 3
                    first_guess = closest_idx
                elif closest_idx == len(energy_results):
                    second_guess = closest_idx - 3
                    first_guess = closest_idx
                else:
                    first_guess = closest_idx - 2
                    second_guess = closest_idx + 2
                first_guess_bl = energy_results[first_guess - 1][0]
                second_guess_bl = energy_results[second_guess - 1][0]
                print(f"Using {str(first_guess).zfill(3)} and {str(second_guess).zfill(3)} as guesses for rerun.")
                return [False, "max energy not close enough", [first_guess, first_guess_bl], [second_guess, second_guess_bl]]
def check_orca_bl_scan(path_to_orca_out, orca_in_filename):
    """" Check if ORCA bond length scan was successful.
    Parameters
    ----------
    path_to_orca_out : str
        Path to the ORCA output file.
    orca_in_filename : str
        ORCA input file NAME (not path) for the bond length scan.
    Returns
    -------
    bool
        True if the bond length scan was successful, False otherwise.
    """""
    # Check if the ORCA job finished successfully
    current_dir = os.getcwd()
    job_status = findtext("ORCA TERMINATED NORMALLY", path_to_orca_out, return_type='boolean')
    if not job_status:
        print("ORCA bond length scan did not finish successfully, terminating.")
        sys.exit(1)
    # Extract Relaxed Surface Scan results
    scan_results_start = findtext("The Calculated Surface using", path_to_orca_out,first_instance=True, return_type='line_no')
    scan_results_end = findtext("The Calculated Surface using", path_to_orca_out,last_instance=True, return_type='line_no')
    energy_lines = extract_lines(path_to_orca_out, scan_results_start[0]+1, scan_results_end[0]-2)
    if not energy_lines:
        print("No energy results found in ORCA output, terminating.")
        sys.exit(1)
    # Write the energy results to a file
    energy_results = [line.split() for line in energy_lines.splitlines()]
    # Find maximum energy guess and its corresponding bond length
    bl_max_energy, max_energy=max(energy_results, key=lambda x: float(x[1]))
    bl_min_energy, min_energy=min(energy_results, key=lambda x: float(x[1]))
    max_idx_num = energy_results.index([bl_max_energy, max_energy])
    max_idx = str(max_idx_num+1).zfill(3)
    # Plot the energy results (without fitting)
    energies = [(float(line[1])-float(min_energy))*627.503 for line in energy_results]
    bond_lengths = [float(line[0]) for line in energy_results]
    ids = [i+1 for i in range(len(energies))]
    max_fitted, local_maxima = plot_energy_scan(energies, bond_lengths, ids, f"{current_dir}/{orca_in_filename}_plot.png", fit='spline')
    # Check if the function is quadratic
    local_max = []
    for i in range(1, len(energy_results)-1):
        curr = float(energy_results[i][1])
        prev = float(energy_results[i-1][1])
        next_ = float(energy_results[i+1][1])
        if curr > prev and curr > next_:
            local_max.append(i+1)  # Store the index of the local maximum (1-based)
    if len(local_max)>1:
        print("The function is not quadratic, terminating.")
        error = "multiple maxima"
        result = [False, error, max_idx, bl_max_energy, local_max]
        return result
    identify_result = identify_bl_scan_max(max_fitted, local_maxima, energy_results)
    return identify_result if isinstance(identify_result, list) else [identify_result, "scan successful", max_idx, bl_max_energy]
def rerun_bl_scan_max_not_found(path_to_orca_out, orca_in_filename,bl_check, mech_step, cluster, rerun_no = False, list_constraints=None, custom_cluster_path=None):
    """
    Rerun ORCA bond length scan if the maximum energy was not found.
    Parameters
    ----------
    path_to_orca_out : str
        Path to the ORCA output file.
    bl_check : list
        List containing the results of the bond length scan check.
    position : str
        Position of the maximum energy ('start' or 'end').
    mech_step : str
        Mechanistic step for the scan.
    cluster : str, optional
        Cluster type for job submission. Default is 'slurm'.
    rerun_no : int, optional
        Rerun number for the ORCA bond length scan. Default is False.
    list_constraints : list of int, optional
        List of atom indices to be frozen during optimization (0-indexed). Default is None.
    custom_cluster_path : str, optional
        Custom path to the cluster submission script. Default is None.
    Returns
    -------
    None
    """
    # Check if the ORCA job finished successfully
    if f"_rerun_{rerun_no-1}" in orca_in_filename:
        orca_in_filename = orca_in_filename.replace(f"_rerun_{rerun_no-1}", "")
    current_dir = os.getcwd()
    charge_line = findtext("Total Charge", path_to_orca_out, first_instance=True, return_type='full_line', case_sensitive=True)
    charge = charge_line[0].split()[-1]
    spinmult_line = findtext("Multiplicity", path_to_orca_out, first_instance=True, return_type='full_line', case_sensitive=True)
    spinmult = spinmult_line[0].split()[-1]
    if bl_check[1] == "max at start":
        final_bl= float(bl_check[3]) - 0.3
    elif bl_check[1] == "max at end":
        final_bl = float(bl_check[3]) + 0.3
    else:
        print("Error: Unexpected position of maximum energy in bond length scan check.")
        sys.exit(1)
    if not bl_check[0]:
        print("Rerunning ORCA bond length scan.")
        parameters = {
            "name_xyz": f"{current_dir}/{orca_in_filename}.{str(bl_check[2]).zfill(3)}.xyz",
            "charge": charge,
            "spinmult": spinmult,
            "mech_step": mech_step,
            "custom_final_bl": final_bl,
            "rerun_no": rerun_no,
            "list_constraints": list_constraints,
            "step_no": 10,
            "step_change": 0.05,
            "scan_type": 'breaking'
        }
        orca_bl_id = submit_job('orca', 'bond_length_scan', parameters, cluster=cluster, custom_script_path=custom_cluster_path)
        wait_job(orca_bl_id, cluster, "orca", f"{orca_in_filename}_rerun_{rerun_no}.out", wait_time=5, custom_script_path=custom_cluster_path)
    else:
        print("ORCA bond length scan was successful.")
def rerun_bl_scan_not_close_enough(path_to_orca_out, orca_in_filename, bl_check, mech_step, cluster, scan_type='breaking', rerun_no=False, list_constraints=None, custom_cluster_path=None):
    """
    Rerun ORCA bond length scan if the maximum energy is not close enough to the fitted maximum energy.
    Parameters
    ----------
    path_to_orca_out : str
        Path to the ORCA output file.
    bl_check : list
        List containing the results of the bond length scan check.
    mech_step : str
        Mechanistic step for the scan.
    rerun_no : int, optional
        Rerun number for the ORCA bond length scan. Default is False.
    list_constraints : list of int, optional
        List of atom indices to be frozen during optimization (0-indexed). Default is None.
    Returns
    -------
    None
    """
    # Check if the ORCA job finished successfully
    if f"_rerun_{rerun_no-1}" in orca_in_filename:
        orca_in_filename = orca_in_filename.replace(f"_rerun_{rerun_no-1}", "")
    current_dir = os.getcwd()
    charge_line = findtext("Total Charge", path_to_orca_out, first_instance=True, return_type='full_line', case_sensitive=True)
    charge = charge_line[0].split()[-1]
    spinmult_line = findtext("Multiplicity", path_to_orca_out, first_instance=True, return_type='full_line', case_sensitive=True)
    spinmult = spinmult_line[0].split()[-1]
    # Extract the final bond length from the bl_check
    _, _, [first_guess, first_guess_bl], [second_guess, second_guess_bl] = bl_check
    if scan_type == 'breaking':
        guess = [first_guess, second_guess]
        guesses_bl = [first_guess_bl, second_guess_bl]
        initial_bl = min(guesses_bl)
        initial_idx = guess[guesses_bl.index(initial_bl)]
        final_bl = max(guesses_bl)
    elif scan_type == 'formation':
        guess = [first_guess, second_guess]
        guesses_bl = [first_guess_bl, second_guess_bl]
        initial_bl = max(guesses_bl)
        initial_idx = guess[guesses_bl.index(initial_bl)]
        final_bl = min(guesses_bl)
    print("Rerunning ORCA bond length scan with adjusted final bond length.")
    parameters = {
        "name_xyz": f"{current_dir}/{orca_in_filename}.{str(initial_idx).zfill(3)}.xyz",
        "charge": charge,
        "spinmult": spinmult,
        "mech_step": mech_step,
        "custom_final_bl": final_bl,
        "step_no": 10,
        "step_change": 0.05,
        "scan_type": scan_type,
        "rerun_no": rerun_no,
        "list_constraints": list_constraints
    }
    orca_bl_id = submit_job('orca', 'bond_length_scan', parameters, cluster=cluster, custom_script_path=custom_cluster_path)
    wait_job(orca_bl_id, cluster, "orca", f"{orca_in_filename}_rerun_{rerun_no}.out", wait_time=5, custom_script_path=custom_cluster_path)
    
def check_ts_guess(path_to_xyz, mech_step):
    """
    Check if the transition state guess has a linear or bent O-N-N angle.
    Parameters
    ----------
    path_to_xyz : str
        Path to the .xyz file containing the transition state guess.
    mech_step : str
        Mechanistic step for the transition state guess, either "oxo" or "hat".
    Returns
    -------
    str or bool
        'linear' if the O-N-N angle is linear, 'bent' if it is bent, False if no oxo-N₂ ligand is found.
    """
    guess3D = mol3D()
    guess3D.readfromxyz(path_to_xyz,read_final_optim_step=True)
    oxo_n2_ligand = find_oxo_n2_ligand(path_to_xyz)
    if oxo_n2_ligand:
        o_idx = oxo_n2_ligand[0]
        n1_idx = oxo_n2_ligand[1]
        n2_idx = oxo_n2_ligand[2]
    else:
        return False 
    if mech_step == "oxo":
        o_n_n_angle=guess3D.getAngle(o_idx, n1_idx, n2_idx)
        if o_n_n_angle > 170 and o_n_n_angle < 190:
            return 'linear'
        elif o_n_n_angle > 110 and o_n_n_angle < 130:
            return 'bent'
    return False

def check_double_maxima(orca_in_filename, bl_check, mech_step):
    """
    Check if ORCA bond length scan has multiple maxima.

    Parameters
    ----------
    path_to_orca_out : str
        Path to the ORCA output file.
    orca_in_filename : str
        ORCA input file NAME (not path) for the bond length scan.
    bl_check : list
        List containing scan check results.
    mech_step : str
        Mechanistic step for the scan.

    Returns
    -------
    tuple or bool
        (True, guess_index) if a suitable guess is found, False otherwise.
    """
    current_dir = os.getcwd()
    max_indices = bl_check[4]
    check_results = []
    for max_ in max_indices:
        max3D = mol3D()
        max3D.readfromxyz(f"{current_dir}/{orca_in_filename}.{str(max_).zfill(3)}.xyz", read_final_optim_step=True)
        print("Checking transition state guess at index:", max_)
        check = check_ts_guess(f"{current_dir}/{orca_in_filename}.{str(max_).zfill(3)}.xyz", mech_step)
        if check:
            check_results.append([max_, check])
            print(f"Found suitable transition state guess at index {max_} with {check} O-N-N angle.")
        if not check:
            print(f"No suitable transition state guess found at index {max_}.")
    if len(check_results) == 0:
        print("No transition state guesses found in the bond length scan.")
        return False
    if mech_step == "oxo":
        if len(check_results) > 1:
            # Favor the guess with bent O-N-N angle
            bent_guesses = [res for res in check_results if res[1] == 'bent']
            if bent_guesses:
                if len(bent_guesses) > 1:
                    print("Multiple bent O-N-N angle guesses found. Please check the structures manually.")
                    return False
                elif len(bent_guesses) == 1:
                    print(f"Using the bent O-N-N angle guess: {bent_guesses[0][0]}")
                    return bent_guesses[0][0]
            else:
                linear_guesses = [res for res in check_results if res[1] == 'linear']
                if linear_guesses:
                    if len(linear_guesses) > 1:
                        print("Multiple linear O-N-N angle guesses found. Please check the structures manually.")
                        return False
                    elif len(linear_guesses) == 1:
                        print(f"Using the linear O-N-N angle guess: {linear_guesses[0][0]}")
                        return linear_guesses[0][0]
                else:
                    print("No suitable transition state guess found in the bond length scan, check manually.")
                    return False
        if len(check_results) == 1:
            print(f"Using the only found transition state guess: {check_results[0][0]}")
            return check_results[0][0]
    # Default return if mech_step is not "oxo"
    return False

def troubleshoot_orca_bl_scan(path_to_orca_out, orca_in_filename, out_filename, mech_step, cluster, scan_type='formation', list_constraints=None, custom_cluster_path=None):
    """
    Troubleshoot ORCA bond length scan.
    This function writes files to disk, submits jobs to the scheduler, and may overwrite existing files.
    It does not use or modify global variables.
    Parameters
    ----------
    path_to_orca_out : str
        Path to the ORCA output file.
    orca_in_filename : str
        ORCA input file NAME (not path) for the bond length scan.
    mech_step : str
        Mechanistic step for the scan. "oxo" for oxo-N₂ bond breaking, "hat" for hydroxo-Methane bond breaking.
    out_filename : str
        Name for the output file where the final optimization step will be extracted.
    list_constraints : list of int, optional
        List of atom indices to be frozen during optimization (0-indexed). Default is None.
    Returns
    -------
    None
    """
    # Check if the ORCA job finished successfully
    match = re.search(r'rerun_(\d+)', orca_in_filename)
    if match:
        rerun_no = int(match.group(1))+1
    else:
        rerun_no = 1
    current_dir = os.getcwd()
    bl_check = check_orca_bl_scan(path_to_orca_out, orca_in_filename)
    if not bl_check[0]:
        if rerun_no >= 3:
            print("Two reruns of the ORCA bond length scan were not successful. Using the first maximum energy guess found.")
            if bl_check[1] == "max at start" or bl_check[1] == "max at end":
                extract_final_opt_step(f"{current_dir}/{orca_in_filename}.{str(bl_check[2]).zfill(3)}.xyz", f"{current_dir}/{out_filename}.xyz")
                return True, 0
        elif rerun_no >=4:
            print("Three reruns of the ORCA bond length scan were not successful. Please check the structures manually.")
            sys.exit(1)
        if bl_check[1] == "max at start" or bl_check[1] == "max at end":
            rerun_bl_scan_max_not_found(path_to_orca_out, orca_in_filename, bl_check, mech_step, cluster, rerun_no=rerun_no,list_constraints=list_constraints, custom_cluster_path=custom_cluster_path)
            return False, rerun_no
        elif bl_check[1] == "multiple maxima":
            print(f"Multiple maxima found. Please check the structures manually. Local Maxima are {bl_check[4]}")
            # Ask for manual check and write a new python code to continue from there choosing one structure
            mult_guess = check_double_maxima(orca_in_filename, bl_check, mech_step)
            if mult_guess:
                extract_final_opt_step(f"{current_dir}/{orca_in_filename}.{str(mult_guess).zfill(3)}.xyz", f"{current_dir}/{out_filename}.xyz")
            else:
                print("No suitable transition state guess found in the bond length scan, check manually.")
                sys.exit
        elif bl_check[1] == "max energy not close enough":
            print("Max energy not close enough. Rerunning the finer scan.")
            rerun_bl_scan_not_close_enough(path_to_orca_out, orca_in_filename,bl_check, mech_step, cluster, scan_type = scan_type,rerun_no =rerun_no, list_constraints=list_constraints, custom_cluster_path=custom_cluster_path)
            return False, rerun_no
    else:
        extract_final_opt_step(f"{current_dir}/{orca_in_filename}.{str(bl_check[2]).zfill(3)}.xyz", f"{current_dir}/{out_filename}.xyz")
        print("ORCA bond length scan was successful.")
        print("Final optimization step extracted to TSguess.xyz.")
        print("Plot saved as orca_bl_scan_plot.png.")
    return True, 0
def orca_bond_length_scan(terachem_opt_xyz, orca_bl_name, charge, spinmult, mech_step, ts_guess_name, cluster, step_no, options):
    """
    Run ORCA bond length scan for a mechanistic step and troubleshoot if necessary.
    Parameters
    ----------
    terachem_opt_xyz : str
        Path to the Terachem optimized .xyz file.
    orca_bl_name : str
        Name for the ORCA bond length scan input/output files.
    charge : int
        Charge of the complex to be scanned.
    spinmult : int
        Spin multiplicity of the complex to be scanned (2S+1).
    mech_step : str
        Mechanistic step to be scanned, either 'oxo' or 'hydroxo'.
    ts_guess_name : str
        Name for the transition state guess output file.
    cluster : str
        Cluster type for job submission. Default is 'slurm'.
    step_no : int
        Number of steps for the bond length scan.
    options : dict, optional
        Dictionary of options for the bond length scan.
        {
            'scan_type': 'breaking' or 'formation',
            'step_change': float (step change for the bond length scan, if None will use default step size based on the mechanistic step),
            'list_constraints': list of int (list of atom indices to be frozen during optimization, 0-indexed),
            'custom_cluster_path': str (custom path to the cluster submission script)
        }
    """
    base_dir = os.getcwd()
    scan_type, step_change, list_constraints, custom_cluster_path = options.get('scan_type', scan_type), options.get('step_change', step_change), options.get('list_constraints', list_constraints), options.get('custom_cluster_path', custom_cluster_path)
    # Run ORCA bond length scan
    parameters = {
        "name_xyz": terachem_opt_xyz,
        "charge": charge,
        "spinmult": spinmult,
        "mech_step": mech_step,
        "step_no": step_no,
        "step_change": step_change,
        "scan_type": scan_type,
        "list_constraints": list_constraints
    }
    bl_scan_id = submit_job('orca', 'bond_length_scan', parameters, orca_bl_name, cluster=cluster, custom_script_path=custom_cluster_path)
    # Wait for ORCA bond length scan to finish
    wait_job(bl_scan_id, cluster, "orca", f"{orca_bl_name}.out", custom_script_path=custom_cluster_path)
    # Check ORCA output file and TS guess (extra points if graph bl-scan energy)
    trouble_shoot, rerun_no= troubleshoot_orca_bl_scan(
        f"{base_dir}/{orca_bl_name}.out", orca_bl_name,
        ts_guess_name, mech_step, cluster, scan_type=scan_type, 
        list_constraints=list_constraints, custom_cluster_path=custom_cluster_path)
    while not trouble_shoot:
        print("Troubleshooting ORCA bond length scan again...")
        # Check ORCA output file and TS guess
        trouble_shoot, rerun_no = troubleshoot_orca_bl_scan(
            f"{base_dir}/{orca_bl_name}_rerun_{rerun_no}.out", 
            f"{orca_bl_name}_rerun_{rerun_no}", 
            ts_guess_name, mech_step, cluster, 
            scan_type=scan_type, list_constraints=list_constraints, 
            custom_cluster_path=custom_cluster_path)
        if trouble_shoot:
            print("ORCA bond length scan was successful.")
            break
        else:
            print("ORCA bond length scan failed. Rerunning...")