import os
import sys
import time
import re
import subprocess
import importlib.util

def findtext(text, filename, last_instance=False, first_instance=False, return_type='boolean', case_sensitive=False, whole_word=True):
    """
    Search for a text in a file with options for whole word and return types.
    Parameters
    ----------
    text : str
        Text or substring to search for.
    filename : str
        Path to the file to search.
    last_instance : bool, optional
        If True, return only the last matching line.
    first_instance : bool, optional
        If True, return only the first matching line.
    return_type : str, optional
        Determines the type of output:
        - 'boolean' → True if word is found, False otherwise.
        - 'line_no' → List of line numbers where the word was found.
        - 'full_line' → List of matching full lines.
    case_sensitive : bool, optional
        If False (default), search is case-insensitive.
    whole_word : bool, optional
        If True, match whole words only. E.g. "energy" will not match "finalenergy". Default is True. 
        Phrases like "final energy" will work as expected always.
    Returns
    -------
    bool or list
        Depending on return_type.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    if whole_word:
        pattern = re.compile(r'\b' + re.escape(text) + r'\b', flags)
    else:
        pattern = re.compile(re.escape(text), flags)

    matches = []

    with open(filename, 'r', encoding ='utf-8', errors='replace') as f:
        for lineno, line in enumerate(f, start=1):
            if pattern.search(line):
                matches.append((lineno, line.rstrip()))

    if first_instance and matches:
        matches = [matches[0]]
    elif last_instance and matches:
        matches = [matches[-1]]

    if return_type == 'boolean':
        return bool(matches)
    elif return_type == 'line_no':
        return [lineno for lineno, _ in matches]
    elif return_type == 'full_line':
        return [line for _, line in matches]
    else:
        raise ValueError("return_type must be 'boolean', 'line_no', or 'full_line'")

def extract_lines(filename, start_line, end_line):
    """
    Extract text between two given line numbers (inclusive) and return as a single string.

    Parameters
    ----------
    filename : str
        Path to the file.
    start_line : int
        The starting line number (1-based).
    end_line : int
        The ending line number (1-based, inclusive).

    Returns
    -------
    str
        Text between start_line and end_line as a single string.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    return ''.join(lines[start_line - 1:end_line])

def extract_final_opt_step(original_filename, output_filename):
    """
    Extract the final optimization step from a .xyz file and write it to a new file.
    Parameters
    ----------
    original_filename : str
        Path to the original .xyz file.
    output_filename : str
        Path to the output .xyz file where the final optimization step will be written.
    """

    with open(original_filename, 'r') as f:
        lines = f.read().splitlines()
        try:
            atom_count = int(lines[0].strip())
        except ValueError:
            atom_count = 0  # Handle case where first line is not an integer
        start_lines = findtext(
            f"{atom_count}",
            original_filename,
            last_instance=True,
            return_type='line_no',
            case_sensitive=True,
            whole_word=True
        )
        if not start_lines:
            raise ValueError("Could not find the final optimization step in the file.")
        start = start_lines[0]  # Convert to 0-based index
    # Write lines from 'start' to the end to a new file
    with open(output_filename, 'w') as out_f:
        out_f.write(f"{atom_count}\n")
        for line in lines[start:]:
            out_f.write(line + '\n')

def load_custom_cluster_module(custom_script_path):
    """
    Load a custom cluster module from a script file.
    File should have the following structure, example from slurm cluster:
    '''
    def submit_job(job_script):
        # Simulate: sbatch job_script
        result = subprocess.run(['sbatch', job_script], stdout=subprocess.PIPE, text=True)
        # Typical sbatch output: "Submitted batch job 123456"
        job_id = result.stdout.strip().split()[-1]
        return job_id
    def get_job_state(job_id):
        # Simulate: squeue -j job_id -h -o "%T"
        result = subprocess.run(['squeue', '-j', str(job_id), '-h', '-o', '%T'], stdout=subprocess.PIPE, text=True)
        state = result.stdout.strip().lower()
        if state == 'RUNNING': # Change according to cluster
            state = 'running'
        elif state == 'PENDING': # Change according to cluster
            state = 'pending' 
        elif state == 'COMPLETED':
            state = 'completed' 
        elif state == '':
            state = None
        return state
    def cancel_job(job_id):
        # Simulate: scancel job_id
        result = subprocess.run(['scancel', str(job_id)], stdout=subprocess.PIPE, text=True)
        return result.returncode == 0
    def write_jobscript(calc_type, in_file, hours, out_file="job.out"):
        # Write slurm file
        if calc_type == "orca":
            slurm_base = "" # Write your cluster specific slurm script here for orca 
        elif calc_type == "terachem":
            slurm_base = "" # Write your cluster specific slurm script here  for terachem
        return slurm_base
    '''
    Parameters
    ----------
    custom_script_path : str
        Path to the custom cluster module (without .py extension).
    Returns
    -------
    module
        The loaded custom cluster module.
    """
    module_name = os.path.splitext(os.path.basename(custom_script_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, custom_script_path)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    return custom_module
    
def get_job_state(job_id, cluster, custom_script_path=None):
    """
    Returns the cluster's job state.
    Parameters
    ----------
    job_id : str
        Cluster job ID for the job to check.
    cluster : str
        Cluster type, e.g. "slurm", "sge", "custom".
    custom_script_path : str, optional
        Path to the custom cluster module (required if cluster is "custom").
    """
    if cluster == "slurm":
        try:
            result = subprocess.run(
                ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            state = result.stdout.strip()
            return state if state else None
        except subprocess.CalledProcessError:
            return None
    if cluster == "sge":
        try:
            result = subprocess.run(
                ["qstat", "-u", "$USER"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            if "Following jobs do not exist" in result.stdout:
                return None
            for line in result.stdout.splitlines():
                if line.startswith(str(job_id)):
                    return line.split()[4]  # State is the 5th column
            return None
        except subprocess.CalledProcessError:
            return None
    if cluster == "custom":
        if not custom_script_path:
            raise ValueError("custom_script_path must be provided for custom cluster type.")
        custom_module = load_custom_cluster_module(custom_script_path)
        if hasattr(custom_module, 'get_job_state'):
            return custom_module.get_job_state(job_id)
        else:
            raise AttributeError("The custom cluster module must have a 'get_job_state' function.")

def wait_job(job_id, cluster, job_type, output_file, wait_time = 1, custom_script_path=None):
    """
    Wait for Terachem job to finish.
    Parameters
    ----------
    job_id : str
        Cluster job ID for the job to wait for.
    cluster : str
        Cluster type, e.g. "slurm", "sge", "custom".
    job_type : str
        Type of job to wait for, e.g. "terachem", "orca",
    output_file : str
        Name of the output file to check for job completion.
    wait_time : int, optional
        Time in minutes to wait between checks. Default is 5 minutes.
    custom_script_path : str, optional
        Path to the custom cluster module (required if cluster is "custom").
    """
    time.sleep(60)  # Initial wait to allow job to start
    # Check if the job started or is in queue:
    run_slurm = 0
    while True:
        run_slurm += 1
        if cluster == "slurm":
            job_state = get_job_state(job_id, cluster)
            if job_state is None:
                if run_slurm > 5:  # If job not found after 5 attempts, exit
                    print(f"Job {job_id} not found after multiple attempts. Exiting.")
                    break
                else:
                    print(f"Job {job_id} not found. Retrying in {wait_time} minutes...")
                    time.sleep(wait_time*60)
            elif job_state == 'PENDING':
                print(f"Job {job_id} is pending. Waiting for it to start...")
                time.sleep(wait_time*60)
            elif job_state in ['RUNNING', 'COMPLETING']:
                print(f"Job {job_id} is running. Waiting for it to finish...")
                break
        elif cluster == "sge":
            job_state = get_job_state(job_id, cluster)
            if job_state is None:
                if run_slurm > 5:  # If job not found after 5 attempts, exit
                    print(f"Job {job_id} not found after multiple attempts. Exiting.")
                    break
                else:
                    print(f"Job {job_id} not found. Retrying in {wait_time} minutes...")
                    time.sleep(wait_time*60)
            elif job_state == 'qw':
                print(f"Job {job_id} is pending. Waiting for it to start...")
                time.sleep(wait_time*60)
            elif job_state in ['r', 't']:
                print(f"Job {job_id} is running. Waiting for it to finish...")
                break
        elif cluster == "custom":
            job_state = get_job_state(job_id, cluster, custom_script_path)
            if job_state is None:
                if run_slurm > 5:  # If job not found after 5 attempts, exit
                    print(f"Job {job_id} not found after multiple attempts. Exiting.")
                    break
                else:
                    print(f"Job {job_id} not found. Retrying in {wait_time} minutes...")
                    time.sleep(wait_time*60)
            elif job_state == 'pending':
                print(f"Job {job_id} is pending. Waiting for it to start...")
                time.sleep(wait_time*60)
            elif job_state == 'running':
                print(f"Job {job_id} is running. Waiting for it to finish...")
                break
    if job_type == "terachem":
        runs = 0
        while not os.path.exists(output_file):
            print(f"Waiting for {output_file} to be created...")
            time.sleep(5)
        while True:
            # Check if the job is still running
            job_status = findtext("Job finished", output_file, return_type='boolean')
            if job_status:
                print("Terachem job finished.")
                break
            else:
                if findtext("Job terminated", output_file, return_type='boolean'):
                    error_start = findtext("SCF did not converge", output_file, return_type='line_no')
                    error_end = findtext("FINAL ENERGY", output_file, return_type='line_no')
                    if error_start and error_end:
                        error_message = extract_lines(output_file, error_start[0]-1, error_end[0]-2)
                    elif findtext("Incorrect molecular charge or spin multiplicity", output_file, return_type='boolean'):
                        error_line = findtext("Incorrect molecular charge or spin multiplicity", output_file, return_type='full_line')
                        error_message = "Incorrect molecular charge or spin multiplicity."
                        return False
                    else:
                        error_message = ["Job terminated without a clear error message."]
                    print(f"Terachem job terminated with error: {error_message}")
                    sys.exit(1)
                else:
                    if runs % 10 == 0:
                        print("Waiting for Terachem job to finish...")
                    runs += 1
                    if runs > (300/int(wait_time)): # 5 hours max
                        print("Terachem job ran out of time, terminating.")
                        sys.exit(1)
                    else:
                        time.sleep(wait_time*60) # Wait for 30 minutes before checking again
    elif job_type == "orca":
        runs = 0
        while not os.path.exists(output_file):
            print(f"Waiting for {output_file} to be created...")
            time.sleep(5)
        while True:
            # Check if the job is still running
            job_status = findtext("ORCA TERMINATED NORMALLY", output_file, return_type='boolean')
            if job_status:
                print("Orca job finished.")
                break
            else:
                if runs % 10 == 0:
                        print("Waiting for Orca job to finish...")
                runs += 1
                if runs > ((14*60)/int(wait_time)): # 14 hours max
                    print("Orca job ran out of time, checking calculation")
                    break
                else:
                    time.sleep(wait_time*60) # Wait for 30 minutes before checking again
    else:
        raise ValueError("Unsupported job type. Use 'terachem' or 'orca'.")
    
def write_slurm_job(calc_type, in_file, hours, out_file="job.out"):
    """
    Write Terachem input file for geometry optimization.
    Parameters
    ----
    calc_type : str
        Type of calculation to be conducted, e.g. Terachem, Orca, etc..
    in_file: str
        Name of input file.
    hours: str
        Number of hours for the calculation
    out_file: str
        Name of output file.
    Return
    ----
    slurm_job : str
        text for slurm submission file.
    """
    # Write slurm file
    if calc_type == "orca":
        slurm_base = f"""#!/bin/bash
#SBATCH --job-name={calc_type}
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=100G
#SBATCH --time={hours}:00:00
module purge
module use /orcd/pool/004/hjkulik_shared/modulefiles
module load community-modules
module load gcc/12.2.0
module load intel/2024.2.1
module load nvhpc
module unload mpi
module load orca/6.0.1
/orcd/pool/004/hjkulik_shared/src/orca_6_0_1_linux_x86-64_shared_openmpi416/orca {in_file} > {out_file}
"""
    elif calc_type == "terachem":
        slurm_base = f"""#!/bin/bash
#SBATCH --job-name={calc_type}
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=72G
#SBATCH --time={hours}:00:00
module purge
module use /orcd/pool/004/hjkulik_shared/modulefiles
module load community-modules
module load gcc/12.2.0
module load openmpi/5.0.6
module load intel/2024.2.1
module load cuda/12.4.0
module load cudnn/9.8.0.87-cuda12
module load nvhpc
module load terachem_cuda12/v1.9-2025.06-dev
export KMP_WARNINGS="off"
terachem {in_file} > {out_file}
"""
    return slurm_base

def write_sge_job(calc_type, in_file, hours, out_file="job.out", job_name="job"):
    if calc_type == "orca":
        sge_base = f"""#!/bin/bash
#$ -N {job_name}
#$ -R y
#$ -cwd
#$ -l h_rt={hours}:00:00
#$ -l h_rss=48G
#$ -q cpus
#$ -l cpus=1
#$ -pe smp 16
# -fin *
# -fout *

module load intel
module load orca/6.0

$ORCANRUN {in_file} > {out_file}
"""
    elif calc_type == "terachem":
        sge_base = f"""#$ -S /bin/bash
#$ -N {job_name}
#$ -cwd
#$ -l h_rt={hours}:00:00
#$ -l h_rss=8G 
#$ -q gpus 
#$ -l gpus=1
#$ -pe smp 1 
# # -fin {in_file} 
# # -fin *.xyz 
# # -fout {job_name}/ 

module load cuda
module load terachem 

export OMP_NUM_THREADS=1

terachem {in_file} > {out_file}
"""
    return sge_base

def wait_avogadro_optimization(output_file):
    run = 0
    while not os.path.exists(output_file):
        if run % 10 == 0:
            print(f"Waiting for {output_file} to be created...")
        run += 1
        time.sleep(12)

def submit_job(software, calculation, parameters, filename, cluster='slurm', rerun=False,custom_script_path=None):
    """
    Submit a job to the specified cluster.
    Parameters
    ----------
    software : str
        Software to use, e.g. "orca", "terachem".
    calculation : str
        Type of calculation, e.g. "sp","opt", "TS", "bond_length_scan".
    parameters : dict
        Dictionary of parameters required for writing the job. 
        Keys depend on software and calculation type, refer to the function documentation.
        {
            "name_xyz": ...,
            "charge": ...,
            "spinmult": ...,
            "basis_set": ...,
            "functional": ...,
            "lvlshift": ...,
            "lvla": ...,
            "lvlb": ...,
            "unrestricted": ...,
            "involved_atoms": atoms involved in TS search/bond length scan,
            "mech_step": mechanistic step, e.g. "oxo", "hat",
            "step_no": number of step in bond length scan,
            "step_change": increment or decrement in bond length,
            "scan_type": bond formation or bond breaking,
            "involved_atoms": atoms involved in TS search/bond length scan,
            "custom_final_bl": ,
            "list_constraints": list of constraints in calculation e.g. ["freeze 1 2", "distance 1 2 1.5"]
        }
    filename : str
        Base name for input/output files.
    cluster : str
        Cluster type, e.g. "slurm", "sge", "custom". Default is "slurm".
    rerun : bool or int, optional
        If True or int, indicates a rerun and appends to filenames. Default is False.
    custom_script_path : str, optional
        Path to the custom cluster module (required if cluster is "custom").
    Returns
    -------
    str
        Job ID from the cluster submission.
    """
    if software == "orca":
        if calculation == "bond_length_scan":
            from orca_bl_scan import write_orca_bl_scan
            job_text = write_orca_bl_scan(
                parameters["name_xyz"], parameters["charge"], parameters["spinmult"],
                parameters["mech_step"], parameters["step_no"], parameters["step_change"],
                parameters["scan_type"], parameters["unrestricted"], parameters["custom_final_bl"],
                list_constraints=parameters["list_constraints"]
            )
        else:
            from orca_opt import write_orca_opt
            job_text = write_orca_opt(
                parameters["name_xyz"], parameters["charge"], parameters["spinmult"],
                parameters["involved_atoms"], calculation,
                unrestricted=parameters.get("unrestricted", False),
                list_constraints=parameters.get("list_constraints"),
                rerun=False
            )
    elif software == "terachem":
        from terachem import write_terachem_opt
        job_text = write_terachem_opt(
            parameters["name_xyz"], filename, parameters["charge"], parameters["spinmult"], parameters["basis_set"],
            parameters["functional"], parameters["lvlshift"], parameters["lvla"], parameters["lvlb"],
            parameters["unrestricted"], parameters["list_constraints"]
        )
    if rerun:
        input= f"{filename}_rerun_{rerun}.in"
        output= f"{filename}_rerun_{rerun}.out"
    else:
        input= f"{filename}.in"
        output= f"{filename}.out"
    with open(input,"w") as f:
        f.write(job_text)
    if cluster.lower() == 'slurm':
        jobscript = write_slurm_job(software, input, "12", output)
        with open(f"{filename}_job","w") as f:
            f.write(jobscript)
        result = subprocess.run(["sbatch", f"{filename}_job"], capture_output=True, text=True)
        stdout_id = result.stdout.split()[-1] if result.stdout else "Unknown"
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    elif cluster.lower() == 'sge':
        jobscript = write_sge_job(software, input, "12", output)
        with open(f"{filename}_job","w") as f:
            f.write(jobscript)
        result = subprocess.run(["qsub", f"{filename}_job"], capture_output=True, text=True)
        stdout_id = result.stdout.split()[2] if result.stdout else "Unknown"
    elif cluster.lower() == 'custom':
        custom_module = load_custom_cluster_module(custom_script_path=custom_script_path)
        jobscript = custom_module.write_jobscript(software, input, "12", output)
        with open (f"{filename}_job","w") as f:
            f.write(jobscript)
        stdout_id=custom_module.submit_job(f"{filename}_job")
    return stdout_id