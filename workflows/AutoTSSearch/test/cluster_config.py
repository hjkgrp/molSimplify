# slurm_custom.py

import subprocess

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

