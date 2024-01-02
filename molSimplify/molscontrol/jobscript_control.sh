#!/usr/bin/env bash
#$ -S /bin/bash
#$ -N on-the-fly-control
#$ -R y
#$ -cwd
#$ -l h_rt=64:00:00
#$ -l h_rss=8G
#$ -q (gpus|gpusnew)
#$ -l gpus=1
#$ -pe smp 1
# -fin terachem_input
# -fin *.xyz
# -fout scr
#$ -cwd
#$ -V



module load anaconda
source activate mols_keras

module load terachem/tip
export OMP_NUM_THREADS=1
export RSH_COMMAND=ssh
export plm_rsh_agent=ssh


terachem terachem_input > $SGE_O_WORKDIR/dft.out &
PID_KILL=$!
molscontrol &
wait

mv *.log $SGE_O_WORKDIR
