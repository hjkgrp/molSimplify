# @file jobgen.py
#  Generates jobscripts for queueing systems
#
#  Written by Tim Ioannidis for HJK Group
#
#  Dpt of Chemical Engineering, MIT

import argparse
import glob
import sys
import os
import subprocess

# Generates jobscripts for SGE queueing system
#  @param args Namespace of arguments
#  @param jobdirs Subdirectories for jobscript placement


def sgejobgen(args, jobdirs):
    # consolidate lists
    jd = []
    for i, s in enumerate(jobdirs):
        if isinstance(s, list):
            for ss in s:
                jd.append(ss)
        else:
            jd.append(s)
    jobdirs = jd
    gpus = '1'  # initialize gpus
    cpus = '1'  # initialize cpus
    # loop over job directories
    for job in jobdirs:
        # form jobscript identifier
        if args.jname:
            jobname = args.jname+str(args.jid)
            #jobname = jobname[:8]
        else:
            # jobname = 'job'+str(args.jid)
            jobname = [f for f in os.listdir(job) if f.endswith('.xyz')][0][:-4]
        args.jid += 1
        output = open(job+'/'+jobname+'_jobscript', 'w')
        output.write('#$ -S /bin/bash\n')
        output.write('#$ -N %s\n' % (jobname))
        output.write('#$ -R y\n')
        output.write('#$ -cwd\n')
        if not args.wtime:
            output.write('#$ -l h_rt=168:00:00\n')
        else:
            wtime = args.wtime.split(':')[0]
            wtime = wtime.split('h')[0]
            output.write('#$ -l h_rt='+wtime+':00:00\n')
        if not args.memory:
            output.write('#$ -l h_rss=8G\n')
        else:
            mem = args.memory.split('G')[0]
            output.write('#$ -l h_rss='+mem+'G\n')
        if not args.queue:
            if args.qccode and args.qccode in 'terachem TeraChem TERACHEM tc TC Terachem':
                output.write('#$ -q gpus\n')
                if args.gpus:
                    output.write('#$ -l gpus='+args.gpus+'\n')
                    gpus = args.gpus
                else:
                    output.write('#$ -l gpus=1\n')
            else:
                output.write('#$ -q cpus\n')
                if args.cpus:
                    output.write('#$ -l cpus='+args.cpus+'\n')
                    cpus = args.cpus
                else:
                    output.write('#$ -l cpus=1\n')
        else:
            output.write('#$ -q '+args.queue+'\n')
            if args.cpus:
                output.write('#$ -l cpus='+args.cpus+'\n')
                cpus = args.cpus
            elif args.gpus:
                output.write('#$ -l gpus='+args.gpus+'\n')
                gpus = args.gpus
            else:
                output.write('#$ -l gpus=1\n')
        if args.gpus:
            output.write('#$ -pe smp '+args.gpus+'\n')
        elif args.cpus:
            output.write('#$ -pe smp '+args.cpus+'\n')
        else:
            output.write('#$ -pe smp 1\n')
        if args.joption:
            for jopt in args.joption:
                output.write('# '+jopt+'\n')
        if args.modules:
            for mod in args.modules:
                output.write('module load '+mod+'\n')
        if args.gpus:
            output.write('export OMP_NUM_THREADS='+args.gpus+'\n')
        elif args.cpus:
            output.write('export OMP_NUM_THREADS='+args.cpus+'\n')
        else:
            output.write('export OMP_NUM_THREADS=1\n')
        if args.jcommand:
            for com in args.jcommand:
                output.write(com+'\n')
        if args.qccode and args.qccode in 'terachem TeraChem TERACHEM tc TC Terachem':
            tc = False
            if args.jcommand:
                for jc in args.jcommand:
                    if 'terachem' in jc:
                        tc = True
            if not tc:
                output.write(
                    'terachem ' + jobname + '.in > $SGE_O_WORKDIR/' + jobname + '.out')
        elif args.qccode and ('gam' in args.qccode.lower() or 'qch' in args.qccode.lower()):
            gm = False
            qch = False
            if args.jcommand:
                for jc in args.jcommand:
                    if 'rungms' in jc:
                        gm = True
                    if 'qchem' in jc:
                        qch = True
            if not gm and 'gam' in args.qccode.lower():
                output.write('rungms gam.inp '+cpus + ' > gam.out')
            elif not qch and 'qch' in args.qccode.lower():
                output.write('qchem qch.inp '+cpus + ' > qch.out')
        elif args.qccode and ('orc' in args.qccode.lower() or 'molc' in args.qccode.lower()):
            orc = False
            molc = False
            if args.jcommand:
                for jc in args.jcommand:
                    if 'orca' in jc:
                        orc = True
                    if 'molcas' in jc:
                        molc = True
            if not orc and 'orca' in args.qccode.lower():
                output.write('orca orca.in > orca.out')
            elif not molc and 'molc' in args.qccode.lower():
                output.write('pymolcas molcas.input -f')
        else:
            print(
                'Not supported QC code requested. Please input execution command manually')
        output.close()

# Generates jobscripts for SLURM queueing system
#  @param args Namespace of arguments
#  @param jobdirs Subdirectories for jobscript placement


def slurmjobgen(args, jobdirs):
    # consolidate lists
    jd = []
    for i, s in enumerate(jobdirs):
        if isinstance(s, list):
            for ss in s:
                jd.append(ss)
        else:
            jd.append(s)
    jobdirs = jd
    gpus = '1'  # initialize gpus
    cpus = '1'  # initialize cpus
    # loop over job directories
    for job in jobdirs:
        # form jobscript identifier
        if args.jname:
            jobname = args.jname+str(args.jid)
            jobname = jobname[:8]
        else:
            # jobname = 'job'+str(args.jid)
            jobname = [f for f in os.listdir(job) if f.endswith('.xyz')][0][:-4]
        args.jid += 1
        output = open(job+'/'+jobname+'_jobscript', 'w')
        output.write('#!/bin/bash\n')
        output.write('#SBATCH --job-name=%s\n' % (jobname))
        output.write('#SBATCH --output=batch.log\n')
        output.write('#SBATCH --export=ALL\n')
        if not args.wtime:
            output.write('#SBATCH -t 168:00:00\n')
        else:
            wtime = args.wtime.split(':')[0]
            wtime = wtime.split('h')[0]
            output.write('#SBATCH -t '+wtime+':00:00\n')
        if not args.memory:
            output.write('#SBATCH --mem==8G\n')
        else:
            mem = args.memory.split('G')[0]
            output.write('#SBATCH --mem='+mem+'G\n')
        if not args.queue:
            if args.qccode and args.qccode in 'terachem TeraChem TERACHEM tc TC Terachem':
                output.write('#SBATCH --partition=gpus\n')
            else:
                output.write('#SBATCH --partition=cpus\n')
        else:
            output.write('#SBATCH --partition='+args.queue+'\n')
        nod = False
        nnod = False
        if args.joption:
            for jopt in args.joption:
                output.write('#SBATCH '+jopt+'\n')
                if 'nodes' in jopt:
                    nod = True
                if 'ntasks' in jopt:
                    nnod = True
        if not nod:
            output.write('#SBATCH --nodes=1\n')
        if not nnod:
            output.write('#SBATCH --ntasks-per-node=1\n')
        if args.modules:
            for mod in args.modules:
                output.write('module load '+mod+'\n')
        if args.jcommand:
            for com in args.jcommand:
                output.write(com+'\n')
        if args.qccode and args.qccode in 'terachem TeraChem TERACHEM tc TC Terachem':
            tc = False
            if args.jcommand:
                for jc in args.jcommand:
                    if 'terachem' in jc:
                        tc = True
            if not tc:
                output.write('terachem ' + jobname + '.in > ' + jobname + '.out')
        elif args.qccode and ('gam' in args.qccode.lower() or 'qch' in args.qccode.lower()):
            gm = False
            qch = False
            if args.jcommand:
                for jc in args.jcommand:
                    if 'rungms' in jc:
                        gm = True
                    if 'qchem' in jc:
                        qch = True
            if not gm and 'gam' in args.qccode.lower():
                output.write('rungms gam.inp '+cpus + ' > gam.out')
            elif not qch and 'qch' in args.qccode.lower():
                output.write('qchem qch.inp '+cpus + ' > qch.out')
        elif args.qccode and ('orc' in args.qccode.lower() or 'molc' in args.qccode.lower()):
            orc = False
            molc = False
            if args.jcommand:
                for jc in args.jcommand:
                    if 'orca' in jc:
                        orc = True
                    if 'molcas' in jc:
                        molc = True
            if not orc and 'orca' in args.qccode.lower():
                output.write('orca orca.in > orca.out')
            elif not molc and 'molc' in args.qccode.lower():
                output.write('pymolcas molcas.input -f')
        else:
            print(
                'No supported QC code requested. Please input execution command manually')
        output.close()
