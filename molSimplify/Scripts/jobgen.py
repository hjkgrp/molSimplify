# @file jobgen.py
#  Generates jobscripts for queueing systems
#
#  Written by Tim Ioannidis for HJK Group
#
#  Dpt of Chemical Engineering, MIT

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
    cpus = '1'  # initialize cpus
    # loop over job directories
    for job in jobdirs:
        # form jobscript identifier
        if args.jname:
            jobname = args.jname+str(args.jid)
            # jobname = jobname[:8]
        else:
            jobname = 'job'+str(args.jid)
        args.jid += 1
        output = []
        output.append('#$ -S /bin/bash\n')
        output.append('#$ -N %s\n' % (jobname))
        output.append('#$ -R y\n')
        output.append('#$ -cwd\n')
        if not args.wtime:
            output.append('#$ -l h_rt=168:00:00\n')
        else:
            wtime = args.wtime.split(':')[0]
            wtime = wtime.split('h')[0]
            output.append('#$ -l h_rt='+wtime+':00:00\n')
        if not args.memory:
            output.append('#$ -l h_rss=8G\n')
        else:
            mem = args.memory.split('G')[0]
            output.append('#$ -l h_rss='+mem+'G\n')
        if not args.queue:
            if args.qccode and args.qccode in 'terachem TeraChem TERACHEM tc TC Terachem':
                output.append('#$ -q gpus\n')
                if args.gpus:
                    output.append('#$ -l gpus='+args.gpus+'\n')
                else:
                    output.append('#$ -l gpus=1\n')
            else:
                output.append('#$ -q cpus\n')
                if args.cpus:
                    output.append('#$ -l cpus='+args.cpus+'\n')
                    cpus = args.cpus
                else:
                    output.append('#$ -l cpus=1\n')
        else:
            output.append('#$ -q '+args.queue+'\n')
            if args.cpus:
                output.append('#$ -l cpus='+args.cpus+'\n')
                cpus = args.cpus
            elif args.gpus:
                output.append('#$ -l gpus='+args.gpus+'\n')
            else:
                output.append('#$ -l gpus=1\n')
        if args.gpus:
            output.append('#$ -pe smp '+args.gpus+'\n')
        elif args.cpus:
            output.append('#$ -pe smp '+args.cpus+'\n')
        else:
            output.append('#$ -pe smp 1\n')
        if args.joption:
            multi_option = args.joption[0].split('-')
            if len(multi_option) > 1:
                args.joption = []
                for option in multi_option[1:]:
                    args.joption += ["-" + option]
            for jopt in args.joption:
                output.append(f'# {jopt.strip()}\n')
        if args.modules:
            for mod in args.modules:
                output.append('module load '+mod+'\n')
        if args.gpus:
            output.append('export OMP_NUM_THREADS='+args.gpus+'\n')
        elif args.cpus:
            output.append('export OMP_NUM_THREADS='+args.cpus+'\n')
        else:
            output.append('export OMP_NUM_THREADS=1\n')
        if args.jcommand:
            for com in args.jcommand:
                output.append(com+'\n')
        if args.qccode and args.qccode in 'terachem TeraChem TERACHEM tc TC Terachem':
            tc = False
            if args.jcommand:
                for jc in args.jcommand:
                    if 'terachem' in jc:
                        tc = True
            if not tc:
                output.append(
                    'terachem terachem_input > $SGE_O_WORKDIR/opttest.out\n')
            output.append('\nsleep 30\n')
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
                output.append('rungms gam.inp '+cpus + ' > gam.out\n')
            elif not qch and 'qch' in args.qccode.lower():
                output.append('qchem qch.inp '+cpus + ' > qch.out\n')
            output.append('\nsleep 30\n')
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
                output.append('orca orca.in > orca.out\n')
            elif not molc and 'molc' in args.qccode.lower():
                output.append('pymolcas molcas.input -f\n')
            output.append('\nsleep 30\n')
        else:
            print(
                'Not supported QC code requested. Please input execution command manually')
        with open(job+'/'+'jobscript', 'w') as f:
            f.writelines(output)

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
    cpus = '1'  # initialize cpus
    # loop over job directories
    for job in jobdirs:
        # form jobscript identifier
        if args.jname:
            jobname = args.jname+str(args.jid)
            jobname = jobname[:8]
        else:
            jobname = 'job'+str(args.jid)
        args.jid += 1
        output = []
        output.append('#!/bin/bash\n')
        output.append('#SBATCH --job-name=%s\n' % (jobname))
        output.append('#SBATCH --output=batch.log\n')
        output.append('#SBATCH --export=ALL\n')
        if not args.wtime:
            output.append('#SBATCH -t 168:00:00\n')
        else:
            wtime = args.wtime.split(':')[0]
            wtime = wtime.split('h')[0]
            output.append('#SBATCH -t '+wtime+':00:00\n')
        if not args.memory:
            output.append('#SBATCH --mem==8G\n')
        else:
            mem = args.memory.split('G')[0]
            output.append('#SBATCH --mem='+mem+'G\n')
        if not args.queue:
            if args.qccode and args.qccode in 'terachem TeraChem TERACHEM tc TC Terachem':
                output.append('#SBATCH --partition=gpus\n')
            else:
                output.append('#SBATCH --partition=cpus\n')
        else:
            output.append('#SBATCH --partition='+args.queue+'\n')
        nod = False
        nnod = False
        if args.joption:
            for jopt in args.joption:
                output.append('#SBATCH '+jopt+'\n')
                if 'nodes' in jopt:
                    nod = True
                if 'ntasks' in jopt:
                    nnod = True
        if not nod:
            output.append('#SBATCH --nodes=1\n')
        if not nnod:
            output.append('#SBATCH --ntasks-per-node=1\n')
        if args.modules:
            for mod in args.modules:
                output.append('module load '+mod+'\n')
        if args.jcommand:
            for com in args.jcommand:
                output.append(com+'\n')
        if args.qccode and args.qccode in 'terachem TeraChem TERACHEM tc TC Terachem':
            tc = False
            if args.jcommand:
                for jc in args.jcommand:
                    if 'terachem' in jc:
                        tc = True
            if not tc:
                output.append('terachem terachem_input > tc.out\n')
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
                output.append('rungms gam.inp '+cpus + ' > gam.out\n')
            elif not qch and 'qch' in args.qccode.lower():
                output.append('qchem qch.inp '+cpus + ' > qch.out\n')
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
                output.append('orca orca.in > orca.out\n')
            elif not molc and 'molc' in args.qccode.lower():
                output.append('pymolcas molcas.input -f\n')
        else:
            print(
                'No supported QC code requested. Please input execution command manually')
        with open(job+'/'+'jobscript', 'w') as f:
            f.writelines(output)
