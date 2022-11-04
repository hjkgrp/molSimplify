# @file rungen.py
#  Top level script that coordinates generation of all files
#
#  Written by Tim Ioannidis for HJK Group
#
#  Dpt of Chemical Engineering, MIT

import itertools
import os
import random
import shutil
import sys
from collections import Counter
import glob

from molSimplify.Scripts.isomers import generateisomers
from molSimplify.Scripts.jobgen import (sgejobgen,
                                        slurmjobgen)
from molSimplify.Scripts.io import (core_load,
                                    getlicores,
                                    lig_load,
                                    name_complex,
                                    name_ts_complex,
                                    substr_load)
from molSimplify.Scripts.qcgen import (mlpgen,
                                       multigamgen,
                                       multimolcgen,
                                       multiogen,
                                       multiqgen,
                                       multitcgen)
from molSimplify.Scripts.structgen import (structgen)

###################################################################
### define input for cross-compatibility between python 2 and 3 ###
###################################################################
get_input = input
if sys.version_info[:2] <= (2, 7):
    get_input = raw_input  # noqa: F821

#####################################
### constrained random generation ###
#####################################


def constrgen(rundir, args, globs):
    emsg = False
    # load global variables
    licores = getlicores()
    print('Random generation started..\n\n')
    # if ligand constraint apply it now
    ligs0 = []
    ligocc0 = []
    coord = False if not args.coord else int(args.coord)
    if args.gui:
        args.gui.iWtxt.setText('\n----------------------------------------------------------------------------------\n' +
                               'Random generation started\nGenerating ligand combinations.\n\n'+args.gui.iWtxt.toPlainText())
        args.gui.app.processEvents()
    if args.lig:
        for i, li in enumerate(args.lig):
            ligs0.append(li)
            ligentry, emsg = lig_load(li)  # check ligand
            # update ligand
            if ligentry:
                args.lig[i] = ligentry.name
            if emsg:
                return False, emsg
            if args.ligocc:
                if len(args.ligocc) < i and len(args.lig) == 1:
                    args.ligocc.append(coord)
                elif len(args.ligocc) < i:
                    args.ligocc.append(1)
            else:
                args.ligocc = []
                if len(args.lig) == 1:
                    args.ligocc.append(coord)
                else:
                    args.ligocc.append(1)
            ligocc0.append(args.ligocc[i])
            if args.lignum:
                args.lignum = str(int(args.lignum) - 1)
            # check for smiles
            if not ligentry.denticity:
                if args.smicat and len(args.smicat) >= i and args.smicat[i]:
                    ligentry.denticity = len(args.smicat[i])
                else:
                    ligentry.denticity = 1
            if coord:
                coord -= int(args.ligocc[i])*ligentry.denticity
            licores.pop(li, None)  # remove from dictionary
    # check for ligand groups
    licoresnew = dict()
    if args.liggrp and 'all' != args.liggrp.lower():
        for key in list(licores.keys()):
            if args.liggrp.lower() in licores[key][3]:
                if not args.ligctg or args.ligctg.lower() == 'all':
                    licoresnew[key] = licores[key]
                elif args.ligctg and args.ligctg.lower() in licores[key][3]:
                    licoresnew[key] = licores[key]
        licores = licoresnew
    # remove aminoacids
    licoresnew = dict()
    for key in list(licores.keys()):
        if 'aminoacid' not in licores[key][3]:
            licoresnew[key] = licores[key]
    licores = licoresnew
    # get a sample of these combinations
    samps = getconstsample(int(args.rgen[0]), args, licores, coord)
    if len(samps) == 0:
        if coord == 0:
            args.lig = [a for a in ligs0]
            args.ligocc = [int(a) for a in ligocc0]
            # run structure generation
            emsg = rungen(rundir, args, False, globs)
        else:
            if args.gui:
                from molSimplify.Classes.mWidgets import mQDialogErr
                qqb = mQDialogErr(
                    'Error', 'No suitable ligand sets were found for random generation. Exiting...')
                qqb.setParent(args.gui.wmain)
            else:
                emsg = 'No suitable ligand sets were found for random generation. Exiting...'
                print(
                    'No suitable ligand sets were found for random generation. Exiting...\n\n')
            return args, emsg
    # loop over samples
    for combo in samps:
        args.lig = [a for a in ligs0]
        args.ligocc = [int(a) for a in ligocc0]
        for cj in set(combo):
            lcount = Counter(combo)
            rocc = lcount[cj]
            args.lig.append(list(licores.keys())[cj])
            args.ligocc.append(rocc)
        # check for keep Hydrogens
        for iiH in range(len(ligs0), len(args.lig)):
            opt = True if args.rkHs else False
            if args.keepHs and len(args.keepHs) > iiH:
                args.keepHs[iiH] = opt
            elif args.keepHs:
                args.keepHs.append(opt)
            else:
                args.keepHs = [opt]
        emsg = rungen(rundir, args, False, globs)  # run structure generation
    return args, emsg


###############################################
### get sample aggreeing to the constraints ###
###############################################


def getconstsample(no_rgen, args, licores, coord):
    samp = []
    # 4 types of constraints: ligand, ligocc, coord, lignum
    # get ligand and ligocc
    combos = []
    generated = 0
    if not coord:
        coord = 6  # default octahedral
    # generate all combinations of ligands
    combos += (list(itertools.combinations_with_replacement(list(range(0, len(licores))), coord)))
    random.shuffle(combos)
    for combo in combos:
        # get total denticity
        totdent = 0
        dents = []
        for li in combo:
            totdent += int(len(licores[list(licores.keys())[li]][2]))
            dents.append(int(len(licores[list(licores.keys())[li]][2])))
        # check for multiple multidentate ligands
        dsorted = sorted(dents)
        if not coord or (coord and totdent == coord):
            if not (len(dsorted) > 1 and (dsorted[-1]+dsorted[-2] > totdent)):
                if (args.lignum and len(set(combo)) == int(args.lignum)):
                    # reorder with high denticity atoms in the beginning
                    keysl = sorted(list(range(len(dents))),
                                   key=lambda k: dents[k])
                    ncombo = [combo[i] for i in keysl]
                    # add combo
                    samp.append(ncombo)
                    generated += 1
                elif not args.lignum:
                    # reorder with high denticity atoms in the beginning
                    keysl = sorted(list(range(len(dents))),
                                   key=lambda k: dents[k])
                    ncombo = [combo[i] for i in keysl]
                    # add combo
                    samp.append(ncombo)
                    generated += 1
            if (generated >= no_rgen):
                break
    return samp

# Generates multiple runs for different oxidation and spin states
#  @param rundir Run directory
#  @param args Namespace of arguments
#  @param globs Global variables
#  @return Error messages


def multigenruns(rundir, args, globs, write_files=True):
    emsg = False
    args.jid = 0  # initilize global name identifier
    multch = False
    multsp = False
    charges = args.charge
    spins = args.spin
    # check if multiple charges specified
    if args.charge and len(args.charge) > 1:
        multch = True
    # check if multiple spin states specified
    if (args.spin and len(args.spin) > 1):
        multsp = True
    # iterate over all
    fname = False
    if (multch and multsp):
        for ch in charges:
            for sp in spins:
                args.charge = ch
                args.spin = sp
                if ch[0] == '-':
                    fname = 'N'+ch[1:]+'S'+sp
                else:
                    fname = 'P'+ch+'S'+sp
                # if args.tsgen:
                #     emsg = tsgen_supervisor(rundir,args,fname,globs)
                # else:
                emsg = rungen(rundir, args, fname, globs, write_files=write_files)
                if emsg:
                    return emsg
    elif (multch):
        for ch in charges:
            args.charge = ch
            if (args.spin):
                args.spin = args.spin[0]
            if ch[0] == '-':
                fname = 'N'+ch[1:]
            else:
                fname = 'P'+ch[1:]
            # if args.tsgen:
            #     emsg = tsgen_supervisor(rundir,args,fname,globs)
            # else:
            emsg = rungen(rundir, args, fname, globs, write_files=write_files)
            if emsg:
                return emsg
    elif (multsp):
        if args.charge:
            args.charge = args.charge[0]
        for sp in spins:
            args.spin = sp
            fname = 'S'+sp
            # if args.tsgen:
            #     emsg = tsgen_supervisor(rundir,args,fname,globs)
            # else:
            emsg = rungen(rundir, args, fname, globs, write_files=write_files)
            if emsg:
                return emsg
    elif args.isomers or args.stereos:
        isomers = generateisomers(args)
        if args.charge:
            args.charge = args.charge[0]
        if args.spin:
            args.spin = args.spin[0]
        args.ligloc = True

        # store information about the overall ligand complex
        keepHs_dict = {}
        ligocc_dict = {}
        for counter, i in enumerate(args.lig):
            keepHs_dict[i] = args.keepHs[counter]
            ligocc_dict[i] = args.ligocc[counter]

        for counter, isomer in enumerate(isomers):
            # set args for the generation of a specific isomer
            args.lig = isomer
            if args.stereos:
                if (counter+1) % 2 == 0:
                    unique_name = '_stereo_'+str(int(counter/2+1))
                else:
                    unique_name = '_isomer_'+str(int(counter/2+1))
            else:
                unique_name = '_isomer_'+str(counter+1)
            args.ligocc = [1]*len(isomer)
            args.keepHs = []
            args.name = str(args.core)+'_ox_' + \
                str(args.oxstate)+'_spin_'+str(args.spin)
            for ligand in list(keepHs_dict.keys()):
                args.name += '_'+ligand+'_'+str(ligocc_dict[ligand])
            args.name += unique_name
            for ligand in args.lig:
                if ligand.endswith('_flipped'):
                    ligand = ligand[:-8]
                args.keepHs.append(keepHs_dict[ligand])

            print('**************************************************************')
            print(('******************* Generating isomer ' +
                  str(counter+1) + '! *******************'))
            print('**************************************************************')
            emsg = rungen(rundir, args, fname, globs, write_files=write_files)
        return emsg
    else:
        if args.charge:
            args.charge = args.charge[0]
        if args.spin:
            args.spin = args.spin[0]
        # if args.tsgen:
        #     emsg = tsgen_supervisor(rundir,args,fname,globs)
        # else:
        emsg = rungen(rundir, args, fname, globs, write_files=write_files)
    return emsg

# Check for multiple ligands specified in one file
#  @param ligs List of ligands
#  @return Ligand list, connecting atoms, multiple ligand flag


def checkmultilig(ligs):
    mligs = []
    tcats = []
    multidx = -1
    # loop over ligands

    for i, lig in enumerate(ligs):
        connatoms = []
        if '.smi' in lig:
            if '~' in lig:
                lig = lig.replace('~', os.path.expanduser("~"))
            # read molecule
            if glob.glob(lig):
                print('found ligand file')
                with open(lig, 'r') as f:
                    s = f.read().splitlines()
                for ss in s:
                    ss = ss.replace('\t', ' ')
                    sf = [_f for _f in ss.split(' ') if _f]
                    print(sf)
                    if len(sf) > 0:
                        connatoms.append(sf[-1])
                        multidx = i
                    else:
                        connatoms.append(False)
                if len(s) > 1:
                    mligs.append(s)
                else:
                    mligs.append([lig])
            else:
                mligs.append([lig])
        else:
            mligs.append([lig])
        tcats.append(connatoms)
    ligandslist = list(itertools.product(*mligs))
    # convert tuple to list
    llist = []
    for l0 in ligandslist:
        loclist = []
        if len(l0) > 0:
            for l1 in l0:
                loclist.append(l1)
            llist.append(loclist)

    return llist, tcats, multidx

# Draw mode supervisor
#  @param args Namespace of arguments
#  @param rundir Run directory


def draw_supervisor(args, rundir):
    if args.lig:
        print('Due to technical limitations, we will draw only the first ligand.')
        print('To view multiple ligands at once, consider using the GUI instead.')
        li = args.lig[0]
        lig, emsg = lig_load(li)
        lig.draw_svg(li)
    elif args.core:
        if len(args.core) > 1:
            print('Due to technical limitations, we will draw only the first core.')
        print('Drawing the core.')
        if args.substrate:
            print('Due to technical limitations, we can draw only one structure per run. To draw the substrate, run the program again.')
        cc, emsg = core_load(args.core[0])
        cc.draw_svg(args.core[0])
    elif args.substrate:
        if len(args.substrate) > 1:
            print('Due to technical limitations, we will draw only the first substrate.')
        print('Drawing the substrate.')
        print((args.substrate[0]))
        substrate, subcatoms, emsg = substr_load(args.substrate[0], 0, args.subcatoms)
        substrate.draw_svg(args.substrate[0])
    else:
        print('You have not specified anything to draw. Currently supported: ligand, core, substrate')

# Normal structure generation
#  @param rundir Run directory
#  @param args Namespace of arguments
#  @param chspfname Folder name for charges and spins
#  @param globs Global variables
#  @return Error messages


def rungen(rundir, args, chspfname, globs, write_files=True):
    try:
        from Classes.mWidgets import mQDialogInf
    except ImportError:
        args.gui = False
    emsg = False
    globs.nosmiles = 0  # reset smiles ligands for each run. This is an index
    # check for specified ligands/functionalization
    ligocc = []
    # check for files specified for multiple ligands
    mligs, catoms = [False], [False]
    if args.lig is not None:
        if '.smi' in args.lig[0]:
            ligfilename = args.lig[0].split('.')[0]
        if args.lig:
            mligs, catoms, multidx = checkmultilig(args.lig)
        if args.debug:
            print(('after checking for multiple ligs, we found  ' +
                   str(multidx) + ' ligands'))
    # save initial
    smicat0 = [ss for ss in args.smicat] if args.smicat else False
    # preparing to store user-provided SMILES ligand names
    SMILES_lig_names = []
    # loop over ligands
    for mcount, mlig in enumerate(mligs):
        args.smicat = [ss for ss in smicat0] if smicat0 else False
        args.checkdir, skip = False, False  # initialize flags
        if len(mligs) > 0 and mligs[0]:
            args.lig = mlig  # get combination
            if multidx != -1:
                if catoms[multidx][mcount]:
                    ssatoms = catoms[multidx][mcount].split(',')
                    lloc = [int(scat)-1 for scat in ssatoms]
                    # append connection atoms if specified in smiles
                    if args.smicat and len(args.smicat) > 0:
                        for i in range(len(args.smicat), multidx):
                            args.smicat.append([])
                    else:
                        args.smicat = [lloc]
                    args.smicat[multidx] = lloc
        if (args.lig):
            ligands = args.lig
            if (args.ligocc):
                ligocc = args.ligocc
            else:
                ligocc = ['1']
            for i in range(len(ligocc), len(ligands)):
                ligocc.append('1')
            lig = ''
            for i, l in enumerate(ligands):
                ligentry, emsg = lig_load(l)
                # update ligand
                if ligentry:
                    ligands[i] = ligentry.name
                    args.lig[i] = ligentry.name
                if emsg:
                    skip = True
                    break
                if ligentry.ident == 'smi':
                    ligentry.ident += str(globs.nosmiles)
                    globs.nosmiles += 1
                    if args.sminame:
                        if len(args.sminame) > int(ligentry.ident[-1]):                            
                            ligentry.ident = args.sminame[globs.nosmiles-1]
                    # lig += ''.join("%s%s" % (ligentry.ident, ligocc[i])) # <User-provided name for SMILES ligand><number of occurrences ofligand>
                    SMILES_lig_names.append(ligentry.ident) # Appending the user-provided name for the SMILES ligand to the list.
                    # The user-provided name affects folder names and file names down the line.
        else:
            ligands = []
            ligocc = ''
    # fetch smart name
        fname = name_complex(rundir, args.core, args.geometry, ligands, ligocc,
                             mcount, args, nconf=False, sanity=False, bind=args.bind, bsmi=args.nambsmi, SMILES_lig_names=SMILES_lig_names)
        if args.tsgen:
            substrate = args.substrate
            subcatoms = ['multiple']
            if args.subcatoms:
                subcatoms = args.subcatoms
            mlig = args.mlig
            mligcatoms = args.mligcatoms
            fname = name_ts_complex(rundir, args.core, args.geometry, ligands, ligocc, substrate, subcatoms,
                                    mlig, mligcatoms, mcount, args, nconf=False, sanity=False, bind=args.bind, bsmi=args.nambsmi)
        if globs.debug:
            print(('fname is ' + str(fname)))
        rootdir = fname
        # check for charges/spin
        rootcheck = False
        if (chspfname):
            rootcheck = rootdir
            rootdir = rootdir + '/'+chspfname
        if (args.suff):
            rootdir += args.suff
        # check for mannual overwrite of
        # directory name
        if args.jobdir:
            rootdir = rundir + args.jobdir
            # check for top directory
        if rootcheck and os.path.isdir(rootcheck) and not args.checkdirt and not skip:
            args.checkdirt = True
            if not args.rprompt:
                flagdir = get_input('\nDirectory ' + rootcheck +
                                    ' already exists. Keep both (k), replace (r) or skip (s) k/r/s: ')
                if 'k' in flagdir.lower():
                    flagdir = 'keep'
                elif 's' in flagdir.lower():
                    flagdir = 'skip'
                else:
                    flagdir = 'replace'
            else:
                # qqb = qBoxFolder(args.gui.wmain,'Folder exists','Directory '+rootcheck+' already exists. What do you want to do?')
                # flagdir = qqb.getaction()
                flagdir = 'replace'
                # replace existing directory
            if (flagdir == 'replace'):
                shutil.rmtree(rootcheck)
                os.mkdir(rootcheck)
            # skip existing directory
            elif flagdir == 'skip':
                skip = True
            # keep both (default)
            else:
                ifold = 1
                while glob.glob(rootdir+'_'+str(ifold)):
                    ifold += 1
                    rootcheck += '_'+str(ifold)
                    os.mkdir(rootcheck)
        elif rootcheck and (not os.path.isdir(rootcheck) or not args.checkdirt) and not skip:
            if globs.debug:
                print(('rootcheck is  ' + str(rootcheck)))
            args.checkdirt = True
            try:
                os.mkdir(rootcheck)
            except FileExistsError:
                print(('Directory '+rootcheck+' can not be created. Exiting..\n'))
                return
            # check for actual directory
        if os.path.isdir(rootdir) and not args.checkdirb and not skip and not args.jobdir:
            args.checkdirb = True
            if not args.rprompt:
                flagdir = get_input(
                    '\nDirectory '+rootdir + ' already exists. Keep both (k), replace (r) or skip (s) k/r/s: ')
                if 'k' in flagdir.lower():
                    flagdir = 'keep'
                elif 's' in flagdir.lower():
                    flagdir = 'skip'
                else:
                    flagdir = 'replace'
            else:
                # qqb = qBoxFolder(args.gui.wmain,'Folder exists','Directory '+rootdir+' already exists. What do you want to do?')
                # flagdir = qqb.getaction()
                flagdir = 'replace'
            # replace existing directory
            if (flagdir == 'replace'):
                shutil.rmtree(rootdir)
                os.mkdir(rootdir)
            # skip existing directory
            elif flagdir == 'skip':
                skip = True
            # keep both (default)
            else:
                ifold = 1
                while glob.glob(rootdir+'_'+str(ifold)):
                    ifold += 1
                rootdir += '_'+str(ifold)
                os.mkdir(rootdir)
        elif not os.path.isdir(rootdir) or not args.checkdirb and not skip:
            if not os.path.isdir(rootdir):
                if write_files:
                    args.checkdirb = True
                    os.mkdir(rootdir)
            ####################################
            ############ GENERATION ############
            ####################################
        if not skip:
            # check for generate all
            if args.genall:
                tstrfiles = []
                # generate xyz with FF and trained ML
                args.ff = 'mmff94'
                args.ffoption = 'ba'
                args.MLbonds = False
                strfiles, emsg, this_diag = structgen(
                    args, rootdir, ligands, ligocc, globs, mcount, write_files=write_files, SMILES_lig_names=SMILES_lig_names)
                for strf in strfiles:
                    tstrfiles.append(strf+'FFML')
                    os.rename(strf+'.xyz', strf+'FFML.xyz')
                # generate xyz with FF and covalent
                args.MLbonds = ['c' for i in range(0, len(args.lig))]
                strfiles, emsg, this_diag = structgen(
                    args, rootdir, ligands, ligocc, globs, mcount, write_files=write_files, SMILES_lig_names=SMILES_lig_names)
                for strf in strfiles:
                    tstrfiles.append(strf+'FFc')
                    os.rename(strf+'.xyz', strf+'FFc.xyz')
                args.ff = False
                args.ffoption = False
                args.MLbonds = False
                # generate xyz without FF and trained ML
                strfiles, emsg, this_diag = structgen(
                    args, rootdir, ligands, ligocc, globs, mcount, write_files=write_files, SMILES_lig_names=SMILES_lig_names)
                for strf in strfiles:
                    tstrfiles.append(strf+'ML')
                    os.rename(strf+'.xyz', strf+'ML.xyz')
                args.MLbonds = ['c' for i in range(0, len(args.lig))]
                # generate xyz without FF and covalent ML
                strfiles, emsg, this_diag = structgen(
                    args, rootdir, ligands, ligocc, globs, mcount, write_files=write_files, SMILES_lig_names=SMILES_lig_names)
                for strf in strfiles:
                    tstrfiles.append(strf+'c')
                    os.rename(strf+'.xyz', strf+'c.xyz')
                strfiles = tstrfiles
            else:
                # generate xyz files
                strfiles, emsg, this_diag = structgen(
                    args, rootdir, ligands, ligocc, globs, mcount, write_files=write_files, SMILES_lig_names=SMILES_lig_names)
            # generate QC input files
            if args.qccode and (not emsg) and write_files:
                if args.charge and (isinstance(args.charge, list)):
                    args.charge = args.charge[0]
                if args.spin and (isinstance(args.spin, list)):
                    args.spin = args.spin[0]
                if args.qccode.lower() in 'terachem tc Terachem TeraChem TERACHEM TC':
                    jobdirs = multitcgen(args, strfiles)
                    print('TeraChem input files generated!')
                elif 'gam' in args.qccode.lower():
                    jobdirs = multigamgen(args, strfiles)
                    print('GAMESS input files generated!')
                elif 'qch' in args.qccode.lower():
                    jobdirs = multiqgen(args, strfiles)
                    print('QChem input files generated!')
                elif 'orc' in args.qccode.lower():
                    jobdirs = multiogen(args, strfiles)
                    print('ORCA input files generated!')
                elif 'molc' in args.qccode.lower():
                    jobdirs = multimolcgen(args, strfiles)
                    print('MOLCAS input files generated!')
                else:
                    print(
                        'Only TeraChem, GAMESS, QChem, ORCA, MOLCAS are supported right now.\n')
            # check molpac
            if args.mopac and (not emsg) and write_files:
                print('Generating MOPAC input')
                if globs.debug:
                    print(strfiles)
                jobdirs = mlpgen(args, strfiles, rootdir)
            # generate jobscripts
            if args.jsched and (not emsg) and (not args.reportonly) and (write_files):
                if args.jsched in 'SBATCH SLURM slurm sbatch':
                    slurmjobgen(args, jobdirs)
                    print('SLURM jobscripts generated!')
                elif args.jsched in 'SGE Sungrid sge':
                    sgejobgen(args, jobdirs)
                    print('SGE jobscripts generated!')

            elif multidx != -1:  # if ligand input was a list of smiles strings, write good smiles strings to separate list
                with open(ligfilename+'-good.smi', 'a') as f:
                    f.write(args.lig[0])
        elif not emsg:
            if args.gui:
                qq = mQDialogInf('Folder skipped', 'Folder ' +
                                 rootdir+' was skipped.')
                qq.setParent(args.gui.wmain)
            else:
                print(('Folder '+rootdir+' was skipped..\n'))
    if write_files:
        return emsg  # Default behavior
    else:
        return strfiles, emsg, this_diag  # Assume that user wants these if they're not writing files
