# Written by Tim Ioannidis for HJK Group
# Dpt of Chemical Engineering, MIT

##############################################################
########## This script processes the input file  #############
##############################################################

# import std modules
import glob, os, re, argparse, sys
from io import *
from Classes.globalvars import *

######################################################
########## check core/ligands specified  #############
######################################################
### checks input for correctness ###
def checkinput(args):
    emsg = False
    # check if core is specified
    if not (args.core):
            emsg = 'You need to specify at least the core of the structures.\n'
            print emsg
            return emsg
    # check if ligands are specified
    if not args.lig and not args.rgen:
        if args.gui:
            from Classes.mWidgets import mQDialogWarn
            qqb = mQDialogWarn('Warning','You specified no ligands. Please use the -lig flag. Core only generation..')
            qqb.setParent(args.gui.wmain)
        else:
            print 'WARNING: You specified no ligands. Please use the -lig flag. Forced generation..\n'
    return emsg

###########################################
########## check true or false  ###########
###########################################
def checkTrue(arg):
    if 'y' in arg.lower() or '1' in arg.lower() or 't' in arg.lower() or arg==1:
        return True
    else:
        return False
                
########################################
########## check for number  ###########
########################################
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

###########################################
########## consolidate lists  #############
###########################################
### consolidate arguments ###
def cleaninput(args):
    globs = globalvars()
    # check ligands
    if args.lig:
        ls = []
        ligdic = readdict(globs.installdir+'/Ligands/simple_ligands.dict')
        for i,s in enumerate(args.lig):
            if isinstance(s,list):
                for ss in s:
                    if ss in ligdic.keys():
                        ss = ligdic[ss][0]
                    ls.append(ss)
            else:
                if s in ligdic.keys():
                        s = ligdic[s][0]
                ls.append(s)
        args.lig = ls
    # check sminame
    if args.sminame:
        ls = []
        for i,s in enumerate(args.sminame):
            if isinstance(s,list):
                for ss in s:
                    ls.append(ss)
            else:
                ls.append(s)
        args.sminame = ls
    # check qoption
    if args.qoption:
        ls = []
        for i,s in enumerate(args.qoption):
            if isinstance(s,list):
                for ss in s:
                    ls.append(ss)
            else:
                ls.append(s)
        args.qoption = ls
    # check sysoption
    if args.sysoption:
        ls = []
        for i,s in enumerate(args.sysoption):
            if isinstance(s,list):
                for ss in s:
                    ls.append(ss)
            else:
                ls.append(s)
        args.sysoption = ls
    # check ctrloption
    if args.ctrloption:
        ls = []
        for i,s in enumerate(args.ctrloption):
            if isinstance(s,list):
                for ss in s:
                    ls.append(ss)
            else:
                ls.append(s)
        args.ctrloption = ls
    # check scfoption
    if args.scfoption:
        ls = []
        for i,s in enumerate(args.scfoption):
            if isinstance(s,list):
                for ss in s:
                    ls.append(ss)
            else:
                ls.append(s)
        args.scfoption = ls
    # check statoption
    if args.statoption:
        ls = []
        for i,s in enumerate(args.statoption):
            if isinstance(s,list):
                for ss in s:
                    ls.append(ss)
            else:
                ls.append(s)
        args.statoption = ls
    # check remoption
    if args.remoption:
        ls = []
        for i,s in enumerate(args.remoption):
            if isinstance(s,list):
                for ss in s:
                    ls.append(ss)
            else:
                ls.append(s)
        args.remoption = ls
    # convert keepHs to boolean
    if args.keepHs:
        for i,s in enumerate(args.keepHs):
            args.keepHs[i]=checkTrue(s)
    # convert ff option to abe code
    if args.ff and args.ffoption:
        b = False
        a = False
        e = False
        opts = args.ffoption
        if 'ba' in opts[0].lower():
            args.ffoption = 'ba'
        else:
            args.ffoption = ''
            for op in opts:
                op = op.strip(' ')
                if op[0].lower()=='b':
                    args.ffoption += 'b'
                if op[0].lower()=='a':
                    args.ffoption += 'a'
    elif args.ff:
        args.ffoption = 'ba'

###################################################
##########  parse command line input  #############
###################################################
### parses inputfile ###
def parseCLI(args):
    cliargs = ' '.join(args)
    s = filter(None,cliargs.split('-'))
    fname = 'CLIinput.inp'
    f = open(fname,'w')
    f.write('# molSimplify input file generated from CLI input\n')
    for line in s:
       f.write('-'+line+'\n')
    f.close()
    return fname
    
###########################################
##########  parse input file  #############
###########################################
### parses inputfile ###
def parseinput(args):
    for line in open(args.i):
        if '-lig' not in line and '-core' not in line and '-bind' not in line:
            line = line.split('#')[0] # remove comments
        li = line.strip()
        li = li.replace('\n','')
        line = line.replace('\n','')
        if not li.startswith("#") and len(li)>0: # remove comments/empty lines
            l = filter(None,re.split(' |,|\t|&',li))
            # parse general arguments
            if (l[0]=='-core' and len(l[1:]) > 0):
                args.core = [ll for ll in l[1:]]
            if (l[0]=='-ccatoms' and len(l[1:]) > 0):
                args.ccatoms = [int(ll)-1 for ll in l[1:]]
            if (l[0]=='-rundir'):
                args.rundir = line.split("#")[0].strip('\n')
                args.rundir = args.rundir.split('-rundir')[1]
                args.rundir = args.rundir.lstrip(' ')
                if (args.rundir[-1]=='/'):
                    args.rundir = args.rundir[:-1]
            if (l[0]=='-suff'):
                args.suff = l[1].strip('\n')
            ### parse structure generation arguments ###
            if (l[0]=='-bind' and len(l[1:]) > 0):
                l = filter(None,re.split(' |,|\t',line))
                # discard comments
                for ibind,lbind in enumerate(l):
                    if lbind=='#':
                        l=l[:ibind]
                        break
                args.bind = l[1]
            if (l[0]=='-nbind' and len(l[1:]) > 0):
                args.bindnum = l[1]
            if (l[0]=='-bcharge' and len(l[1:]) > 0):  # parse charge for binding species
                args.bcharge = l[1]
            if (l[0]=='-btheta' and len(l[1:]) > 0):
                args.btheta = l[1]
            if (l[0]=='-bphi' and len(l[1:]) > 0):
                args.bphi = l[1]
            if (l[0]=='-bsep' and len(l[1:]) > 0):
                args.bsep = l[1]
            if (l[0]=='-bref' and len(l[1:]) > 0):
                args.bref = l[1:]
            if (l[0]=='-nambsmi' and len(l[1:]) > 0):
                args.nambsmi = l[1]
            if (l[0]=='-maxd' and len(l[1:]) > 0):
                args.maxd = l[1]
            if (l[0]=='-mind' and len(l[1:]) > 0):
                args.mind = l[1]
            if (l[0]=='-oxstate' and len(l[1:]) > 0):
                args.oxstate = l[1]
            if (l[0]=='-coord' and len(l[1:]) > 0):
                args.coord = l[1]
            if (l[0]=='-geometry' and len(l[1:]) > 0):
                args.geometry = l[1].lower()
            # parse ligands
            if (l[0]=='-lig'):
                ll = line.split('-lig')[-1]
                args.lig = filter(None,re.split(' |,|\t',ll))
            if (l[0]=='-lignum' and len(l[1:]) > 0):
                args.lignum = l[1]
            if (l[0]=='-liggrp' and len(l[1:]) > 0):
                args.liggrp = l[1]
            if (l[0]=='-ligctg' and len(l[1:]) > 0):
                args.ligctg = l[1]
            if (l[0]=='-ligocc' and len(l[1:]) > 0):
                args.ligocc = l[1:]
            if (l[0]=='-rkHs'):
                args.rkHs = checkTrue(l[1])
            if (l[0]=='-ligloc'):
                args.ligloc = checkTrue(l[1])
            if (l[0]=='-ligalign'):
                args.ligalign = checkTrue(l[1])
            if (l[0]=='-replig'):
                args.replig = checkTrue(l[1])
            if (l[0]=='-genall'):
                args.genall = checkTrue(l[1])
            if (l[0]=='-MLbonds' and len(l[1:]) > 0):
                args.MLbonds = l[1:]
            if (l[0]=='-distort' and len(l[1:]) > 0):
                args.distort = l[1]
            if (l[0]=='-rgen' and len(l[1:]) > 0):
                args.rgen = l[1:]
            if (l[0]=='-keepHs' and len(l[1:]) > 0):
                args.keepHs = l[1:]
            if (l[0]=='-ff'):
                args.ff = l[1].lower()
            if (l[0]=='-ffoption' and len(l[1:]) > 0):
                args.ffoption = l[1:]
            if (l[0]=='-place' and len(l[1:]) > 0):
                args.place = l[1]
            if (l[0]=='-sminame' and len(l[1:]) > 0):
                if args.sminame:
                    args.sminame.append(l[1:])
                else:
                    args.sminame = l[1:]
            if '-smicat' in line:
                args.smicat = []
                l = line.split('smicat',1)[1]
                l = l.replace(' ','')
                l = re.split('/|\t|&',l)
                for ll in l:
                    lloc = []
                    l1 = filter(None,re.split(',| ',ll))
                    for lll in l1:
                        if lll.lower()!='cm':
                            lloc.append(int(lll)-1)
                        else:
                            lloc.append(lll.lower())
                    args.smicat.append(lloc)
            if '-pangles' in line:
                args.pangles = []
                l = filter(None,line.split('pangles',1)[1])
                l = l.replace(' ','')
                l = re.split(',|\t|&|\n',l)
                for ll in l:
                    args.pangles.append(ll) if ll!='' else args.pangles.append(False)
            # parse qc arguments
            if (l[0]=='-qccode' and len(l[1:]) > 0):
                args.qccode = l[1]
            if (l[0]=='-calccharge'):
                args.calccharge = checkTrue(l[1])
            if (l[0]=='-charge' and len(l[1:]) > 0):
                args.charge = l[1:]
            if (l[0]=='-spin' and len(l[1:]) > 0):
                args.spin = l[1:]
            if (l[0]=='-runtyp' and len(l[1:]) > 0):
                args.runtyp = l[1]
            if (l[0]=='-method' and len(l[1:]) > 0):
                args.method = l[1:]
            # parse terachem arguments
            if (l[0]=='-basis'):
                args.basis = l[1]
            if (l[0]=='-dispersion'):
                args.dispersion = l[1].strip('\n').lower()
            if (l[0]=='-qoption'):
                if args.qoption:
                    args.qoption.append(l[1:])
                else:
                    args.qoption = l[1:]
            # parse qchem arguments
            if (l[0]=='-exchange'):
                args.exchange = l[1]
            if (l[0]=='-correlation'):
                args.correlation = l[1]
            if (l[0]=='-unrestricted'):
                args.unrestricted = checkTrue(l[1])
            if (l[0]=='-remoption'):
                if args.remoption:
                    args.remoption.append(l[1:])
                else:
                    args.remoption = l[1:]
            # parse gamess arguments
            if (l[0]=='-gbasis'):
                args.gbasis = l[1]
            if (l[0]=='-ngauss'):
                args.ngauss = l[1]
            if (l[0]=='-npfunc'):
                args.npfunc = l[1]
            if (l[0]=='-ndfunc'):
                args.ndfunc = l[1]
            if (l[0]=='-sysoption'):
                if args.sysoption:
                    args.sysoption.append(l[1:])
                else:
                    args.sysoption = l[1:]
            if (l[0]=='-ctrloption'):
                if args.ctrloption:
                    args.ctrloption.append(l[1:])
                else:
                    args.ctrloption = l[1:]
            if (l[0]=='-scfoption'):
                if args.scfoption:
                    args.scfoption.append(l[1:])
                else:
                    args.scfoption = l[1:]
            if (l[0]=='-statoption'):
                if args.statoption:
                    args.statoption.append(l[1:])
                else:
                    args.statoption = l[1:]
            # parse jobscript arguments
            if (l[0]=='-jsched'):
                args.jsched = l[1]
            if (l[0]=='-jname'):
                args.jname = l[1]
            if (l[0]=='-memory'):
                args.memory = l[1]
            if (l[0]=='-wtime'):
                args.wtime = l[1]
            if (l[0]=='-queue'):
                args.queue = l[1]
            if (l[0]=='-gpus'):
                args.gpus = l[1]
            if (l[0]=='-cpus'):
                args.cpus = l[1]
            if (l[0]=='-modules'):
                args.modules = l[1:]
            if (l[0]=='-joption'):
                if not args.joption:
                    args.joption = []
                opts = ''
                for ll in l[1:]:
                    opts += ll+' '
                args.joption.append(opts)
            if (l[0]=='-jcommand'):
                if not args.jcommand:
                    args.jcommand = []
                opts = ''
                for ll in l[1:]:
                    opts += ll+' '
                args.jcommand.append(opts)
            # parse database arguments
            if (l[0]=='-dbsim'):
                args.dbsearch = True
                args.dbsim = l[1]
            if (l[0]=='-dbresults'):
                args.dbresults = l[1]
            if (l[0]=='-dboutputf'):
                args.dboutputf = l[1]
            if (l[0]=='-dbbase'):
                args.dbbase = l[1]
            if (l[0]=='-dbsmarts'):
                args.dbsearch = True
                args.dbsmarts = l[1]
            if (l[0]=='-dbcatoms'):
                args.dbcatoms = l[1:]
            if (l[0]=='-dbfinger'):
                args.dbfinger = l[1]
            if (l[0]=='-dbatoms'):
                args.dbatoms = l[1]
            if (l[0]=='-dbbonds'):
                args.dbbonds = l[1]
            if (l[0]=='-dbarbonds'):
                args.dbarbonds = l[1]
            if (l[0]=='-dbsbonds'):
                args.dbsbonds = l[1]
            if (l[0]=='-dbmw'):
                args.dbmw = l[1]
            # parse postprocessing arguments
            if (l[0]=='-postp'):
                args.postp = True
            if (l[0]=='-postqc'):
                args.postqc = l[1]
            if (l[0]=='-postdir'):
                args.postdir = line.split("#")[0].strip('\n')
                args.postdir = args.postdir.split('-postdir')[1]
                args.postdir = args.postdir.lstrip(' ')
            if (l[0]=='-pres'):
                args.pres = True
            if (l[0]=='-pwfninfo'):
                args.pwfninfo = True
            if (l[0]=='-pcharge'):
                args.pcharge = True
            if (l[0]=='-pgencubes'):
                args.pgencubes = True
            if (l[0]=='-porbinfo'):
                args.porbinfo = True
            if (l[0]=='-pdeloc'):
                args.pdeloc = True
            #if (l[0]=='-pdorbs'):
            #    args.pdorbs = True
            if (l[0]=='-pnbo'):
                args.pnbo = True
            # parse slab building arguments
            if (l[0]=='-slab_gen'): #0
               args.slab_gen = True
            if (l[0]=='-unit_cell'): #1 
                args.unit_cell = l[1]
            if (l[0]=='-cell_vector'): #2
                args.cell_index = [[float(j.strip('(){}<>[],.')) for j in i]  for i in l[1:]] # list-of-lists
            if (l[0]=='-cif_path'): #3
                args.cif_path = l[1]
            if (l[0]=='-duplication_vector'): #4
                args.duplication_vector = [int(i.strip('(){}<>[],.')) for i in l[1:]]
            if (l[0]=='-slab_size'): #5
                 args.slab_size = [float(i.strip('(){}<>[],.')) for i in l[1:]]
            if (l[0]=='-miller_index'): #6
                args.miller_index = [int(i.strip('(){}<>[],.')) for i in l[1:]]
            # parse place on slab options
            if (l[0]=='-place_on_slab'): #0
                args.place_on_slab = True
            if (l[0]=='-target_molecule'): #1 
                args.target_molecule = l[1]
            if (l[0]=='-align_distance_method'): #2
                args.align_distance_method = l[1]
            if (l[0]=='-align_dist'): #3
                args.align_dist = float(l[1].strip('(){}<>[],.'))
            if (l[0]=='-align_method'): #4
                args.align_method = l[1]
            if (l[0]=='-object_align'): #5
                globs = globalvars()
                elements = globs.elementsbynum()
                if l[1] in elements:
                    args.object_align = l[1]
                else:
                    args.object_align = [int(i.strip('(){}<>[],.')) for i in l[1:]]
            if (l[0]=='-surface_atom_type'):#6
                args.surface_atom_type = l[1]
            if (l[0] == '-num_surface_atoms'): #7
                args.num_surface_atoms = int(l[1].strip('()[]<>.'))
            if (l[0] == '-num_placements'): #8
                args.num_placements = int(l[1].strip('()[]<>.'))
            if (l[0]=='-coverage'):#9
                args.coverage = float(l[1].strip('()[]<>.'))
            if (l[0]=='-multi_placement_centering'):#10
                args.multi_placement_centering = float(l[1].strip('()[]<>.'))
            if (l[0]=='-control_angle'):#11
                args.control_angle = float(l[1].strip('()[]<>.'))
            if (l[0] == '-angle_control_partner'): #12
                args.angle_control_partner = int(l[1].strip('()[]<>.'))
            if (l[0] == '-angle_surface_axis'): #13
                args.angle_surface_axis = [float(i.strip('(){}<>[],.')) for i in l[1:]]
            if (l[0] == '-duplicate'):#14
                args.duplicate = l[1]


                
#############################################################
########## mainly for help and listing options  #############
#############################################################
### parses commandline arguments and prints help information ###
def parsecommandline(parser):
    globs = globalvars()
    installdir = globs.installdir+'/'
    # first variable is the flag, second is the variable in the structure. e.g -i, --infile assigns something to args.infile
    parser.add_argument("-i","--i",help="input file")
    # top directory options
    parser.add_argument("-rundir","--rundir",help="directory for jobs",action="store_true")
    parser.add_argument("-suff","--suff", help="suffix for jobs folder.",action="store_true")
    # structure generation options
    parser.add_argument("-ccatoms","--ccatoms", help="core connection atoms indices, indexing starting from 1",action="store_true")
    parser.add_argument("-coord","--coord", help="coordination such as 4,5,6",action="store_true") # coordination e.g. 6 
    parser.add_argument("-core","--core", help="core structure with currently available: "+getcores(installdir),action="store_true") #e.g. ferrocene
    parser.add_argument("-bind","--bind", help="binding species with currently available: "+getbinds(installdir),action="store_true") #e.g. bisulfate, nitrate, perchlorate -> For binding
    parser.add_argument("-bcharge","--bcharge",default='0', help="binding species charge, default 0",action="store_true") 
    parser.add_argument("-bphi","--bphi", help="azimuthal angle phi for binding species, default random between 0 and 180",action="store_true") 
    parser.add_argument("-bref","--bref", help="reference atoms for placement of extra molecules, default COM (center of mass). e.g. 1,5 or 1-5, Fe, COM",action="store_true") 
    parser.add_argument("-bsep","--bsep", help="flag for separating extra molecule in input or xyz file",action="store_true")
    parser.add_argument("-btheta","--btheta", help="polar angle theta for binding species, default random between 0 and 360",action="store_true") 
    parser.add_argument("-geometry","--geometry", help="geometry such as TBP (trigonal bipyramidal)",action="store_true") # geometry
    parser.add_argument("-genall","--genall", help="Generate complex both with and without FF opt.",action="store_true") # geometry
    parser.add_argument("-lig","--lig", help="ligand structure name or SMILES with currently available: "+getligs(installdir),action="store_true") #e.g. acetate (in smilesdict)
    parser.add_argument("-ligocc","--ligocc", help="number of corresponding ligands e.g. 2,2,1",action="store_true") # e.g. 1,2,1
    parser.add_argument("-lignum","--lignum", help="number of ligand types e.g. 2",action="store_true") 
    parser.add_argument("-liggrp","--liggrp", help="ligand group for random generation",action="store_true") 
    parser.add_argument("-ligctg","--ligctg", help="ligand category for random generation",action="store_true") 
    parser.add_argument("-rkHs","--rkHs", help="keep Hydrogens for random generation",action="store_true") 
    parser.add_argument("-ligloc","--ligloc", help="force location of ligands in the structure generation yes/True/1 or no/False/0",action="store_true") 
    parser.add_argument("-ligalign","--ligalign", help="smart alignment of ligands in the structure generation yes/True/1 or no/False/0",action="store_true") 
    parser.add_argument("-MLbonds","--MLbonds", help="custom M-L bond length for corresponding ligand in A e.g. 1.4",action="store_true") 
    parser.add_argument("-distort","--distort", help="randomly distort backbone. Ranges from 0 (no distortion) to 100. e.g. 20",action="store_true") 
    parser.add_argument("-langles","--langles", help="custom angles (polar theta, azimuthal phi) for corresponding ligand in degrees separated by '/' e.g. 20/30,10/20",action="store_true") 
    parser.add_argument("-pangles","--pangles", help="custom angles (polar theta, azimuthal phi) for corresponding connectino points in degrees separated by '/' e.g. 20/30,10/20",action="store_true") 
    parser.add_argument("-nbind","--bindnum", help="number of binding species copies for random placement",action="store_true") #different geometric arrangements for calculating binding energy
    parser.add_argument("-rgen","--rgen", help="number of random generated molecules",action="store_true")
    parser.add_argument("-replig","--replig", help="flag for replacing ligand at specified connection point",action="store_true")
    parser.add_argument("-ff","--ff",help="select force field for FF optimization. Available: MMFF94, UFF, GAFF, Ghemical",action="store_true")
    parser.add_argument("-ffoption","--ffoption",help="select when to perform FF optimization. Options: B(Before),A(After),BA",action="store_true")
    parser.add_argument("-keepHs","--keepHs", help="force keep hydrogens. By default ligands are stripped one hydrogen in order to connect to the core",action="store_true") 
    parser.add_argument("-smicat","--smicat", help="connecting atoms corresponding to smiles. Indexing starts at 1 which is the default value as well",action="store_true")
    parser.add_argument("-sminame","--sminame", help="name for smiles species used in the folder naming. e.g. amm",action="store_true") 
    parser.add_argument("-nambsmi","--nambsmi", help="name of SMILES string for binding species e.g. carbonmonoxide",action="store_true")
    parser.add_argument("-maxd","--maxd", help="maximum distance above cluster size for molecules placement maxdist=size1+size2+maxd", action="store_true")
    parser.add_argument("-mind","--mind", help="minimum distance above cluster size for molecules placement mindist=size1+size2+mind", action="store_true")
    parser.add_argument("-place","--place", help="place binding species relative to core. Takes either angle (0-360) or ax/s for axial side",action="store_true")
    parser.add_argument("-oxstate","--oxstate", help="oxidation state of the metal, used for bond lengths",action="store_true")
    # quantum chemistry options
    parser.add_argument("-qccode","--qccode", help="quantum chemistry code. Choices: TeraChem or GAMESS or QChem",action="store_true") 
    parser.add_argument("-charge","--charge", help="charge for system (default: neutral).",action="store_true")
    parser.add_argument("-calccharge","--calccharge", help="Flag to calculate charge.",action="store_true")
    parser.add_argument("-spin","--spin", help="spin multiplicity for system (default: singlet) e.g. 1",action="store_true")
    parser.add_argument("-runtyp","--runtyp", help="run type. Choices: optimization, energy",action="store_true")
    parser.add_argument("-method","--method", help="electronic structure method. Specify UDFT for unrestricted calculation(default: b3lyp) e.g. ub3lyp",action="store_true")
    # terachem arguments
    parser.add_argument("-basis","--basis", help="basis for terachem or qchem job (default: LACVP* or lanl2dz)",action="store_true")
    parser.add_argument("-dispersion","--dispersion", help="dispersion forces. Default: no e.g. d2,d3",action="store_true")
    parser.add_argument("-qoption","--qoption", help="extra arguments for TeraChem in syntax keyword value, e.g. maxit 100",action="store_true")
    # qchem arguments
    parser.add_argument("-exchange","--exchange",help="exchange in qchem job (default b3lyp)",action="store_true")
    parser.add_argument("-correlation","--correlation",help="correlation in qchem job (default none)",action="store_true")
    parser.add_argument("-remoption","--remoption", help="extra arguments for qchem $rem block in syntax keyword value, e.g. INCFOCK 0",action="store_true")
    parser.add_argument("-unrestricted","--unrestricted", help="unrestricted calculation, values: 0/1 False/True",action="store_true")
    # gamess arguments
    parser.add_argument("-gbasis","--gbasis", help="GBASIS option in GAMESS e.g. CCT",action="store_true")
    parser.add_argument("-ngauss","--ngauss", help="NGAUSS option in GAMESS e.g. N31",action="store_true")
    parser.add_argument("-npfunc","--npfunc", help="NPFUNC option for diffuse functions in GAMESS e.g. 2",action="store_true")
    parser.add_argument("-ndfunc","--ndfunc", help="NDFUNC option for diffuse functions in GAMESS e.g. 1",action="store_true")
    parser.add_argument("-sysoption","--sysoption", help="extra arguments for $SYSTEM GAMESS block in syntax keyword value, e.g. MWORDS 20",action="store_true")
    parser.add_argument("-ctrloption","--ctrloption", help="extra arguments for $CONTRL GAMESS block in syntax keyword value, e.g. ISPHER 1",action="store_true")
    parser.add_argument("-scfoption","--scfoption", help="extra arguments for $SCF GAMESS block in syntax keyword value, e.g. DIIS .TRUE.",action="store_true")
    parser.add_argument("-statoption","--statoption", help="extra arguments for $STATPT GAMESS block in syntax keyword value, e.g. NSTEP 100",action="store_true")
    # jobscript arguments
    parser.add_argument("-jsched","--jsched", help="job scheduling system. Choices: SLURM or SGE",action="store_true")
    parser.add_argument("-jname","--jname", help="jobs main identifier",action="store_true")
    parser.add_argument("-memory","--memory", help="memory reserved per thread for job file in G(default: 2G)e.g.2",action="store_true")
    parser.add_argument("-wtime","--wtime", help="wall time requested in hours for queueing system (default: 168hrs) e.g. 8",action="store_true")
    parser.add_argument("-queue","--queue", help="queue name e.g gpus",action="store_true")
    parser.add_argument("-gpus","--gpus", help="number of GPUS (default: 1)",action="store_true")
    parser.add_argument("-cpus","--cpus", help="number of CPUs (default: 1)",action="store_true")
    parser.add_argument("-modules","--modules", help="modules to be loaded for the calculation",action="store_true")
    parser.add_argument("-joption","--joption", help="additional options for jobscript",action="store_true")
    parser.add_argument("-jcommand","--jcommand", help="additional commands for jobscript",action="store_true")
    # database search arguments
    parser.add_argument("-dbsim","--dbsim", help="SMILES/ligand/file for similarity search",action="store_true")
    parser.add_argument("-dbcatoms","--dbcatoms", help="connection atoms for similarity search",action="store_true")
    parser.add_argument("-dbresults","--dbresults", help="how many results for similary search or screening",action="store_true")
    parser.add_argument("-dboutputf","--dboutputf", help="output file for search results",action="store_true")
    parser.add_argument("-dbbase","--dbbase", help="database for search",action="store_true")
    parser.add_argument("-dbsmarts","--dbsmarts", help="SMARTS string for screening",action="store_true")
    parser.add_argument("-dbfinger","--dbfinger", help="fingerprint for similarity search",action="store_true")
    parser.add_argument("-dbatoms","--dbatoms", help="number of atoms to be used in screening",action="store_true")
    parser.add_argument("-dbbonds","--dbbonds", help="number of bonds to be used in screening",action="store_true")
    parser.add_argument("-dbarbonds","--dbarbonds", help="Number of aromatic bonds to be used in screening",action="store_true")
    parser.add_argument("-dbsbonds","--dbsbonds", help="number of single bonds to be used in screening",action="store_true")
    parser.add_argument("-dbmw","--dbmw", help="molecular weight to be used in screening",action="store_true")
    # post-processing arguments
    parser.add_argument("-postp","--postp",help="post process results",action="store_true")
    parser.add_argument("-postqc","--postqc", help="quantum chemistry code used. Choices: TeraChem or GAMESS",action="store_true") 
    parser.add_argument("-postdir","--postdir", help="directory with results",action="store_true") 
    parser.add_argument("-pres","--pres",help="generate calculations summary",action="store_true")
    parser.add_argument("-pdeninfo","--pdeninfo",help="calculate average properties for electron density",action="store_true")
    parser.add_argument("-pcharge","--pcharge",help="calculate charges",action="store_true")
    parser.add_argument("-pgencubes","--pgencubes",help="generate cubefiles",action="store_true")
    parser.add_argument("-pwfninfo","--pwfninfo",help="get information about wavefunction",action="store_true")
    parser.add_argument("-pdeloc","--pdeloc",help="get delocalization and localization indices",action="store_true")
    parser.add_argument("-porbinfo","--porbinfo",help="get information about MO",action="store_true")
    parser.add_argument("-pnbo","--pnbo",help="post process nbo analysis",action="store_true")
    #parser.add_argument("-pdorbs","--pdorbs",help="get information on metal d orbitals",action="store_true")
    # auxiliary
    parser.add_argument("-dbsearch","--dbsearch",action="store_true") # flag for db search
    parser.add_argument("-checkdirt","--checkdirt",action="store_true") # directory removal check flag
    parser.add_argument("-checkdirb","--checkdirb",action="store_true") # directory removal check flag 2
    parser.add_argument("-jid","--jid",action="store_true")           # job id for folders
    parser.add_argument("-gui","--gui",action="store_true")           # gui placeholder
    # calculations summary (terapostp gampost)
    # nbo
    # charges
    # wavefunction - cube files
    # deloc indices - basin analysis
    # moldparse
    # Slab builder arguments
    # slab builder input: controli
    parser.add_argument("-slab_gen","--slab_gen",
                        help = "enables slab generation/periodic extension",action="store_true") #0
    # slab builder input: required
    parser.add_argument("-unit_cell","--unit_cell",
                        help = "path to xyz, or give generation instructions ") #1
    parser.add_argument("-cell_vector","--cell_vector",
                        help = "unit cell lattice vector, list of 3 list of float (Ang)") #2
    parser.add_argument("-cif_path","--cif_path",
                        help = "path to cif file") #3
    parser.add_argument("-duplication_vector","--duplication_vector",
                        help = "list of 3 in, lattice vector repeats") #4
    parser.add_argument("-slab_size","--slab_size",
                        help = "slab size, list of 3 floats (Ang)") #5
    # slab buidler: optional
    parser.add_argument("-miller_index","--miller_index",
                        help="list of 3 int, miller indicies") #6
    # placement input: control
    parser.add_argument("-place_on_slab","--place_on_slab",
                        help = "enables  placement on slab ",action="store_true") #0
    # placemnt input: required
    parser.add_argument("-target_molecule",'--target_molecule',
                        help = "path to target molecule") #1
    parser.add_argument("-align_distance_method","--align_distance_method",
                        help = "align distance method",
                        choices = ['chemisorption','physisorption','custom']) #2
    parser.add_argument("-align_dist","--align_dist",
                        help = "align distance, float") #3
    # placement input: optional
    parser.add_argument("-align_method","--align_method",
                        help = "align method ",choices = ['center', 'staggered','alignpair']) #4
    parser.add_argument("-object_align","--object_align",
                        help = "atom symbol or index for alignment partner in placed object")  #5
    parser.add_argument("-surface_atom_type","--surface_atom_type",
                        help = "atom symbol for surface aligment partner") #6
    parser.add_argument("-num_surface_atoms","--num_surface_atoms",
                        help = "number of surface sites to attach per adsorbate")#7
    parser.add_argument("-num_placements","--num_placements",
                        help = "number of copies of object to place.") #8
    parser.add_argument("-coverage","--coverage",
                        help = "coverage fraction, float between 0 and 1") #9
    parser.add_argument("-multi_placement_centering",'--multi_placement_centering',
                        help = "float between 0 and 1, controls centering of placment.Reccomend leaving as default") #10
    parser.add_argument("-control_angle","--control_angle",
                        help =  "angle in degrees to rotate object axis to surface")#11
    parser.add_argument("-angle_control_partner","-angle_control_partner",
                        help = 'atom index, int. Controls angle between object_align and this atom') #12
    parser.add_argument('-angle_surface_axis','--angle_surface_axis',
                        help = 'list of two floats, vector in surface plane to control angle relative to') #13
    parser.add_argument('-duplicate','--duplicate',
                        help = "boolean, duplicate asorbate above and below slab",action = "store_true") #14

    args=parser.parse_args()
    return args
