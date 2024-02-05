# @file dbinteract.py
#  Interacts with databases for similarity searches and screening.
#
#  Written by Tim Ioannidis for HJK Group
#
#  Dpt of Chemical Engineering, MIT


import os
import sys
import re
import shutil
import string
try:
    import pymol
except ImportError:
    pass
try:
    from openbabel import openbabel  # version 3 style import
except ImportError:
    import openbabel  # fallback to version 2

from molSimplify.Classes.globalvars import (amassdict,
                                            glob,
                                            globalvars,
                                            mybash)
from molSimplify.Scripts.io import plugin_defs


def float_from_str(txt):
    numeric_const_pattern = r'[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    float_arr = rx.findall(txt)
    if not len(float_arr):
        return txt
    else:
        return float_arr[0]


# Setup database
#  @param dbselect Name of database
#  @return sdf and fs databases
def setupdb(dbselect):
    globs = globalvars()
    dbdir = os.path.relpath(globs.chemdbdir) + '/'
    # get files in directory
    dbfiles = os.listdir(dbdir)
    # search for db files
    dbmatches = [dbf for dbf in dbfiles if dbselect.lower() in dbf.lower()]
    dbsdf = [dbm for dbm in dbmatches if '.sdf' in dbm]
    dbfs = [dbm for dbm in dbmatches if '.fs' in dbm]
    # print('thefile list' + str(dbfiles))
    if len(dbsdf) == 0:
        print((dbselect + ' sdf database file missing from ' + dbdir +
              '. Please make sure file ' + dbselect + '.sdf is there..'))
        dbf1 = False
    else:
        dbf1 = dbdir + dbsdf[0]
    if len(dbfs) == 0:
        print((dbselect + ' fastsearch database file missing from ' + dbdir +
              '. Please make sure file ' + dbselect + '.fs is there, it speeds up search significantly..'))
        dbf2 = False
    else:
        dbf2 = dbdir + dbfs[0]
    return [dbf1, dbf2]


# Print prebuilt openbabel filters
#  @return String of prebuilt openbabel filters
def obfilters():
    s = " A list of available filters for Database searching is listed below.\n"
    s += """
    -abonds    Number of aromatic bonds
    -atoms    Number of atoms
    -bonds    Number of bonds
    -cansmi    Canonical SMILES
    -cansmiNS    Canonical SMILES without isotopes or stereo
    -dbonds    Number of double bonds
    -formula    Chemical formula
    -HBA1    Number of Hydrogen Bond Acceptors 1 (JoelLib)
    -HBA2    Number of Hydrogen Bond Acceptors 2 (JoelLib)
    -HBD    Number of Hydrogen Bond Donors (JoelLib)
    -InChI    IUPAC InChI identifier
    -InChIKey    InChIKey
    -L5    Lipinski Rule of Five
    -logP    octanol/water partition coefficient
    -MR    molar refractivity
    -MW    Molecular Weight filter
    -nF    Number of Fluorine Atoms
    -s    SMARTS filter
    -sbonds    Number of single bonds
    -smarts    SMARTS filter
    -tbonds    Number of triple bonds
    -title    For comparing a molecule's title
    -TPSA    topological polar surface area
    """
    s += "\n Similarity search can be performed using 4 fingerprints. Available fingerprints are:\n"
    s += """
    -FP2    Indexes linear fragments up to 7 atoms. (Default)
    -FP3    SMARTS patterns specified in the file /usr/local/share/openbabel/*/patterns.txt
    -FP4    SMARTS patterns specified in the file /usr/local/share/openbabel/*/SMARTS_InteLigand.txt
    -MACCS    SMARTS patterns specified in the file /usr/local/share/openbabel/*/MACCS.txt
    """
    return s


# Parse screening input from arguments
#  @param args Argument namespace
#  @return String of screening options
def checkscr(args):
    scr = '"'
    # if args.dbsmarts:
    #    scr += "s'"+args.dbsmarts+"' &"
    if args.dbatoms:
        nts = args.dbatoms.split('<')
        print('adding atom constraints')
        if nts[0] != '':
            scr += " atoms>" + nts[0] + " &"
        if nts[1] != '':
            scr += " atoms<" + nts[1] + " &"
    if args.dbbonds:
        nts = args.dbbonds.split('<')
        if nts[0] != '':
            scr += " bonds>" + nts[0] + " &"
        if nts[1] != '':
            scr += " bonds<" + nts[1] + " &"
    if args.dbarbonds:
        nts = args.dbarbonds.split('<')
        if nts[0] != '':
            scr += " abonds>" + nts[0] + " &"
        if nts[1] != '':
            scr += " abonds<" + nts[1] + " &"
    if args.dbsbonds:
        nts = args.dbsbonds.split('<')
        if nts[0] != '':
            scr += " sbonds>" + nts[0] + " &"
        if nts[1] != '':
            scr += " sbonds<" + nts[1] + " &"
    if args.dbmw:
        nts = args.dbmw.split('<')
        if nts[0] != '':
            scr += " MW>" + nts[0] + " &"
        if nts[1] != '':
            scr += " MW<" + nts[1] + " &"
    if scr == '"':
        scr = ''
    else:
        scr = scr[:-2] + '"'
    return scr


# Substructure search
#  @param smi Reference SMILES string
#  @param nmols Number of hits desired
#  @param dbselect Database to be searched
#  @param finger Fingerprint to be used
#  @param squery Filters to be applied
#  @param args Argument namespace
#  @return Filename of screening results
def getsimilar(smi, nmols, dbselect, finger, squery, args):
    # get database files
    [dbsdf, dbfs] = setupdb(dbselect)
    print(('database set up :' + str(dbsdf) + ' || ' + str(dbfs)))
    print(('Finding results similar, comparing to ' + smi))

    obab = 'obabel'
    if dbfs and args.dbfs:
        com = obab + ' ' + dbfs + ' ' + 'simres.smi -d -xf' + \
            finger + ' -s"' + smi + '" -al' + nmols
    else:
        mybash(obab + ' -isdf ' + dbsdf + ' -osdf -O tmp.sdf -d')
        com = obab + ' tmp.sdf -O simres.smi -xf' + finger + ' -s"' + smi + '"'
    # perform search using bash commandline
    print('Performing substructure search:')
    print(('running:  ' + str(com)))
    res = mybash(com)
    print(('res = ' + str(res)))
    print(('number of SMILES returned : ' + str(mybash('cat simres.smi | wc -l'))))

    if os.path.isfile('tmp.sdf'):
        os.remove('tmp.sdf')
    shutil.copy('simres.smi', 'initial.smi')
    if args.dbmaxsmartsmatches:
        print('Applying filters: inside get similar')
        com = obab + " -ismi simres.smi -osmi -O simres.smi -h --filter " + squery
        print(('running:  ' + str(com)))
        mybash(com)
        print(('number of lines in simres.smi: ' +
               str(mybash('cat simres.smi | wc -l'))))

    # com = obab+" -ismi simres.smi -osmi -O simres.smi -d --filter 'nsmartsmatches<="+args.dbmaxsmartsmatches+"'"
    # print('running:  '+ str(com))

    # res = mybash(com)
    # print('number of lines in simres.smi after dxbsmartmatches: '+str(mybash('cat simres.smi | wc -l')))

    # print res
    shutil.copy('simres.smi', 'afterfilteringsmarts.smi')
    # check output and print error if nothing was found
    if ('errors' in res):
        ss = 'No matches were found in DB. Log info:\n' + res
        print(ss)
        return ss, True
    else:
        return 'simres.smi', False


# Strip salts from list of SMILES results
#
#  Performs text matching
#  @param fname Filename of screening results
def stripsalts(fname):
    acc0 = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Si']
    acc1 = ['O-', 'F-', 'Cl-', 'Br-', 'I-', 'C@@H', 'C@H', 'N+', 'C@']
    rejected = ['@@H', '@H', "/", "\\"]
    accepted = acc0 + acc1
    if glob.glob(fname):
        with open(fname, 'r') as f:
            s = f.read().splitlines()
    else:
        print(('not found fname ' + str(fname)))

        return 0
    with open(fname, 'w') as f:
        for i, ss in enumerate(s):
            ss = ss.split('\t')[0]
            for r in rejected:
                if r in ss:
                    ss = ss.replace(r, '')
            ls = ss.split('[')
            for li in ls:
                if ']' in li:
                    lq = li.split(']')[0]
                    if lq not in accepted:
                        lq0 = '.[' + lq + ']'
                        lq1 = '[' + lq + '].'
                        if lq0 in ss:
                            ss = ss.replace(lq0, '')
                        elif lq1 in ss:
                            ss = ss.replace(lq1, '')
            ss = ss.split('.')[0]
            f.write(ss + '\n')
    return 0


# Get list of unique elements in SMILES string
#
#  Performs text matching
#  @param smistr SMILES string
#  @return List of elements
def getels(smistr):
    els = []
    els1 = ['H', 'B', 'C', 'N', 'O', 'F', 'K', 'P', 'S', 'V', 'Y', 'I']
    els2 = ['He', 'Li', 'Be', 'Na', 'Al', 'Si', 'Cl', 'Ar', 'Ca', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
            'Ge', 'As', 'Se', 'se', 'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
            'Sn', 'Sb', 'Te', 'Xe']
    for char in (string.punctuation + string.digits):
        smistr = smistr.replace(char, '')
    smistr = list(smistr[::-1])
    i = 0
    while i < len(smistr):
        if smistr[i].upper() in els1 and smistr[i].upper() not in els:
            els.append(smistr[i].upper())
        elif i < len(smistr) - 1:
            if smistr[i + 1] + smistr[i] in els2:
                if smistr[i + 1] + smistr[i] not in els:
                    els.append(smistr[i + 1] + smistr[i])
                smistr[i + 1] = ''
        i = i + 1
    return els


# Filters screening results based on list of allowed elements
#
#  Performs text matching
#  @param fname Filename of screening results
#  @param allowedels List of allowed elements
def checkels(fname, allowedels):
    print(('Filtering by allowed elements:' + str(allowedels)))
    if glob.glob(fname):
        with open(fname, 'r') as f:
            s = f.read().splitlines()
    else:
        return 0
    with open(fname, 'w') as f:
        nf = 0
        for i, ss in enumerate(s):
            ss = ss.split('\t')[0]
            els = getels(ss)
            flag = False
            for el in els:
                if el not in allowedels:
                    flag = True
            # print(el)
            if not flag:
                f.write(ss + '\n')
                nf = nf + 1
    print(('Element filter returns', str(nf), 'results'))
    return 0


# Maximal dissimilarity search
#
#  Uses a greedy algorithm that maximizes sums of Tanimoto distances with all elements picked
#
#  Results are written into dissimres.smi file.
#  @param outf Filename containing SMILES strings to be processed
#  @param n Number of dissimilar molecules required
def dissim(outf, n):
    obab = 'obabel'

    # clone hitlist file
    hit_list_path = "hitlist.smi"

    with open(outf) as f:
        smiles_list = f.readlines()
    with open(hit_list_path, 'w') as f:
        f.writelines(smiles_list)

    # generate fs of original hit list
    mybash(obab + ' -ismi ' + hit_list_path + ' -osdf -O tmp.sdf')
    mybash(obab + ' tmp.sdf -ofs')
    # number of hits
    numcpds = mybash('obabel tmp.sdf -onul')
    numcpds = int(numcpds.split(None)[0])
    # pick first element of list
    mybash('obabel tmp.sdf -O 1.smi -f 1  -l 1')
    del smiles_list[0]
    with open(hit_list_path, 'w') as f:
        f.writelines(smiles_list)
    # recompute the fs and number of hits parameters
    numcpds += -1  # decrease number of hits
    mybash(obab + ' -ismi ' + hit_list_path + ' -osdf tmp.sdf')
    mybash(obab + ' tmp.sdf -ofs')

    print('Performing dissimilarity search:')
    mostdissim = []
    if n > 1:
        # find most dissimilar structure
        for i in range(n - 1):

            # initialize list of total similarities
            simsum = [0] * numcpds
            # compute total similarity of each dissimilar structure with hit list
            for j in range(i + 1):
                a = mybash('obabel ' + str(j + 1) + '.smi tmp.sdf -ofpt')
                a = a.splitlines()
                a = [s.split('= ') for s in a]
                a = [item for sublist in a for item in sublist]
                aa = []
                for k in a:
                    try:
                        aa.append(float(k))
                    except ValueError:
                        pass
                a = aa
                simsum = [x + y for x, y in zip(simsum, a)]
            # pick most dissimilar structure by greedily minimizing total similarity
            mostdissim = simsum.index(min(simsum))
            mybash('obabel tmp.sdf -O ' + str(i + 2) + '.smi -f ' +
                   str(mostdissim + 1) + ' -l' + str(mostdissim + 1))

            # remove most dissimilar from the list and re-write the smi file
            del smiles_list[mostdissim]
            with open(hit_list_path, 'w') as f:
                f.writelines(smiles_list)

            # recompute the fs and number of hits parameters
            numcpds += -1  # decrease number of hits
            mybash(obab + ' -ismi ' + hit_list_path + ' -osdf tmp.sdf')
            mybash(obab + ' tmp.sdf -ofs')

    # combine results into one file
    with open('dissimres.smi', 'w') as f:
        for i in range(n):
            with open(str(i + 1) + '.smi', 'r') as ff:
                s = ff.read().splitlines()
            f.write(s[0] + '\n')
            os.remove(str(i + 1) + '.smi')
    return 0


# Matches initial SMARTS and computes connection atoms
#  @param smarts SMARTS string
#  @param outf Filename containing SMILES strings
#  @param catoms Connection atoms of SMARTS string
def matchsmarts(smarts, outf, catoms, args):
    sm = openbabel.OBSmartsPattern()
    print('---Test for developer version----')
    sm.Init(smarts)
    print(('smart is:', smarts))
    current_path = os.getcwd()
    print(('current path:', current_path))
    print(('file open:', outf))
    with open(outf, 'r') as f:
        s = f.read().splitlines()

    with open(outf, 'w') as f:
        # print('in file is:', s)
        moll = openbabel.OBMol()  # add
        obConversion = openbabel.OBConversion()  # add
        obConversion.SetInAndOutFormats("smi", "smi")  # add
        # print('!!!s:', s)
        # Set practically infinite size limit
        max_atoms = sys.maxsize
        if args.dbatoms:  # Overwrite max_atoms if dbatoms is given
            max_atoms = int(float_from_str(args.dbatoms))
        for i, mol in enumerate(s):
            obConversion.ReadString(moll, mol)  # add
            sm.Match(moll)
            smm = list(sm.GetUMapList())
            if 0 < len(smm) and len(mol) < max_atoms:
                print(('#:', i))
                print(('mol current:', mol))
                print(('smm current', smm, len(smm)))
                print(('catoms:', catoms))
                print(('!!!dbatoms:', max_atoms))
                pmatch = smm[0]
                cc = ''
                for at in catoms:
                    att = at - 1  # indexing
                    cc += str(pmatch[att]) + ','
                # if i < nres:
                f.write(mol + ' ' + cc[:-1] + '\n')
                # f.write(s[i]+'\n')
            else:
                pass
    return 0


# Main driver for database search
#  @param rundir Run directory
#  @param args Argument namespace
#  @param globs Global variables
def dbsearch(rundir, args, globs):
    cwd = os.getcwd()
    flag = False

    obab = 'obabel'
    # in any case do similarity search over indexed db #
    outf = args.dbfname if args.dbfname else 'simres.smi'  # output file
    # convert to SMILES/SMARTS if file
    if not args.dbbase:
        if args.gui:
            from molSimplify.Classes.mWidgets import mQDialogWarn
            qqb = mQDialogWarn('Warning', "No database file found within " +
                               globs.chemdbdir + '. Search not possible.')
            qqb.setParent(args.gui.DBWindow)
        print(("No database file found within " +
              globs.chemdbdir + '. Search not possible.'))
        return True
    # if args.dbsim:
    # print('similarity searching')
    # if '.smi' in args.dbsim:
    # if glob.glob(args.dbsim):
    # with open(args.dbsim,'r') as f:
    #     smistr = f.read()
    # else:
    # print 'File '+args.dbsim+' not existing. Check your input.'
    # print 'Similarity search terminating..'
    # return True
    # elif ('.mol' in args.dbsim or '.xyz' in args.dbsim):
    # if glob.glob(args.dbsim):
    # ftype = args.dbsim.split('.')[-1]
    # obConversion = openbabel.OBConversion()
    # obConversion.SetInFormat(ftype)
    # OBMol = openbabel.OBMol()
    # obConversion.ReadFile(OBMol,args.dbsim)
    # smistr = pybel.write("smi")
    # else:
    # print 'File '+args.dbsim+' not existing. Check your input.'
    # print 'Similarity search terminating..'
    # return True
    # else:
    # smistr = args.dbsim
    # print smistr
    if args.dbsmarts:
        if '.smi' in args.dbsmarts:
            if glob.glob(args.dbsmarts):
                with open(args.dbsmarts, 'r') as f:
                    smistr = f.read()
            else:
                print(('File ' + args.dbsmarts +
                      ' does not exist. Check your input.'))
                print('Substructure search terminating..')
                return 1
        elif ('.mol' in args.dbsmarts or '.xyz' in args.dbsmarts):
            if glob.glob(args.dbsmarts):
                smistr = pymol.write("smi")
            else:
                print(('File ' + args.dbsmarts +
                      ' does not exist. Check your input.'))
                print('Substructure search terminating..')
                return True
        else:
            smistr = args.dbsmarts
    elif args.dbhuman:
        smistr = []
        denticity = args.dbvdent if args.dbvdent else '1'
        coordatoms = args.dbvconns if args.dbvconns else 'N'
        hyb = args.dbvhyb if args.dbvhyb else '3'
        nlinks = args.dbvlinks if args.dbvlinks else '2'
        monod = ['1', 'mono', 'Mono', 'monodentate', 'Monodentate']
        bid = ['2', 'bi', 'Bi', 'bidentate', 'Bidentate']
        if args.debug:
            print('dbhuman conversion')
            print(('dbhuman coordatoms ' + str(coordatoms)))
            print(('dbhuman nlinks ' + str(nlinks)))
            print(('dbhuman hyb ' + str(hyb)))
        if denticity in monod:
            smistr = '[#' + str(amassdict[coordatoms[0]][1]
                                ) + '^' + hyb[0] + ';!+]'
        elif denticity in bid:
            smistr = '[#' + str(amassdict[coordatoms[0]][1]
                                ) + '^' + hyb[0] + ';!+]'
            for i in range(int(nlinks)):
                smistr = smistr + '[#6;R0]'
            print((coordatoms, hyb))
            smistr = smistr + \
                '[#' + str(amassdict[coordatoms[1]][1]) + '^' + hyb[1] + ';!+]'
        print(('setting smistr from dbhuman ' + smistr))

    # else:
    # get database
    # [dbsdf,dbfs] = setupdb(args.dbbase)
    # convert to smiles and print to output
    # if globs.osx:
    # cmd = "/usr/local/bin/obabel "+dbsdf+" -f0 -l100 -o"+outf[-3:]+" -O "+outf
    # else:
    # cmd = obab+" "+dbsdf+" -f0 -l100 -o"+outf[-3:]+" -O "+outf
    # t = mybash(cmd)
    # os.rename(outf,args.rundir+'/'+outf)
    # print t
    # return False
    # parse filters
    squery = checkscr(args)
    if args.debug:
        print(("squery is " + str(squery)))

    if args.dbmaxsmartsmatches:
        plugin_path = plugin_defs()
        shutil.copy(plugin_path, 'plugindefines.txt')
        cmd = "sed -i '/nsmartsmatches/!b;n;c" + smistr + "' " + 'plugindefines.txt'
        mybash(cmd)
    # run substructure search #
    nmols = '10000' if not args.dbnsearch else args.dbnsearch
    finger = 'FP2' if not args.dbfinger else args.dbfinger
    if int(nmols) > 3000 and args.gui:
        from molSimplify.Classes.mWidgets import mQDialogInf
        qqb = mQDialogInf(
            'Warning', "Database search is going to take a few minutes. Please wait..OK?")
        qqb.setParent(args.gui.DBWindow)
    if args.dbsmarts or args.dbhuman or args.dbsim:
        outputf, flag = getsimilar(
            smistr, nmols, args.dbbase, finger, squery, args)
        try:
            shutil.copy('simres.smi', outf)
        except (FileNotFoundError, shutil.SameFileError):
            pass

    if args.debug:
        print(('after similarity search, outf is ' + str(outputf)))
    if flag:
        if args.gui:
            from molSimplify.Classes.mWidgets import mQDialogWarn
            qqb = mQDialogWarn('Warning', "No matches found in search..")
            qqb.setParent(args.gui.DBWindow)
        print("No matches found in search..")
        return True
    # strip metals and clean-up, remove duplicates etc
    # print('mb ' +  str(mybash('cat '+outf)))
    if args.dbsmarts or args.dbhuman:
        print('Stripping salts and removing duplicates')

        print(('number of smiles strings BEFORE salt stripping: ' +
               mybash("cat " + outf + '| wc -l')))
        _ = stripsalts(outf)
        print(('number of smiles strings AFTER salt stripping: ' +
               mybash("cat " + outf + '| wc -l')))
        print(('number of smiles strings BEFORE unique: ' +
               mybash("cat " + outf + '| wc -l')))
        cmd = obab + " -ismi " + outf + " -osmi -O " + outf + " --unique"
        # print('running:' + str(cmd))
        shutil.copy(outf, 'afterstrippingsalts.smi')
        _ = mybash(cmd)
        print(('number of smiles strings AFTER unique: ' +
               mybash("cat " + outf + '| wc -l')))

    # print 't (ret from bash) is  '+ str(t)
    # filter results containing elements that aren't allowed
    if args.dballowedels:
        if args.dballowedels == 'organic':  # HCNO only
            allowedels = ['H', 'C', 'N', 'O']
        elif args.dballowedels == 'organohalides':
            allowedels = ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I']
        elif args.dballowedels == 'common':
            allowedels = ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'P', 'S']
        else:
            allowedels = args.dballowedels
        print(('number of smiles strings BEFORE element filter: ' +
               mybash("cat " + outf + '| wc -l')))
        checkels(outf, allowedels)
        print(('number of smiles strings AFTER element filter unique: ' +
               mybash("cat " + outf + '| wc -l')))

        shutil.copy(outf, 'afterfilteringels.smi')
    # check if defined connection atoms
    if args.dbcatoms:
        catoms = [int(a) for a in args.dbcatoms]
    else:
        catoms = [1]
    # do pattern matching
    # nres = 50 if not args.dbresults else int(args.dbresults)
    if args.dbsmarts or args.dbhuman:
        print(('number of smiles strings BEFORE SMARTS filter: ' +
               mybash("cat " + outf + '| wc -l')))
        _ = matchsmarts(smistr, outf, catoms, args)
        print(('number of smiles strings AFTER SMARTS filter: ' +
               mybash("cat " + outf + '| wc -l')))
    if args.debug:
        print(('outf is ' + str(outf)))
    # maximal dissimilarity search
    if args.dbdissim:
        dissim(outf, int(args.dbdissim))
    if args.rundir:
        print(('writing output to ' + str(args.rundir) + '/' + str(outf) + '\n'))
        os.rename(outf, args.rundir + '/' + outf)
    else:
        print(('writing output to ' + str(cwd) + '/' + str(outf) + '\n'))
    # os.chdir(cwd)
    return False
