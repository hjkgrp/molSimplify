# @file addtodb.py
#  Adds new molecules to database
#
#  Written by Tim Ioannidis for HJK Group
#  Modified by JP Janet and Aditya Nandy
#  Dpt of Chemical Engineering, MIT

from molSimplify.Scripts.io import (copy_to_custom_path, readdict,
                                    lig_load, core_load, bind_load)
from molSimplify.Classes.globalvars import globalvars
import os
import sys
import re
import unicodedata
try:
    from openbabel import openbabel  # version 3 style import
except ImportError:
    import openbabel  # fallback to version 2
import shutil
import pathlib


def initialize_custom_database(globs):
    print('To add to database, you need to set a custom path. Please enter a writeable file path:')
    # Strip double quotes from the string
    raw_input = input('path=').replace('"', '')
    # Resolve the path to get the absolute path and convert back to string
    new_path = str(pathlib.Path(raw_input).resolve())
    globs.add_custom_path(new_path)
    globs.custom_path = new_path
    copy_to_custom_path()


# Add molecule to ligand database
#  @param smimol SMILES string or molecule file to be added
#  @param sminame Name of ligand for key in dictionary
#  @param smident Denticity of ligand
#  @param smicat Ligand connecting atoms
#  @param smigrps Ligand groups
#  @param smictg Ligand category
#  @param ffopt Flag for ligand FF optimization
#  @return Error messages
def addtoldb(smimol, sminame, smident, smicat, smigrps, smictg, ffopt, smichg=None):
    emsg = False
    globs = globalvars()
    if not globs.custom_path or not os.path.exists(str(globs.custom_path)):
        initialize_custom_database(globs)

    lipath = globs.custom_path + "/Ligands/ligands.dict"
    licores = readdict(lipath)
    ligands_folder = globs.custom_path + "/Ligands/"
    print("ligands_folder is : " + str(ligands_folder))
    # check if ligand exists
    if sminame in list(licores.keys()):
        emsg = 'Ligand '+sminame+' already existing in ligands database.'
        emsg += ' To replace, delete the existing entry first.'
        return emsg
    else:
        # get connection atoms
        ccats = [_f for _f in re.split(' |,|\t', smicat) if _f]
        # get groups
        groups = [_f for _f in re.split(' |,|\t', smigrps) if _f]
        grp = 'build '+' '.join(groups)
        grp += ' '+smictg
        if smicat == '':
            cats = list(range(0, int(smident)))
        else:
            cats = [int(a)-1 for a in ccats]
        cs = [str(a) for a in cats]
        css = ' '.join(cs)
        # convert to unicode
        smimol = unicodedata.normalize(
            'NFKD', smimol)
        sminame = unicodedata.normalize(
            'NFKD', sminame)
        if '~' in smimol:
            smimol = smimol.replace('~', os.expanduser('~'))
        # convert ligand from smiles/file
        lig, emsg = lig_load(smimol, licores)
        if emsg:
            return emsg
        lig.convert2mol3D()  # convert to mol3D

        shortname = sminame
        print("smimol is "+str(smimol))
        print("sminame is "+str(sminame))
        # sanitize ff options:
        if ffopt not in ["A", "B", "BA", "N"]:
            print('warning: incompatible ffopt choice. Options are ' +
                  str(["A", "B", "BA", "N"]))
            sys.exit(1)

        if smichg is not None:
            lig.charge = smichg
        # new entry for dictionary
        if '.mol' in smimol:
            shutil.copy2(smimol, ligands_folder + sminame+'.mol')
            snew = str(sminame)+':'+str(sminame)+'.mol,'+str(shortname)+','+str(css)+','+str(grp)+','+str(ffopt)+','+str(lig.charge)
        elif '.xyz' in smimol:
            shutil.copy2(smimol, ligands_folder + sminame+'.xyz')
            snew = str(sminame)+':'+str(sminame)+'.xyz,'+str(shortname)+','+str(css)+','+str(grp)+','+str(ffopt)+','+str(lig.charge)
        elif lig.OBMol:
            # write smiles file in Ligands directory
            obConversion = openbabel.OBConversion()
            obConversion.SetOutFormat("smi")
            obConversion.Read(lig.OBMol)
            obConversion.WriteFile(lig.OBMol, ligands_folder + sminame+'.smi')
            # lig.OBMol.write('smi',ligands_folder + sminame+'.smi')
            snew = str(sminame)+':'+str(sminame)+'.smi,'+str(shortname)+','+str(css)+','+str(grp)+','+str(ffopt)+','+str(lig.charge)
        else:
            # write xyz file in Ligands directory
            lig.writexyz(ligands_folder+sminame+'.xyz')  # write xyz file
            snew = str(sminame)+':'+str(sminame)+'.xyz,'+str(shortname)+','+str(css)+','+str(grp)+','+str(ffopt)+','+str(lig.charge)
        # update dictionary

        with open(lipath, 'r') as f:
            ss = f.read().splitlines()
        ss.append(snew)
        ssort = sorted(ss[1:])
        with open(lipath, 'w') as f:
            f.write(ss[0]+'\n')
            for s in ssort:
                f.write(s+'\n')
    return emsg


# Add molecule to cores database
#  @param smimol SMILES string or molecule file to be added
#  @param sminame Name of core for key in dictionary
#  @param smicat Core connecting atoms
#  @return Error messages
def addtocdb(smimol, sminame, smicat):
    emsg = False
    globs = globalvars()
    if not globs.custom_path or not os.path.exists(str(globs.custom_path)):
        initialize_custom_database(globs)
    cpath = globs.custom_path + "/Cores/cores.dict"
    mcores = readdict(cpath)
    cores_folder = globs.custom_path + "/Cores/"
    # check if core exists
    if sminame in list(mcores.keys()):
        emsg = 'Core '+sminame+' already existing in core database.'
        return emsg
    else:
        # get connection atoms
        ccats = [_f for _f in re.split(' |,|\t', smicat) if _f]
        cats = [int(a)-1 for a in ccats]
        if len(cats) == 0:
            cats = [0]
        cs = [str(a) for a in cats]
        css = ' '.join(cs)
        # convert to unicode
        smimol = unicodedata.normalize(
            'NFKD', smimol)
        if '~' in smimol:
            smimol = smimol.replace('~', os.expanduser('~'))
        # convert ligand from smiles/file
        core, emsg = core_load(smimol, mcores)
        if emsg:
            return emsg
        core.convert2mol3D()  # convert to mol3D
        # write xyz file in Cores directory
        # new entry for dictionary
        if '.mol' in smimol:
            shutil.copy2(smimol, cores_folder+sminame+'.mol')
            snew = sminame+':'+sminame+'.mol,'+css+','+'1'
        elif '.xyz' in smimol:
            shutil.copy2(smimol, cores_folder + sminame+'.xyz')
            snew = sminame+':'+sminame+'.xyz,'+css+','+'1'
        else:
            core.writexyz(cores_folder + sminame+'.xyz')  # write xyz file
            # new entry for dictionary
            snew = sminame+':'+sminame+'.xyz,'+css+','+'1'
        # update dictionary
        with open(cpath, 'r') as f:
            ss = f.read().splitlines()
        ss.append(snew)
        ssort = sorted(ss[1:])
        with open(cpath, 'w') as f:
            f.write(ss[0]+'\n')
            for s in ssort:
                f.write(s+'\n')
    return emsg


# Add molecule to binding species database
#  @param smimol SMILES string or molecule file to be added
#  @param sminame Name of binding species for key in dictionary
#  @return Error messages
def addtobdb(smimol, sminame):
    globs = globalvars()
    if not globs.custom_path or not os.path.exists(str(globs.custom_path)):
        initialize_custom_database(globs)
    bpath = globs.custom_path + "/Bind/bind.dict"
    bindcores = readdict(bpath)
    bind_folder = globs.custom_path + "/Bind/"
    # check if binding species exists
    if sminame in list(bindcores.keys()):
        emsg = 'Molecule '+sminame+' already existing in binding species database.'
        return emsg
    else:
        # convert to unicode
        smimol = unicodedata.normalize(
            'NFKD', smimol)
        sminame = unicodedata.normalize(
            'NFKD', sminame)
        if '~' in smimol:
            smimol = smimol.replace('~', os.expanduser('~'))
        # convert ligand from smiles/file
        bind, bsmi, emsg = bind_load(smimol, bindcores)
        if emsg:
            return emsg
        bind.convert2mol3D()  # convert to mol3D
        # new entry for dictionary
        # create shortname
        if len(sminame) > 5:
            shortname = sminame[0:3]+sminame[-2:]
        else:
            shortname = sminame
        if '.mol' in smimol:
            shutil.copy2(smimol, bind_folder+sminame+'.mol')
            snew = sminame+':'+sminame+'.mol,'+shortname+','
        elif '.xyz' in smimol:
            shutil.copy2(smimol, bind_folder + sminame+'.xyz')
            snew = sminame+':'+sminame+'.xyz,'+shortname+','
        elif bind.OBmol:
            # write smiles file in Bind species directory
            bind.OBmol.write('smi', bind_folder + sminame+'.smi')
            snew = sminame+':'+sminame+'.smi,'+shortname+','
        else:
            # write xyz file in Bind species directory
            bind.writexyz(bind_folder + sminame+'.xyz')  # write xyz file
            snew = sminame+':'+sminame+'.xyz,'+shortname+','
        # update dictionary
        with open(bpath, 'r') as f:
            ss = f.read().splitlines()
        ss.append(snew)
        ssort = sorted(ss[1:])
        with open(bpath, 'w') as f:
            f.write(ss[0]+'\n')
            for s in ssort:
                f.write(s+'\n')
    return emsg


# Remove molecule from database
#  @param sminame Name of molecule for key in dictionary
#  @param ropt Flag for molecule type (0 for core, 1 for ligand, 2 for binding species)
#  @return Error messages
def removefromDB(sminame, ropt):
    emsg = False
    globs = globalvars()
    if not globs.custom_path or not os.path.exists(str(globs.custom_path)):
        initialize_custom_database(globs)
    li_path = globs.custom_path + "/Ligands/ligands.dict"
    li_folder = globs.custom_path + "/Ligands/"
    core_path = globs.custom_path + "/Cores/cores.dict"
    core_folder = globs.custom_path + "/Cores/"
    bind_path = globs.custom_path + "/Bind/bind.dict"
    bind_folder = globs.custom_path + "/Bind/"

    # convert to unicode
    sminame = unicodedata.normalize('NFKD', sminame)

    if ropt == 1:
        # update dictionary
        with open(li_path, 'r') as f:
            ss = f.read().splitlines()
        ssort = sorted(ss[1:])
        with open(li_path, 'w') as f:
            f.write(ss[0]+'\n')
            for s in ssort:
                sss = s.split(':')
                if sminame != sss[0]:
                    f.write(s+'\n')
                else:
                    os.remove(li_folder + sss[1].split(',')[0])
    elif ropt == 0:
        # update dictionary
        with open(core_path, 'r') as f:
            ss = f.read().splitlines()
        ssort = sorted(ss[1:])
        with open(core_path, 'w') as f:
            f.write(ss[0]+'\n')
            for s in ssort:
                sss = s.split(':')
                if sminame != sss[0]:
                    f.write(s+'\n')
                else:
                    os.remove(core_folder+sss[1].split(',')[0])
    elif ropt == 2:
        # update dictionary
        with open(bind_path, 'r') as f:
            ss = f.read().splitlines()
        ssort = sorted(ss[1:])
        with open(bind_path, 'w') as f:
            f.write(ss[0]+'\n')
            for s in ssort:
                sss = s.split(':')
                if sminame != sss[0]:
                    f.write(s+'\n')
                else:
                    os.remove(bind_folder+sss[1].split(',')[0])
    return emsg
