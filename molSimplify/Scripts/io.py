# @file io.py
#  Input/output functions
#
#  Written by Tim Ioannidis for HJK Group
#
#  Dpt of Chemical Engineering, MIT

import copy
import random
import re
import shutil
import glob
import os
import time
import difflib

try:
    from openbabel import openbabel  # version 3 style import
except ImportError:
    import openbabel  # fallback to version 2
from typing import Any, List, Dict, Tuple, Union, Optional
from importlib_resources import files as resource_files

from molSimplify.Classes.globalvars import (globalvars,
                                            romans)
from molSimplify.Classes.mol3D import mol3D


# Print available geometries


def printgeoms():
    globs = globalvars()
    if globs.custom_path:
        f = globs.custom_path + "/Data/coordinations.dict"
    else:
        f = resource_files("molSimplify").joinpath("Data/coordinations.dict")
    with open(f, 'r') as f:
        s = f.read().splitlines()
    s = [_f for _f in s if _f]
    geomnames = []
    geomshorts = []
    coords = []
    for line in s:
        if (line[0] != '#'):
            vals = [_f for _f in re.split(',| |:', line) if _f]
            coords.append(vals[0])
            geomnames.append(vals[1])
            geomshorts.append(vals[2])
    geomgroups = list([] for a in set(coords))
    for i, g in enumerate(coords):
        geomgroups[int(g)-1].append(geomshorts[i])
    for i, g in enumerate(geomnames):
        print(("Coordination: %s, geometry: %s,\t short name: %s " %
              (coords[i], g, geomshorts[i])))
    print('')

# Get available geometries


def getgeoms():
    globs = globalvars()
    if globs.custom_path:
        f = globs.custom_path + "/Data/coordinations.dict"
    else:
        f = resource_files("molSimplify").joinpath("Data/coordinations.dict")
    with open(f, 'r') as f:
        s = f.read().splitlines()
    s = [_f for _f in s if _f]
    geomnames = []
    geomshorts = []
    coords = []
    for line in s:
        if (line[0] != '#'):
            vals = [_f for _f in re.split(',| |:', line) if _f]
            coords.append(vals[0])  # get coordination
            geomnames.append(vals[1])  # get name of geometry
            geomshorts.append(vals[2])  # get short names
    geomgroups = list([] for a in set(coords))  # get unique coordinations
    count = 0
    geomgroups[count].append(geomshorts[0])
    for i in range(1, len(coords)):
        if coords[i-1] != coords[i]:
            count += 1
        geomgroups[count].append(geomshorts[i])
    return coords, geomnames, geomshorts, geomgroups

# Read data into dictionary
#  @param fname Filename containing dictionary data
#  @return Dictionary


def readdict(fname):
    d = dict()
    with open(fname, 'r') as f:
        lines = [_f for _f in f.readlines() if _f]
    for line in lines:
        if (line[0] != '#') and line.strip():
            key = "".join([_f for _f in line.split(':')[0] if _f])
            val = "".join([_f for _f in line.split(':')[1] if _f])
            vals = [_f.strip() for _f in val.split(',') if _f]
            vv = []
            for i, val in enumerate(vals):
                vvs = [_f for _f in val.split(' ') if _f]
                if len(vvs) > 1 or i > 2:
                    vv.append(vvs)
                else:
                    vv += vvs
            d[key] = vv
    return d

# Read data into dictionary for substrate
#  @param fname Filename containing dictionary data
#  @return Dictionary


def readdict_sub(fname):
    # Constructor
    #  @param self The object pointer
    #  @param subname The name of the substrate
    class substrate:
        def __init__(self, subname):
            self.subname = subname
    d = dict()
    with open(fname, 'r') as f:
        txt = f.read()
    lines = [_f for _f in txt.splitlines() if _f]
    for line in lines:
        if (line[0] != '#') and line.strip():
            key = "".join([_f for _f in line.split(':')[0] if _f])
            val = "".join([_f for _f in line.split(':')[1] if _f])
            vals = [_f.strip() for _f in val.split(',') if _f]
            vv = []
            for i, val in enumerate(vals):
                vvs = ([_f for _f in val.split(' ') if _f])
                if len(vvs) > 1 or i > 2:
                    vv.append(vvs)
                else:
                    vv += vvs
            # dict keys are instances of the substrate class
            d[substrate(key)] = vv
    return d

# Get ligands in dictionary
#  @return List of ligands in dictionary


def getligs() -> str:
    licores = getlicores()
    return ' '.join(sorted(licores.keys()))

# Get ligands cores
#
#  This form of the function is used extensively in the GUI so it got it's own call.
#  This is basically the same as getligs() but returns the full dictionary
#  @param flip if we want to return flipped versions of bidentates
#  @return Ligands dictionary


def getlicores(flip: bool = True) -> Dict[str, Any]:
    globs = globalvars()
    if globs.custom_path:  # test if a custom path is used:
        licores_path = str(globs.custom_path).rstrip('/') + "/Ligands/ligands.dict"
    else:
        licores_path = resource_files("molSimplify").joinpath("Ligands/ligands.dict")
    licores = readdict(licores_path)
    if flip:
        for ligand in list(licores.keys()):
            if len(licores[ligand][2]) == 2 and type(licores[ligand][2]) == list:
                licores[ligand+'_flipped'] = copy.deepcopy(licores[ligand])
                licores[ligand+'_flipped'][2].reverse()
    return licores

# Get simple ligands in dictionary
#  @return List of ligands in simple ligands dictionary


def getsimpleligs() -> str:
    slicores = getslicores()
    return ' '.join(sorted(slicores.keys()))

# Get simple ligands cores
#
#  This form of the function is used extensively in the GUI so it got it's own call.
#  This is basically the same as getsimpleligs() but returns the full dictionary
#  @return Simple ligands dictionary


def getslicores() -> Dict[str, Any]:
    globs = globalvars()
    if globs.custom_path:  # test if a custom path is used:
        slicores_path = str(globs.custom_path).rstrip(
            '/') + "/Ligands/simple_ligands.dict"
    else:
        slicores_path = resource_files("molSimplify").joinpath("Ligands/simple_ligands.dict")
    slicores = readdict(slicores_path)
    return slicores

# Get ligand groups
#  @param licores Ligand dictionary
#  @return Ligand groups


def getligroups(licores: dict) -> str:
    groups = []
    for entry in licores:
        groups += licores[entry][3]
    groups = sorted(list(set(groups)))
    a = ' '.join(groups)
    return a

# Enclose metal elements in SMILES string with square brackets
#  @param smi SMILES string
#  @return Processed SMILES string


def checkTMsmiles(smi: str) -> str:
    g = globalvars()
    for m in g.metalslist():
        if m in smi:
            smi = smi.replace(m, '['+m+']')
    return smi

# Get binding species in dictionary
#  @return List of binding species in dictionary


def getbinds() -> str:
    bindcores = getbcores()
    return ' '.join(sorted(bindcores.keys()))

# Get binding species cores
#
#  This form of the function is used extensively in the GUI so it got it's own call.
#  This is basically the same as getbinds() but returns the full dictionary
#  @return Binding species dictionary


def getbcores() -> dict:
    globs = globalvars()
    if globs.custom_path:  # test if a custom path is used:
        bcores_path = str(globs.custom_path).rstrip('/') + "/Bind/bind.dict"
    else:
        bcores_path = resource_files("molSimplify").joinpath("Bind/bind.dict")
    bcores = readdict(bcores_path)
    return bcores

# Get cores in dictionary
#  @return List of cores in dictionary


def getcores():
    mcores = getmcores()
    a = []
    for key in mcores:
        a.append(key)
    a = sorted(a)
    a = ' '.join(a)
    return a

# Get core cores
#
#  This form of the function is used extensively in the GUI so it got it's own call.
#  This is basically the same as getcores() but returns the full dictionary
#  @return Cores dictionary


def getmcores():
    globs = globalvars()
    if globs.custom_path:  # test if a custom path is used:
        mcores = str(globs.custom_path).rstrip('/') + "/Cores/cores.dict"
    else:
        mcores = resource_files("molSimplify").joinpath("Cores/cores.dict")
    mcores = readdict(mcores)
    return mcores

# Get substrates in dictionary
#  @return List of substrates in dictionary


def getsubstrates():
    subcores = getsubcores()
    a = []
    for key in subcores:
        a.append(key)
    a = sorted(a)
    a = ' '.join(a)
    return a

# Get substrate cores
#
#  This form of the function is used extensively in the GUI so it got it's own call.
#  This is basically the same as getsubstrates() but returns the full dictionary
#  @return Substrates dictionary


def getsubcores():
    globs = globalvars()
    if globs.custom_path:  # test if a custom path is used:
        subcores = str(globs.custom_path).rstrip(
            '/') + "/Substrates/substrates.dict"
    else:
        subcores = resource_files("molSimplify").joinpath("Substrates/substrates.dict")
    subcores = readdict_sub(subcores)
    return subcores

# Load M-L bond length dictionary from data
#  @param path to data file
#  @return M-L bond length dictionary


def loaddata(path: str) -> dict:
    globs = globalvars()
    # loads ML data from ML.dat file and
    # store to dictionary
    if globs.custom_path:  # test if a custom path is used:
        fname = str(globs.custom_path).rstrip('/') + path
    else:
        fname = resource_files("molSimplify").joinpath(path.strip('/'))
    d = dict()

    with open(fname) as f:
        txt = f.read()
    lines = [_f for _f in txt.splitlines() if _f]
    for line in lines[1:]:
        if '#' != line[0]:  # skip comments
            s = [_f for _f in line.split(None) if _f]
            d[(s[0], s[1], s[2], s[3], s[4])] = s[5]  # read dictionary
    return d

# Load M-L bond length dictionary from data
#  @param path to data file
#  @return M-L bond length dictionary


def loaddata_ts(path: str) -> dict:
    globs = globalvars()
    # loads ML data from ML.dat file and
    # store to dictionary
    if globs.custom_path:  # test if a custom path is used:
        fname = str(globs.custom_path).rstrip('/') + path
    else:
        fname = resource_files("molSimplify").joinpath(path.strip('/'))
    d = dict()

    with open(fname) as f:
        txt = f.read()
    lines = [_f for _f in txt.splitlines() if _f]
    for line in lines[1:]:
        if '#' != line[0]:  # skip comments
            s = [_f for _f in line.split(None) if _f]
            d[(s[0], s[1], s[2], s[3])] = s[4:]  # read dictionary
    return d

# Load a chemdraw cdxml file and write out xyz
# @param cdxml a cdxml file
# return fname the xyz fname for the read-in cdxml


def loadcdxml(cdxml: str) -> Tuple[str, str]:
    # try importing pybel
    try:
        import pybel
    except ImportError:  # What is the purpose of excepting and then raising?
        raise
    fname = re.sub(r'.cdxml', '', cdxml)  # file name for the new xyz
    # check cdxml file for Dashed bonds
    with open(cdxml, 'r') as f:
        lines = f.read().splitlines()
    signal = False
    for i, line in enumerate(lines):
        if 'Dash' in line:
            lnum = i
            signal = True
            break
    # remove the dash bond
    if signal:
        cdxml = cdxml.replace('.cdxml', '.temp.cdxml')
    with open(cdxml, 'a') as f:
        if signal:
            for i, line in enumerate(lines):
                if i not in list(range(lnum-5, lnum+2)):
                    f.write(line + '\n')
        else:
            for i, line in enumerate(lines):
                f.write(line + '\n')
    # load cdxml into obmol
    obconv = openbabel.OBConversion()  # ob Class
    obmol = openbabel.OBMol()  # ob Class
    obconv.SetInFormat('cdxml')  # ob Method to set cdxml
    obconv.ReadFile(obmol, cdxml)  # ob Method to reaad cdxml into a OBMol()
    if signal:
        os.remove(cdxml)
    # substitute Si for metals
    obmol.NumAtoms()
    idx_list = []
    atno_list = []
    for idx in range(obmol.NumAtoms()):
        if obmol.GetAtom(idx+1).IsMetal():
            idx_list.append(idx)
            atno_list.append(obmol.GetAtom(idx+1).GetAtomicNum())
            obmol.GetAtom(idx+1).SetAtomicNum(14)
    # convert 2D to 3D
    pymol = pybel.Molecule(obmol)
    pymol.make3D()
    pymol.localopt()
    # recover metal symbols
    for i in range(len(idx_list)):
        idx = idx_list[i]
        atno = atno_list[i]
        obmol.GetAtom(idx+1).SetAtomicNum(atno)
    # determine the number of fragments in obmol
    mol = mol3D()
    mol.OBMol = obmol
    mol.convert2mol3D()
    fraglist = mol.getfragmentlists()
    # write xyzfs
    msg = ''
    if len(fraglist) > 1:
        for atidxes in fraglist:
            frag = mol3D()
            for atidx in atidxes:
                atom = mol.getAtom(atidx)
                frag.addAtom(atom)
            if len(frag.findMetal()) > 0:
                frag.writexyz(fname + '_cat.xyz')
            else:
                frag.writexyz(fname + '_sub.xyz')
        msg = 'two fragments were saved individually as xyzf'
    else:
        mol.writexyz(fname + '.xyz')
        msg = 'one molecule was saved as xyzf'

    return fname, msg

# Load backbone coordinates
#  @param coord Name of coordination geometry
#  @return List of backbone coordinates


def loadcoord(coord: str) -> List[List[float]]:
    globs = globalvars()
    if globs.custom_path:
        f = globs.custom_path + "/Data/" + coord + ".dat"
    else:
        f = resource_files("molSimplify").joinpath(f"Data/{coord}.dat")
    with open(f) as f:
        txt = [_f for _f in f.read().splitlines() if _f]
    b = []
    for line in txt:
        s = [_f for _f in line.split(None) if _f]
        b.append([float(s[0]), float(s[1]), float(s[2])])
    return b

# Load core and convert to mol3D
#  @param usercore Name of core
#  @param mcores Cores dictionary (reloads if not specified - default, useful when using an externally modified dictionary)
#  @return mol3D of core, error messages


def core_load(usercore: str, mcores: Optional[dict] = None) -> Tuple[Union[mol3D, None], str]:
    if mcores is None:
        mcores = getmcores()
    globs = globalvars()
    if '~' in usercore:
        homedir = os.path.expanduser("~")
        usercore = usercore.replace('~', homedir)
    emsg = ''
    core = mol3D()  # initialize core molecule
    # check if core exists in dictionary
    if usercore.lower() in list(mcores.keys()):
        # print('loading core from dictionary')
        dbentry = mcores[usercore.lower()]
        # load core mol file (with hydrogens
        if globs.custom_path:
            fcore = globs.custom_path + "/Cores/" + dbentry[0]
        else:
            fcore = str(resource_files("molSimplify").joinpath(f"Cores/{dbentry[0]}"))
        # check if core xyz/mol file exists
        if not glob.glob(fcore):
            emsg = "We can't find the core structure file %s right now! Something is amiss. Exiting..\n" % fcore
            print(emsg)
            return None, emsg
        if ('.xyz' in fcore):
            core.OBMol = core.getOBMol(fcore, 'xyzf')
        elif ('.mol' in fcore):
            core.OBMol = core.getOBMol(fcore, 'molf')
        elif ('.smi' in fcore):
            core.OBMol = core.getOBMol(fcore, 'smif')
        core.cat = [int(i) for i in [_f for _f in dbentry[1] if _f]]
        core.denticity = dbentry[2]
        core.ident = usercore
    # load from file
    elif ('.mol' in usercore or '.xyz' in usercore or '.smi' in usercore):
        if glob.glob(usercore):
            ftype = usercore.split('.')[-1]
            print(('Core is a '+ftype+' file'))
            # try and catch error if conversion doesn't work
            try:
                core.OBMol = core.getOBMol(
                    usercore, ftype+'f')  # convert from file
                print('Core successfully converted to OBMol')
            except IOError:
                emsg = 'Failed converting file ' + usercore+' to molecule..Check your file.\n'
                print(emsg)
                return None, emsg
            core.ident = usercore.split('.')[0]
            core.ident = core.ident.rsplit('/')[-1]
        else:
            emsg = 'Core file '+usercore+' does not exist. Exiting..\n'
            print(emsg)
            return None, emsg
    # if not, try converting from SMILES
    else:
        # check for transition metals
        usercore = checkTMsmiles(usercore)
        # try and catch error if conversion doesn't work
        try:
            core.OBMol = core.getOBMol(
                usercore, 'smistring', True)  # convert from smiles
            print('Core successfully interpreted as smiles')
        except IOError:
            emsg = "We tried converting the string '%s' to a molecule but it wasn't a valid SMILES string.\n" % usercore
            emsg += "Furthermore, we couldn't find the core structure: '%s' in the cores dictionary. Try again!\n" % usercore
            emsg += "\nAvailable cores are: %s\n" % getcores()
            print(emsg)
            return None, emsg
        core.cat = [0]
        core.denticity = 1
        core.ident = 'core'
    return core, emsg

# Load substrate and convert to mol3D
#  @param usersubstrate Name of substrate
#  @param subcores Substrates dictionary
#       (reloads if not specified - default, useful when using an externally modified dictionary)
#  @return mol3D of substrate, subscatom, error messages
#  attributes of substrate: OBMol, denticity, ident (identity), charge,
#                           cat (connection atom index), and grps (substrate group)


def substr_load(usersubstrate: str,
                sub_i: int,
                subcatoms: List[int],
                subcores: Optional[dict] = None) -> Tuple[Union[mol3D, None], List[int], str]:
    # if not using a user-defined substrate dictionary
    if subcores is None:
        subcores = getsubcores()
    # load global variables
    globs = globalvars()
    if '~' in usersubstrate:
        homedir = os.path.expanduser("~")
        usersubstrate = usersubstrate.replace('~', homedir)
    emsg = ''
    sub = mol3D()  # initialize core molecule
    # default attributes of the sub3D
    sub.denticity = 1
    sub.ident = None
    sub.charge = 0
    sub.cat = [0]
    sub.grps = ['inter']
    # check if substrate exists in dictionary
    if usersubstrate.lower() in [i.subname for i in list(subcores.keys())]:
        print('loading substrate from dictionary')
        # create a list for each item column in the dictionary
        var_list = []
        for var in [subcores[i][0:] for i in list(subcores.keys()) if i.subname == usersubstrate.lower()]:
            var_list.append(var)
        var_list = sorted(var_list)
        var_list_sub_i = var_list[sub_i]
        if globs.custom_path:
            fsubst = globs.custom_path + "/Substrates/" + var_list_sub_i[0]
        else:
            fsubst = str(resource_files("molSimplify").joinpath(f"Substrates/{var_list_sub_i[0]}"))
        # check if substrate xyz/mol file exists
        if not glob.glob(fsubst):
            emsg = "We can't find the substrate structure file %s right now! Something is amiss. Exiting..\n" % fsubst
            print(emsg)
            return None, subcatoms, emsg
        if ('.xyz' in fsubst):
            sub.OBMol = sub.getOBMol(fsubst, 'xyzf')
        elif ('.mol' in fsubst):
            sub.OBMol = sub.getOBMol(fsubst, 'molf')
        elif ('.smi' in fsubst):
            sub.OBMol = sub.getOBMol(fsubst, 'smif')
        # Parsing substrate denticity
        # modified the check for length,
        # as it parsing string length instead of
        # list length!
        if isinstance(var_list_sub_i[2], str):
            sub.denticity = 1
        else:
            sub.denticity = len(var_list_sub_i[2])
        # Parsing substrate identity
        sub.ident = var_list_sub_i[1]
        # Parsing substrate charge
        sub.charge = sub.OBMol.GetTotalCharge()
        # Parsing substrate connection atoms
        if 'pi' in var_list_sub_i[2]:
            sub.denticity = 1
            sub.cat = [int(li) for li in var_list_sub_i[2][:-1]]
            sub.cat.append('pi')
        else:
            sub.cat = [int(li) for li in var_list_sub_i[2]]
        if not subcatoms:
            subcatoms = sub.cat
        # Parsing substrate group
        sub.grps = [li for li in var_list_sub_i[3]]
        if len(var_list_sub_i[4]) > 0:
            sub.ffopt = var_list_sub_i[4]
    # load from file
    elif ('.mol' in usersubstrate or '.xyz' in usersubstrate or '.smi' in usersubstrate):
        if glob.glob(usersubstrate):
            ftype = usersubstrate.split('.')[-1]
            print(('Substrate is a '+ftype+' file'))
            # try and catch error if conversion doesn't work
            try:
                sub.OBMol = sub.getOBMol(
                    usersubstrate, ftype+'f')  # convert from file
                print('Substrate successfully converted to OBMol')

            except IOError:
                emsg = 'Failed converting file ' + usersubstrate + \
                    ' to molecule..Check your file.\n'
                print(emsg)
                return None, subcatoms, emsg
            sub.ident = usersubstrate.split('/')[-1].split('.')[0]
        else:
            emsg = 'Substrate file '+usersubstrate+' does not exist. Exiting..\n'
            print(emsg)
            return None, subcatoms, emsg
    # if not, try converting from SMILES
    else:
        # check for transition metals
        usersubstrate = checkTMsmiles(usersubstrate)
        # try and catch error if conversion doesn't work
        try:
            sub.OBMol = sub.getOBMol(
                usersubstrate, 'smistring', True)  # convert from smiles
            print('Substrate successfully interpreted as smiles')
        except IOError:
            emsg = f"We tried converting the string '{usersubstrate}' to a molecule but it wasn't a valid SMILES string.\n"
            emsg += f"Furthermore, we couldn't find the substrate structure: '{usersubstrate}' in the substrates dictionary. "
            emsg += f"Try again!\n\nAvailable substrates are: {getsubstrates()}\n"
            print(emsg)
            return None, subcatoms, emsg
        sub.cat = [0]
        sub.denticity = 1
        sub.ident = 'substrate'
    return sub, subcatoms, emsg


# TODO: Output currently typed as any instead of Union[mol3D, None] because many other
# scripts depend on a mol3D as first return value.
def lig_load(userligand: str, licores: Optional[dict] = None) -> Tuple[Any, str]:

    if licores is None:
        licores = getlicores()
        # @licores.pop("x", None)
    globs = globalvars()
    # ## get groups ###
    groups = []
    for entry in licores:
        groups += licores[entry][3]
    groups = sorted(list(set(groups)))
    # check if user requested group
    if userligand.lower() in groups:
        subligs = [key for key in licores if userligand.lower()
                   in licores[key][3]]
        # randomly select ligand
        userligand = random.choice(subligs)
    if '~' in userligand:
        homedir = os.path.expanduser("~")
        userligand = userligand.replace('~', homedir)
    emsg = ''
    lig = mol3D()  # initialize ligand molecule
    lig.needsconformer = False
    # get similarity of userligand to ligands in dictionary, from the sequence point of view
    text_similarities = [difflib.SequenceMatcher(None, userligand, i).ratio() for i in list(licores.keys())]
    # check if ligand exists in dictionary
    if userligand in list(licores.keys()) or max(text_similarities) > 0.6:  # Two cases here
        if userligand in list(licores.keys()):  # Ligand is in the dictionary ligands.dict
            print(('loading ligand from dictionary: ' + str(userligand)))
            dbentry = licores[userligand]
        else:
            # max(text_similarities) > 0.6 --> It is likely the user made a typo in inputting a ligand that is in ligands.dict
            max_similarity = max(text_similarities)
            index_max = text_similarities.index(max_similarity)
            desired_ligand = list(licores.keys())[index_max]
            print(f'ligand was not in dictionary, but the sequence is very similar to a ligand that is: {str(desired_ligand)}')
            print(('loading ligand from dictionary: ' + str(desired_ligand)))
            dbentry = licores[desired_ligand]  # Loading the typo-d ligand
        # load lig mol file (with hydrogens)
        if globs.custom_path:
            flig = globs.custom_path + "/Ligands/" + dbentry[0]
        else:
            flig = str(resource_files("molSimplify").joinpath(f"Ligands/{dbentry[0]}"))
        # check if ligand xyz/mol file exists
        print(('looking for '+flig))
        if not os.path.isfile(flig):
            emsg = "We can't find the ligand structure file %s right now! Something is amiss. Exiting..\n" % flig
            print(emsg)
            return False, emsg
        if ('.xyz' in flig):
            lig.OBMol = lig.getOBMol(flig, 'xyzf')
            # Set charge to last entry in ligands.dict
            lig.OBMol.SetTotalCharge(int(dbentry[-1][0]))
        elif ('.mol2' in flig):
            lig.OBMol = lig.getOBMol(flig, 'mol2f')
        elif ('.mol' in flig):
            lig.OBMol = lig.getOBMol(flig, 'molf')
        elif ('.smi' in flig):
            print('SMILES conversion')
            lig.OBMol = lig.getOBMol(flig, 'smif')
            lig.needsconformer = True

        # modified the check for length,
        # as it parsing string length instead of
        # list length!
        if isinstance(dbentry[2], str):
            lig.denticity = 1
        else:
            lig.denticity = len(dbentry[2])
        lig.ident = dbentry[1]
        lig.convert2mol3D()
        lig.charge = lig.OBMol.GetTotalCharge()
        if 'pi' in dbentry[2]:
            lig.cat = [int(li) for li in dbentry[2][:-1]]
            lig.cat.append('pi')
        else:
            if lig.denticity == 1:
                lig.cat = [int(dbentry[2])]
            else:
                lig.cat = [int(li) for li in dbentry[2]]
        if lig.denticity > 1:
            lig.grps = dbentry[3]
        else:
            lig.grps = []
        if len(dbentry) > 3:
            lig.ffopt = dbentry[4][0]
    # load from file
    elif ('.mol' in userligand or '.xyz' in userligand or '.smi' in userligand or '.sdf' in userligand):
        if glob.glob(userligand):
            ftype = userligand.split('.')[-1]
            # try and catch error if conversion doesn't work
            try:
                print(('ligand is an '+ftype+' file'))
                lig.OBMol = lig.getOBMol(
                    userligand, ftype+'f')  # convert from file
                # generate coordinates if not existing
                lig.charge = lig.OBMol.GetTotalCharge()
                print('Ligand successfully converted to OBMol')
            except IOError:
                emsg = 'Failed converting file ' + userligand+' to molecule..Check your file.\n'
                return False, emsg
            lig.ident = userligand.rsplit('/')[-1]
            lig.ident = lig.ident.split('.'+ftype)[0]
        else:
            emsg = 'Ligand file '+userligand+' does not exist. Exiting..\n'
            print(emsg)
            return False, emsg
    # if not, try interpreting as SMILES string
    else:
        print(f'Interpreting ligand {userligand} as a SMILES string, as it was not in the ligands dictionary.')
        print('Available ligands in the ligands dictionary can be found at molSimplify/molSimplify/Ligands/ligands.dict\n'
              'Or by running the command `molsimplify -h liganddict`')
        try:
            lig.getOBMol(userligand, 'smistring', True)  # convert from smiles
            lig.convert2mol3D()
            assert lig.natoms
            lig.charge = lig.OBMol.GetTotalCharge()
            print('Ligand successfully interpreted as SMILES')
        except IOError:
            emsg = f"We tried converting the string '{userligand}' to a molecule but it wasn't a valid SMILES string.\n"
            emsg += f"Furthermore, we couldn't find the ligand structure: '{userligand}' in the ligands dictionary. "
            emsg += f"Try again!\n\nAvailable ligands are: {getligs()}\n"
            emsg += f"\nAnd available groups are: {getligroups(licores)}\n"
            print(emsg)
            return False, emsg
        lig.ident = 'smi'
        lig.needsconformer = True
    lig.name = userligand
    return lig, emsg

# Load binding species and convert to mol3D
#  @param userbind Name of binding species
#  @param bindcores Binding species dictionary
#      (reloads if not specified - default, useful when using an externally modified dictionary)
#  @return mol3D of binding species, error messages


def bind_load(userbind: str, bindcores: dict) -> Tuple[Union[mol3D, None], bool, str]:
    globs = globalvars()
    if '~' in userbind:
        homedir = os.path.expanduser("~")
        userbind = userbind.replace('~', homedir)
    emsg = ''
    bind = mol3D()  # initialize binding molecule
    bsmi = False  # flag for smiles
    # check if binding molecule exists in dictionary
    if userbind in list(bindcores.keys()):
        # load bind mol file (with hydrogens)
        #        fbind = installdir+'Bind/'+bindcores[userbind][0]
        if globs.custom_path:
            fbind = globs.custom_path + "/Bind/" + bindcores[userbind][0]
        else:
            fbind = str(resource_files("molSimplify").joinpath(f"Bind/{bindcores[userbind][0]}"))
        # check if bind xyz/mol file exists
        if not glob.glob(fbind):
            emsg = "We can't find the binding species structure file %s right now! Something is amiss. Exiting..\n" % fbind
            print(emsg)
            return None, False, emsg
        if ('.xyz' in fbind):
            bind.OBMol = bind.getOBMol(fbind, 'xyzf')
        elif ('.mol' in fbind):
            bind.OBMol = bind.getOBMol(fbind, 'molf')
        elif ('.smi' in fbind):
            bind.OBMol = bind.getOBMol(fbind, 'smif')
        bind.charge = bind.OBMol.GetTotalCharge()
    # load from file
    elif ('.mol' in userbind or '.xyz' in userbind or '.smi' in userbind):
        if glob.glob(userbind):
            ftype = userbind.split('.')[-1]
            # try and catch error if conversion doesn't work
            try:
                bind.OBMol = bind.getOBMol(
                    userbind, ftype+'f')  # convert from file
                bind.charge = bind.OBMol.GetTotalCharge()
            except IOError:
                emsg = 'Failed converting file ' + userbind+' to molecule..Check your file.\n'
                return None, False, emsg
            bind.ident = userbind.rsplit('/')[-1]
            bind.ident = bind.ident.split('.'+ftype)[0]
        else:
            emsg = 'Binding species file '+userbind+' does not exist. Exiting..\n'
            return None, False, emsg
    # if not, try converting from SMILES
    else:
        # check for transition metals
        userbind = checkTMsmiles(userbind)
        # try and catch error if conversion doesn't work
        try:
            bind.OBMol = bind.getOBMol(userbind, 'smi')  # convert from smiles
            bind.charge = bind.OBMol.GetTotalCharge()
            bsmi = True
            bind.ident = 'smi'
        except IOError:
            emsg = "We tried converting the string '%s' to a molecule but it wasn't a valid SMILES string.\n" % userbind
            emsg += "Furthermore, we couldn't find the binding species structure: "
            emsg += "'%s' in the binding species dictionary. Try again!\n" % userbind
            print(emsg)
            return None, False, emsg
    return bind, bsmi, emsg

# Write input file from arguments
#  @param args Namespace of arguments
#  @param fname File name


def getinputargs(args, fname: str):
    # list with arguments
    # write input args
    with open(fname+'.molinp', 'w') as f:
        f.write("# Input file generated from molSimplify at " +
                time.strftime('%m/%d/%Y %H:%M')+'\n')
        for arg in vars(args):
            if 'nbind' not in arg and 'rgen' not in arg and 'i' != arg:
                if getattr(args, arg):
                    f.write('-'+arg+' ')
                    if isinstance(getattr(args, arg), list):
                        for ii, iar in enumerate(getattr(args, arg)):
                            if isinstance(iar, list):
                                if ii < len(getattr(args, arg))-1:
                                    f.write('/')
                                for jj, iiar in enumerate(iar):
                                    f.write(str(iiar))
                                    if jj < len(iar)-1:
                                        f.write(',')
                            else:
                                f.write(str(iar))
                                if ii < len(getattr(args, arg))-1:
                                    f.write(',')
                    else:
                        f.write(str(getattr(args, arg)))
                    f.write('\n')

# Load plugin definitions


def plugin_defs() -> str:
    plugin_path = str(resource_files("molSimplify").joinpath("plugindefines_reference.txt"))
    return plugin_path

# def get_name(args,rootdir,core,ligname,bind = False,bsmi = False):
    # DEPRECIATED, USE NAME_COMPLEX instead
    # reads in argument namespace
    # and chooses an appropriate name
    # bind_ident is used to pass binding
    # species information
    # print('the root directory for this calc is '+ (rootdir))
    # check if smiles string in binding species
    # if args.bind:
    # if bsmi:
    # if args.nambsmi: # if name specified use it in file
    # fname = rootdir+'/'+core.ident[0:3]+ligname+args.nambsmi[0:2]
    # if args.name:
    # fname = rootdir+'/'+args.name+args.nambsmi[0:2]
    # else: # else use default
    # fname = rootdir+'/'+core.ident[0:3]+ligname+'bsm'
    # if args.name:
    # fname = rootdir+'/'+args.name+'bsm'
    # else: # else use name from binding in dictionary
    # fname = rootdir+'/'+core.ident[0:3]+ligname+bind.ident[0:2]
    # if args.name:
    # fname = rootdir+'/'+args.name + bind.ident[0:2]
    # else:
    # if globs.debug:
    # print('the root calculation directory is' + str(rootdir))
    # fname = rootdir+'/'+core.ident[0:3]+ligname
    # if args.name:
    # fname = rootdir+'/'+args.name

    # return fname

# Generate complex name (this is actually used instead of namegen.py)
#  @param rootdir Root directory
#  @param core mol3D of core
#  @param ligs List of ligand names
#  @param ligoc List of ligand occurrences
#  @param sernum Complex serial number
#  @param args Namespace of arguments
#  @param bind Flag for binding species (default False)
#  @param bsmi Flag for SMILES binding species (default False)
#  @return Complex name


def name_complex(rootdir: str, core, geometry, ligs, ligoc, sernum,
                 args, nconf=False, sanity=False, bind=False, bsmi=False) -> str:
    # new version of the above, designed to
    # produce more human and machine-readable formats
    if args.name:  # if set externerally
        name = rootdir+'/'+args.name
    else:
        center = ''
        if sanity:
            center += 'badjob_'
        try:
            center += core.getAtom(0).symbol().lower()
        except AttributeError:
            if ('.xyz' in core):
                core = core.split('.')[0]
            center += str(core).lower()
        name = rootdir + '/' + center
        if args.oxstate:
            if args.oxstate in list(romans.keys()):
                ox = str(romans[args.oxstate])
            else:
                ox = str(args.oxstate)
        else:
            ox = "0"
        name += "_" + str(geometry)
        name += "_" + str(ox)
        if args.spin:
            spin = str(args.spin)
        else:
            spin = "0"
        licores = getlicores()
        sminum = 0
        for i, lig in enumerate(ligs):
            if lig not in licores:  # indicative of a SMILES string, or a misspelled ligand
                # Checking if it is likely a misspelling
                text_similarities = [difflib.SequenceMatcher(None, lig, i).ratio() for i in list(licores.keys())]
                if max(text_similarities) > 0.6:  # likely a misspelling of a ligand that is in ligands.dict
                    max_similarity = max(text_similarities)
                    index_max = text_similarities.index(max_similarity)
                    desired_ligand = list(licores.keys())[index_max]
                    name += '_' + str(desired_ligand) + '_' + str(ligoc[i])
                else:  # SMILES string
                    lig = lig.split('\t')[0]
                    sminum += 1
                    name += '_smi' + str(int(sernum)+int(sminum)
                                         ) + '_' + str(ligoc[i])
            else:  # ligand is in ligands.dict
                name += '_' + str(lig) + '_' + str(ligoc[i])
        name += "_s_"+str(spin)
        if args.debug:
            print([nconf, args.nconfs])
        if nconf and int(args.nconfs) >= 1:
            name += "_conf_"+str(nconf)
        if args.bind:
            if bsmi:
                if args.nambsmi:  # if name specified use it in file
                    name += "_" + +args.nambsmi[0:2]
        if args.antigeoisomer:
            name += '_antigeoisomer'
    return name

# Generate complex name (this is actually used instead of namegen.py)
#  @param rootdir Root directory
#  @param core mol3D of core
#  @param ligs List of ligand names
#  @param ligoc List of ligand occurrences
#  @param sernum Complex serial number
#  @param args Namespace of arguments
#  @param bind Flag for binding species (default False)
#  @param bsmi Flag for SMILES binding species (default False)
#  @return Complex name


def name_ts_complex(rootdir, core, geometry, ligs, ligoc, substrate, subcatoms,
                    mlig, mligcatoms, sernum, args, nconf=False, sanity=False,
                    bind=False, bsmi=False) -> str:
    # new version of the above, designed to
    # produce more human and machine-readable formats
    if args.name:  # if set externerally
        name = rootdir+'/'+args.name
    else:
        center = ''
        if sanity:
            center += 'badjob_'
        try:
            center += core.getAtom(0).symbol().lower()
        except AttributeError:
            if ('.xyz' in core):
                core = core.split('.')[0]
            center += str(core).lower()
        name = rootdir + '/' + center
        if args.oxstate:
            if args.oxstate in list(romans.keys()):
                ox = str(romans[args.oxstate])
            else:
                ox = str(args.oxstate)
        else:
            ox = "0"
        name += "_" + str(geometry)
        name += "_" + str(ox)
        licores = getlicores()
        sminum = 0
        for i, lig in enumerate(ligs):
            if lig not in licores:
                # unused:
                # lig = lig.split('\t')[0]
                sminum += 1
                name += '_smi' + str(int(sernum)+int(sminum)
                                     ) + '_' + str(ligoc[i])
            else:
                name += '_' + str(lig) + '_' + str(ligoc[i])
        # for i,sub in enumerate(substrate):
        #     name += "_" + str(sub)
        #     for i,subcatom in enumerate(str(subcatoms))):
        #         name += "_" + str(subcatom)
        for sub in substrate:
            if '.' in sub:
                sub = sub.split('.')[0]
            name += "_" + str(sub)
        for subcatom in subcatoms:
            name += "_" + str(subcatom)
        # for i,mlig_i in enumerate(mlig):
        #     name += "_" + str(mlig)
        #     for j,mligcatom in enumerate(mligcatoms):
        #         name += "_" + str(mligcatom)
        if mlig:
            name += "_" + str(mlig[0])
        if mligcatoms:
            name += "_" + str(mligcatoms[0])
        if args.spin:
            spin = str(args.spin)
        else:
            spin = "0"
        name += "_s_"+str(spin)
        if nconf and int(args.nconfs) >= 1:
            name += "_conf_"+str(nconf)
        if args.bind:
            if bsmi:
                if args.nambsmi:  # if name specified use it in file
                    name += "_" + +args.nambsmi[0:2]
    return name

# ## Generate transition state name
# #  @param rootdir Root directory
# #  @param core mol3D of core
# #  @param subst mol3D of substrate
# #  @param args Namespace of arguments
# #  @param bind Flag for binding species (default False)
# #  @param bsmi Flag for SMILES binding species (default False)
# #  @return Transition state name
# def name_TS(rootdir,core,subst,args,bind= False,bsmi=False):
#     ## new version of the above, designed to
#     ## produce more human and machine-readable formats
#     globs = globalvars()
#     if args.name: # if set externerally
#         name = rootdir+'/'+args.name
#     else:
#         try:
#             center = core.getAtom(0).symbol().lower()
#         except AttributeError:
#             center = str(core).lower()
#         name = rootdir + '/' + center
#         #if args.oxstate:
#             #if args.oxstate in romans.keys():
#                 #ox = str(romans[args.oxstate])
#             #else:
#                 #ox = str(args.oxstate)
#         #else:
#             #ox = "0"
#         #name += "_" + str(ox)
#         if args.spin:
#             spin = str(args.spin)
#         else:
#             spin = "0"
#         name += "_s_"+str(spin)
#         name += "_" + str(subst.ident) + "_TS"
#         if args.bind:
#             if bsmi:
#                 if args.nambsmi: # if name specified use it in file
#                     name += "_" + +args.nambsmi[0:2]
#     return name

# Copies ligands, binding species and cores to user-specified path


def copy_to_custom_path():
    globs = globalvars()
    if not globs.custom_path:
        print('Error, custom path not set!')
        raise FileNotFoundError('Error, custom path not set!')
    # create folder
    if not os.path.exists(globs.custom_path):
        os.makedirs(globs.custom_path)
    # copytree cannot overwrite, need to enusre directory does not exist already
    core_dir = resource_files("molSimplify").joinpath("Cores")
    li_dir = resource_files("molSimplify").joinpath("Ligands")
    bind_dir = resource_files("molSimplify").joinpath("Bind")
    data_dir = resource_files("molSimplify").joinpath("Data")
    subs_dir = resource_files("molSimplify").joinpath("Substrates")
    if os.path.exists(str(globs.custom_path).rstrip("/")+"/Cores"):
        print('Note: removing old molSimplify data')
        shutil.rmtree(str(globs.custom_path).rstrip("/")+"/Cores")
    if os.path.exists(str(globs.custom_path).rstrip("/")+"/Ligands"):
        print('Note: removing old molSimplify data')
        shutil.rmtree(str(globs.custom_path).rstrip("/")+"/Ligands")
    if os.path.exists(str(globs.custom_path).rstrip("/")+"/Bind"):
        print('Note: removing old molSimplify data')
        shutil.rmtree(str(globs.custom_path).rstrip("/")+"/Bind")
    if os.path.exists(str(globs.custom_path).rstrip("/")+"/Data"):
        print('Note: removing old molSimplify data')
        shutil.rmtree(str(globs.custom_path).rstrip("/")+"/Data")
    if os.path.exists(str(globs.custom_path).rstrip("/")+"/Substrates"):
        print('Note: removing old molSimplify data')
        shutil.rmtree(str(globs.custom_path).rstrip("/")+"/Substrates")

    shutil.copytree(core_dir, str(globs.custom_path).rstrip("/")+"/Cores")
    shutil.copytree(li_dir, str(globs.custom_path).rstrip("/")+"/Ligands")
    shutil.copytree(bind_dir, str(globs.custom_path).rstrip("/")+"/Bind")
    shutil.copytree(data_dir, str(globs.custom_path).rstrip("/")+"/Data")
    shutil.copytree(subs_dir, str(globs.custom_path).rstrip("/")+"/Substrates")
