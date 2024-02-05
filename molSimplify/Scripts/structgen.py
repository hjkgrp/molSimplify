# file structgen.py
#  Main structure generation routine
#
#  Written by Kulik Group
#
#  Department of Chemical Engineering, MIT

import os
import subprocess
import tempfile
try:
    from openbabel import openbabel  # version 3 style import
except ImportError:
    import openbabel  # fallback to version 2
import random
import itertools
import numpy as np
from typing import Any, List, Tuple, Dict, Union, Optional
from molSimplify.Scripts.distgeom import GetConf
from molSimplify.Scripts.geometry import (aligntoaxis2,
                                          best_fit_plane,
                                          checkcolinear,
                                          distance,
                                          getPointu,
                                          kabsch,
                                          midpt,
                                          norm,
                                          PointTranslateSph,
                                          reflect_through_plane,
                                          rotate_around_axis,
                                          rotate_mat,
                                          rotation_params,
                                          setPdistance,
                                          vecangle,
                                          vecdiff)
from molSimplify.Scripts.io import (core_load,
                                    getgeoms,
                                    getinputargs,
                                    getlicores,
                                    lig_load,
                                    loadcoord,
                                    loaddata,
                                    name_complex)
from molSimplify.Classes.atom3D import atom3D
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.rundiag import run_diag
from molSimplify.Classes.globalvars import (elementsbynum,
                                            globalvars,
                                            romans,
                                            )
from molSimplify.Informatics.decoration_manager import (decorate_ligand)
from molSimplify.Classes.ligand import ligand as ligand_class
import logging

logger = logging.getLogger(__name__)
np.seterr(all='raise')


def getbackbcombsall(nums):
    """Gets all possible combinations for connection atoms in geometry in the
    case of forced order or unknown geometry.

    Parameters
    ----------
        nums : list
            List of connection atoms.

    Returns
    -------
        bbcombs : list
            List of possible backbone atom combinations.

    """
    bbcombs = []
    for i in range(1, len(nums)+1):
        bbcombs += list(itertools.combinations(nums, i))
    for i, tup in enumerate(bbcombs):
        bbcombs[i] = list(tup)
    return bbcombs


def getnupdateb(backbatoms: List[List[int]], denticity: int) -> Tuple[List[int], List[List[int]]]:
    """Gets a combination of backbone points that satisfies denticity and updates possible combinations.

    Parameters
    ----------
        backbatoms : list
            List of possible backbone atom combinations.
        denticity : int
            Denticity of ligand.

    Returns
    -------
        batoms : list
            Selected combination of backbone atoms.
        backbatoms : list
            Updated list of possible backbone atom combinations.

    """
    dlist = []
    batoms = []
    # find matching combination
    for bba in backbatoms:
        if len(bba) == denticity:
            batoms = bba
            break
    # loop and find elements to delete
    for ba in batoms:
        for i, bcomb in enumerate(backbatoms):
            if ba in bcomb and i not in dlist:
                dlist.append(i)
    dlist.sort(reverse=True)  # sort
    # delete used points
    for i in dlist:
        del backbatoms[i]
    if len(batoms) < 1:
        print('No more connecting points available..')
    return batoms, backbatoms


def init_ANN(args, ligands: List[str], occs: List[int], dents: List[int],
             batslist: List[List[int]], tcats: List[str], licores: dict
             ) -> Tuple[bool, List[Any], str, Dict[str, Any], bool]:
    """Initializes ANN.

    Parameters
    ----------
        args : Namespace
            Namespace of arguments.
        ligands : list
            List of ligands, given as names.
        occs : list
            List of ligand occupations (frequencies of each ligand).
        dents : list
            List of ligand denticities.
        batslist : list
            List of backbond points.
        tcats : list
            List of SMILES ligand connecting atoms.
        licores : dict
            Ligand dictionary within molSimplify.

    Returns
    -------
        ANN_flag : bool
            Whether an ANN call was successful.
        ANN_bondl : float
            ANN predicted bond length.
        ANN_reason : str
            Reason for ANN failure, if failed.
        ANN_attributes : dict
            Dictionary of predicted attributes of complex.
        catalysis_flag : bool
            Whether or not complex is compatible for catalytic ANNs.

    """
    # initialize ANN
    globs = globalvars()
    catalysis_flag = False
    if args.skipANN:
        print('Skipping ANN')
        ANN_flag = False
        # there needs to be 1 length per possible lig
        ANN_bondl = len([item for items in batslist for item in items])*[False]
        ANN_attributes: Dict[str, Any] = {}
        ANN_reason = 'ANN skipped by user'
        return ANN_flag, ANN_bondl, ANN_reason, ANN_attributes, catalysis_flag

    if args.oldANN:
        print('using old ANN by request')
        from molSimplify.Scripts.nn_prep import ANN_preproc
        ANN_flag, ANN_reason, ANN_attributes = ANN_preproc(
            args, ligands, occs, dents, batslist, tcats, licores)
    else:
        if globs.testTF():
            # new RACs-ANN
            from molSimplify.Scripts.tf_nn_prep import tf_ANN_preproc
            if args.debug:
                print('Using tf_ANN_preproc')
            # Set default value [] in case decoration is not used
            decoration_index = [] if not args.decoration else args.decoration_index

            ANN_flag, ANN_reason, ANN_attributes, catalysis_flag = tf_ANN_preproc(
                args.core, args.oxstate, args.spin, ligands, occs, dents, batslist,
                tcats, licores, args.decoration, decoration_index, args.exchange,
                args.geometry, args.debug)
        else:
            # old MCDL-25
            print('using old ANN because tensorflow/keras import failed')
            from molSimplify.Scripts.nn_prep import ANN_preproc
            ANN_flag, ANN_reason, ANN_attributes = ANN_preproc(
                args, ligands, occs, dents, batslist, tcats, licores)
    if ANN_flag:
        ANN_bondl = ANN_attributes['ANN_bondl']
        if args.debug:
            print(('ANN bond length is ' + str(ANN_bondl) +
                   ' type ' + str(type(ANN_bondl))))

    else:
        # there needs to be 1 length per possible lig
        ANN_bondl = len(
            [item for items in batslist for item in items])*[False]
        if args.debug:
            if ANN_reason == 'found incorrect ligand symmetry':
                # This is a workaround so as to not have to change
                # report files checked by GitHub CI when running test
                # cases, which would require everyone using molSimplify
                # from source to have to git pull the new files before
                # any new commits
                print("ANN call failed with reason: either found "
                      "incorrect ligand symmetry, or see ANN "
                      "messages above")
            else:
                print(("ANN call failed with reason: " + ANN_reason))
    return ANN_flag, ANN_bondl, ANN_reason, ANN_attributes, catalysis_flag


def init_template(args, cpoints_required: int,
                  globs) -> Tuple[mol3D, mol3D, str, list, int, mol3D]:
    """Initializes core and template mol3Ds and properties.

    Parameters
    ----------
        args : Namespace
            Namespace of arguments.
        cpoints_required : int
            Number of connecting points required.
        globs : dict
            Globalvars dictionary. Globalvars class in molSimplify.

    Returns
    -------
        m3D : mol3D
            Template complex mol3D instance.
        core3D : mol3D
            Core mol3D instance.
        geom : str
            Geometry used.
        backbatoms : list
            List of backbone atoms.
        coord : int
            Coordination number.
        corerefatoms : mol3D
            Core reference atom index, mol3D instance.

    """
    # initialize core and template
    core3D = mol3D()
    m3D = mol3D()
    # container for ordered list of core reference atoms
    corerefatoms = mol3D()
    # geometry load flag
    geom = 'unknown'
    backbatoms = []
    coord = 0
    # build mode
    if args.geometry and not args.ccatoms:
        # determine geometry
        coord = int(args.coord)
        # get available geometries
        coords, geomnames, geomshorts, geomgroups = getgeoms()
        # get list of possible combinations for connecting points
        bbcombsdict = globs.bbcombs_mononuc()
        # get a default geometry
        geom = geomgroups[coord-1][0]
        # check if geometry is defined and overwrite
        if args.geometry in geomshorts:
            geom = args.geometry
        elif  args.geometry in geomnames:
            geom = geomshorts[geomnames.index(args.geometry)]
        else:
            emsg = "Requested geometry not available." + \
                "Defaulting to "+geomgroups[coord-1][0]
            if args.gui:
                from Classes.mWidgets import mQDialogWarn
                qqb = mQDialogWarn('Warning', emsg)
                qqb.setParent(args.gui.wmain)
            print(emsg)
        # load predefined backbone coordinates
        corexyz = loadcoord(geom)
        # load backbone atom combinations
        if geom in list(bbcombsdict.keys()) and not args.ligloc:
            backbatoms = bbcombsdict[geom]
        else:
            nums = list(range(1, len(corexyz)))
            backbatoms = getbackbcombsall(nums)
        # distort if requested
        if args.pangles:
            corexyz = modifybackbonep(
                corexyz, args.pangles)  # point distortion
        if args.distort:
            corexyz = distortbackbone(
                corexyz, args.distort)  # random distortion
        # add center atom
        if args.core[0].upper()+args.core[1:] in elementsbynum:
            centeratom = args.core[0].upper()+args.core[1:]
        else:
            print('WARNING: Core is not an element. Defaulting to Fe')
            centeratom = 'Fe'
        core3D.addAtom(atom3D(centeratom, corexyz[0]))
        m3D.copymol3D(core3D)
        # add connecting points to template
        for m in range(1, coord+1):
            m3D.addAtom(atom3D('X', corexyz[m]))
            corerefatoms.addAtom(core3D.getAtom(0))
            # corerefatoms.append(0)

    # functionalize mode
    else:
        # check ccatoms
        if not args.ccatoms:
            emsg = 'Connection atoms for custom core not specified. Defaulting to 1!\n'
            print(emsg)
            if args.gui:
                from Classes.mWidgets import mQDialogWarn
                qqb = mQDialogWarn('Warning', emsg)
                qqb.setParent(args.gui.wmain)
        ccatoms = args.ccatoms if args.ccatoms else [0]
        coord = len(ccatoms)
        if args.debug:
            print(('setting ccatoms ' + str(ccatoms)))

        # load core
        core, emsg = core_load(args.core)
        if core is None or emsg:
            raise ValueError(emsg)
        core.convert2mol3D()
        core3D.copymol3D(core)
        m3D.copymol3D(core3D)
        for i in range(cpoints_required):
            if not args.replig:
                # not replacing ligands: add Xs to ccatoms
                # NOTE: ccatoms should be a list with # elements = cpoints_required
                cpoint = getconnection(m3D, ccatoms[i], 2)
                # store core reference atom
                conatom3D = atom3D(core3D.getAtom(
                    ccatoms[i]).sym, core3D.getAtom(ccatoms[i]).coords())
                corerefatoms.addAtom(conatom3D)
                # corerefatoms.append(ccatoms[i])
                # add connecting points to template
                m3D.addAtom(atom3D(Sym='X', xyz=cpoint))
            else:
                try:
                    # replacing ligands
                    cpoint = core3D.getAtom(ccatoms[i]).coords()
                    conatoms = core3D.getBondedAtoms(ccatoms[i])
                    # find smaller submolecule, i.e., ligand to remove
                    minmol = 10000
                    mindelats = []
                    atclose = 0
                    # loop over different connected atoms
                    for cat in conatoms:
                        # find submolecule
                        delatoms = core3D.findsubMol(ccatoms[i], cat)
                        if len(delatoms) < minmol:  # check for smallest
                            mindelats = delatoms
                            minmol = len(delatoms)  # size
                            atclose = cat  # connection atom
                        # if same atoms in ligand get shortest distance
                        elif len(delatoms) == minmol:
                            d0 = core3D.getAtom(ccatoms[i]).distance(
                                core3D.getAtom(cat))
                            d1 = core3D.getAtom(ccatoms[i]).distance(
                                core3D.getAtom(mindelats[0]))
                            if d0 < d1:
                                mindelats = delatoms
                                atclose = cat
                    # store core reference atom
                    conatom3D = atom3D(core3D.getAtom(
                        atclose).sym, core3D.getAtom(atclose).coords())
                    corerefatoms.addAtom(conatom3D)
                    # corerefatoms.append(atclose)
                    delatoms = mindelats
                    # add connecting points to template
                    m3D.addAtom(atom3D(Sym='X', xyz=cpoint))
                    # for multidentate ligands: if a submolecule contains multiple ccatoms, add all of them to the template
                    for atomidx in delatoms:
                        if atomidx in ccatoms[i+1:]:
                            # add connecting points to template
                            m3D.addAtom(
                                atom3D(Sym='X', xyz=core3D.getAtom(atomidx).coords()))
                            ccatoms.remove(atomidx)
                            corerefatoms.addAtom(conatom3D)
                    # update remaining ccatoms according to deleted atoms
                    if len(ccatoms) > i+1:
                        for cccat in range(i+1, len(ccatoms)):
                            lshift = len(
                                [a for a in delatoms if a < ccatoms[cccat]])
                            ccatoms[cccat] -= lshift
                    # delete submolecule
                    core3D.deleteatoms(delatoms)
                    m3D.deleteatoms(delatoms)

                except IndexError:
                    pass
            nums = m3D.findAtomsbySymbol('X')
            backbatoms = getbackbcombsall(nums)
    # set charge from oxidation state if desired
    if args.calccharge:
        if args.oxstate:
            if args.oxstate in list(romans.keys()):
                core3D.charge = int(romans[args.oxstate])
            else:
                core3D.charge = int(args.oxstate)
    return m3D, core3D, geom, backbatoms, coord, corerefatoms


def init_ligand(args, lig, tcats, keepHs, i):
    """Initializes ligand 3D geometry and properties.

    Parameters
    ----------
        args : Namespace
            Namespace of arguments.
        lig : mol3D
            mol3D instance of the ligand.
        tcats : list
            List of SMILES ligand connecting atoms.
        keepHs : bool
            Flag for keeping H atoms on connecting atoms.
        i : int
            Ligand index.

    Returns
    -------
        lig3D : mol3D
            Ligand mol3D instance.
        rempi : bool
            Flag for pi coordination.
        ligpiatoms : list
            List of pi coordinating atoms.

    """
    globs = globalvars()
    rempi = False
    # if SMILES string, copy connecting atoms list to mol3D properties
    if not lig.cat and tcats[i]:
        if 'c' in tcats[i]:
            lig.cat = [lig.natoms]
        else:
            lig.cat = tcats[i]
    # change name
    lig3D = mol3D()
    lig3D.copymol3D(lig)
    # check for pi-coordinating ligand
    ligpiatoms = []
    if 'pi' in lig.cat:
        lig3Dpiatoms = mol3D()
        for k in lig.cat[:-1]:
            lig3Dpiatoms.addAtom(lig3D.getAtom(k))
            lig3Dpiatoms.addAtom(lig3D.getAtom(k))
        ligpiatoms = lig.cat[:-1]
        lig3D.addAtom(atom3D('C', lig3Dpiatoms.centermass()))
        lig.cat = [lig3D.natoms-1]
        rempi = True
    # perform FF optimization if requested (not supported for pi-coordinating ligands)
    if args.ff and 'b' in args.ffoption and not rempi:
        if 'b' in lig.ffopt.lower():
            if args.debug:
                print('FF optimizing ligand')
            lig3D.convert2mol3D()
            lig3D, enl = ffopt(args.ff, lig3D, lig3D.cat, 0,
                               [], False, [], 100, debug=args.debug)
    # skip hydrogen removal for pi-coordinating ligands
    if not rempi:
        # check smarts match
        if 'auto' in keepHs[i]:
            for j, catom in enumerate(lig.cat):
                match = findsmarts(lig3D.OBMol, globs.remHsmarts, catom)
                if match:
                    keepHs[i][j] = False
                else:
                    keepHs[i][j] = True
        # remove one hydrogen from each connecting atom with keepH false
        for j, cat in enumerate(lig.cat):  # lig.cat are the connecting atoms
            Hs = lig3D.getHsbyIndex(cat)
            if len(Hs) > 0 and not keepHs[i][j]:
                if args.debug:
                    print(f'modifying charge down from {lig3D.charge}')
                    try:
                        print('Debug keepHs check\n'
                              f'Removing? {keepHs} \n'
                              f'i = {i}, j = {j}\n'
                              f'lig = \n{lig.coords()}\n'
                              f'keepHs[i]: {keepHs[i]}\n'
                              f'length of keepHs list : {len(keepHs)}')
                    except (AttributeError, IndexError):
                        # Could fail because lig has no Attribute coords
                        # or because keepHs has no element with Index i
                        pass
                # Need to shift all connecting atom indices if they are greater
                # than Hs[0], i.e. the index of the hydrogen atom that is
                # connected to the current connecting atom and is to be removed.
                # Note that only one hydrogen atom is removed at the most under
                # the current implementation.
                for _i, connecting_index in enumerate(lig.cat):
                    if connecting_index > Hs[0]:
                        lig.cat[_i] -= 1
                lig3D.deleteatom(Hs[0])
                lig3D.charge = lig3D.charge - 1
    # Conformer search for multidentate SMILES ligands
    lig3D.convert2OBMol()

    if lig.needsconformer:
        tcats[i] = True
        print(('getting conformers for ' + str(lig.ident)))

    if len(lig.cat) > 1 and tcats[i]:
        lig3D = GetConf(lig3D, args, lig.cat)
    return lig3D, rempi, ligpiatoms


def modifybackbonep(backb, pangles):
    """Distorts backbone according to user specified angles.

    Parameters
    ----------
        backb : List
            List with points comprising the backbone.
        pangles : List
            Pairs of theta/phi angles in DEGREES. Should be list of tuples.

    Returns
    -------
        backb : list
            List of distorted backbone points.

    """
    for i, ll in enumerate(pangles):
        if ll:
            theta = np.pi*float(ll.split('/')[0])/180.0
            phi = np.pi*float(ll.split('/')[-1])/180.0
            backb[i+1] = PointTranslateSph(backb[0], backb[i+1],
                                           [distance(backb[0], backb[i+1]), theta, phi])
    return backb


def distortbackbone(backb, distort):
    """Randomly distorts backbone.

    Parameters
    ----------
        backb : List
            List with points comprising the backbone.
        distort : float
            Percentage of backbone to be distorted.

    Returns
    -------
        backb : list
            List of distorted backbone points.

    """
    for i in range(1, len(backb)):
        theta = random.uniform(0.0, 0.01*int(distort))  # *0.5
        phi = random.uniform(0.0, 0.01*int(distort)*0.5)  # *0.5
        backb[i] = PointTranslateSph(
            backb[0], backb[i], [distance(backb[0], backb[i]), theta, phi])
    return backb


def smartreorderligs(ligs: List[str], dentl: List[int],
                     ligalign: bool = True) -> List[int]:
    """Smart reorder ligands by denticity (-ligalign True)

    Parameters
    ----------
        args : Namespace
            Namespace of arguments.
        ligs : list
            List of ligands as ligand names.
        dentl : list
            List of ligand denticities.

    Returns
    -------
        indices : list
            Reordered ligand indices.

    """

    # reorder ligands
    if not ligalign:
        indices = list(range(0, len(ligs)))
        return indices
    lsizes = []
    for ligand in ligs:
        lig, _ = lig_load(ligand)  # load ligand
        lig.convert2mol3D()
        lsizes.append(lig.natoms)
    # sort ligands into subsets by denticity, since set() sort the items
    # this list goes from lowest to highest denticity, e.g. first list entry
    # contains all monodentate indices, second all bidentates...
    ligdentsidcs = [[i for i, dent in enumerate(dentl) if dent == unique_dent]
                    for unique_dent in set(dentl)]
    # sort by highest denticity first
    ligdentsidcs = list(reversed(ligdentsidcs))
    indices = []
    # within each group sort by size (smaller first)
    for ii, dd in enumerate(ligdentsidcs):
        locs = [lsizes[i] for i in dd]
        locind = [i[0] for i in sorted(enumerate(locs), key=lambda x:x[1])]
        for li in locind:
            indices.append(ligdentsidcs[ii][li])
    return indices


def ffopt(ff: str, mol: mol3D, connected, constopt, frozenats: List[int], frozenangles,
          mlbonds, nsteps: Union[int, str], spin: int = 1, debug: bool = False):
    """Main constrained FF opt routine.

    Parameters
    ----------
        ff : str
            Name force field to use. Available options are MMFF94, UFF,
            Ghemical, GAFF, XTB.
            (XTB only works if the xtb command-line program is installed.)
        mol : mol3D
            mol3D instance of molecule to be optimized.
        connected : list
            List of indices of connection atoms to metal.
        constopt : int
            Flag for constrained optimization -
            0: unconstrained,
            1: fixed connecting atom positions,
            2: fixed connecting atom distances.
        frozenats : list
            List of frozen atom indices.
        frozenangles : bool
            Flag for frozen angles, equivalent to constopt==1.
        mlbonds : list
            List of M-L bonds for distance constraints.
        nsteps : int
            Number of steps to take.
        spin: int
            Spin multiplicity
        debug : bool
            Flag to print extra info to debug.

    Returns
    -------
        mol : mol3D
            Optimized molecule mol3D instance.
        en : float
            Forcefield energy of optimized molecule.

    """
    # check requested force field
    ffav = 'mmff94, uff, ghemical, gaff, mmff94s, xtb, gfnff'  # force fields

    if ff.lower() not in ffav:
        print('Requested force field not available. Defaulting to UFF')
        ff = 'uff'
    if debug:
        print(('using ff: ' + ff))
    if ff.lower() in ['xtb', 'gfnff']:
        return xtb_opt(ff, mol, connected, constopt, frozenats,
                       frozenangles, mlbonds, nsteps, spin=spin, debug=debug)
    return openbabel_ffopt(ff, mol, connected, constopt, frozenats,
                           frozenangles, mlbonds, nsteps, debug=debug)


def openbabel_ffopt(ff: str, mol: mol3D, connected, constopt, frozenats, frozenangles,
                    mlbonds, nsteps: Union[int, str], debug: bool = False) -> Tuple[mol3D, float]:
    """ OpenBabel constraint optimization. To optimize metal-containing
    complexes with MMFF94, an intricate procedure of masking the metal
    atoms and manually editing their valences is applied. OpenBabel's
    implementation of MMFF94 may run extremely slowly on some systems.
    If so, consider switching to UFF.

    Parameters
    ----------
        ff : str
            Name force field to use. Available options are MMFF94, UFF, Ghemical, GAFF.
        mol : mol3D
            mol3D instance of molecule to be optimized.
        connected : list
            List of indices of connection atoms to metal.
        constopt : int
            Flag for constrained optimization
                0: unconstrained,
                1: fixed connecting atom positions,
                2: fixed connecting atom distances.
        frozenats : list
            List of frozen atom indices.
        frozenangles : bool
            Flag for frozen angles, equivalent to constopt==1.
        mlbonds : list
            List of M-L bonds for distance constraints.
        nsteps : int
            Number of steps to take.
        debug : bool
            Flag to print extra info to debug.

    Returns
    -------
        mol : mol3D
            Optimized molecule mol3D instance.
        en : float
            Forcefield energy of optimized molecule.

    """
    metals = list(range(21, 31))+list(range(39, 49))+list(range(72, 81))
    # perform constrained ff optimization if requested after #
    if (constopt > 0):
        # get metal
        midx = mol.findMetal()
        # convert mol3D to OBMol
        mol.convert2OBMol()
        OBMol = mol.OBMol
        # initialize force field
        forcefield = openbabel.OBForceField.FindForceField(ff)
        # initialize constraints
        constr = openbabel.OBFFConstraints()
        # openbabel indexing starts at 1 !!!
        # convert metals to carbons for FF
        indmtls = []
        mtlsnums = []
        for iiat, atom in enumerate(openbabel.OBMolAtomIter(OBMol)):
            if atom.GetAtomicNum() in metals:
                indmtls.append(iiat)
                mtlsnums.append(atom.GetAtomicNum())
                atom.SetAtomicNum(6)
        # freeze and ignore metals
        for midxm in indmtls:
            constr.AddAtomConstraint(midxm+1)  # indexing babel
        # add coordinating atom constraints
        for ii, catom in enumerate(connected):

            if constopt == 1 or frozenangles:
                constr.AddAtomConstraint(catom+1)  # indexing babel
                if debug:
                    print('using connnected opt to freeze atom number: '
                          + str(catom))
            else:
                constr.AddDistanceConstraint(
                    midx[0]+1, catom+1, mlbonds[ii])  # indexing babel
        # print('ff is '+ str(ff))
        if not ff.lower() == "uff":
            bridgingatoms = []
            # identify bridging atoms in the case of bimetallic cores,
            # as well as single-atom ligands (oxo, nitrido)
            # these are immune to deletion
            for i in range(mol.natoms):
                nbondedmetals = len([idx for idx in range(len(mol.getBondedAtoms(
                    i))) if mol.getAtom(mol.getBondedAtoms(i)[idx]).ismetal()])
                if nbondedmetals > 1 or (nbondedmetals == 1 and len(mol.getBondedAtoms(i)) == 1):
                    bridgingatoms.append(i)
            # ensure correct valences for FF setup
            deleted_bonds = 0

            for m in indmtls:
                # first delete all metal-ligand bonds excluding bridging atoms
                for i in range(len(mol.getBondedAtoms(m))):
                    if (OBMol.GetBond(m+1, mol.getBondedAtoms(m)[i]+1) is not None
                            and mol.getBondedAtoms(m)[i] not in bridgingatoms):
                        OBMol.DeleteBond(OBMol.GetBond(
                            m+1, mol.getBondedAtoms(m)[i]+1))
                        # print('FFopt deleting bond')
                        deleted_bonds += 1
                print(('FFopt deleted ' + str(deleted_bonds) + ' bonds'))
                # then add back one metal-ligand bond for FF
                try:
                    numNeighbors = OBMol.GetAtom(m+1).GetValence()
                except AttributeError:
                    # quick workaround for openbabel 3.1.0 compatibility
                    numNeighbors = OBMol.GetAtom(m + 1).GetExplicitDegree()
                if numNeighbors == 0:
                    # getBondedAtomsOct(m,deleted_bonds+len(bridgingatoms)):
                    for i in mol.getBondedAtoms(m):
                        # quick workaround for openbabel 3.1.0 compatibility
                        try:
                            _numNeighbors = OBMol.GetAtom(m+1).GetValence()
                        except AttributeError:
                            _numNeighbors = OBMol.GetAtom(m + 1).GetExplicitDegree()
                        if _numNeighbors < 1 and i not in bridgingatoms:
                            OBMol.AddBond(m+1, i+1, 1)
        # freeze small ligands
        for cat in frozenats:
            if debug:
                print(('using frozenats to freeze atom number: ' + str(cat)))
            constr.AddAtomConstraint(cat+1)  # indexing babel
        if debug:
            # for iiat, atom in enumerate(openbabel.OBMolAtomIter(OBMol)):
            #     print((' atom '+str(iiat)+' atomic num '+str(atom.GetAtomicNum())+' valence ' +
            #            str(atom.GetValence()) + ' is fixed ' + str(constr.IsFixed(iiat+1))))
            # Note: Commented out the preceding for loop because it was
            # throwing the following error:
            # AttributeError: 'OBAtom' object has no attribute 'GetValence'
            print('Commented out')
        # set up forcefield
        s = forcefield.Setup(OBMol, constr)
        if not s:
            print('FF setup failed')
        # force field optimize structure
        elif nsteps == 'Adaptive':
            i = 0
            while i < 20:
                forcefield.ConjugateGradients(50)
                forcefield.GetCoordinates(OBMol)
                mol.OBMol = OBMol
                mol.convert2mol3D()
                overlap, mind = mol.sanitycheck(True)
                if not overlap:
                    break
                i += 1
        elif nsteps != 0:
            n = nsteps
            if debug:
                print(('running ' + str(n) + ' steps'))
            forcefield.ConjugateGradients(n)
            forcefield.GetCoordinates(OBMol)
            mol.OBMol = OBMol
            mol.convert2mol3D()
        else:
            forcefield.GetCoordinates(OBMol)
        en = forcefield.Energy()
        mol.OBMol = OBMol
        # reset atomic number to metal
        for i, iiat in enumerate(indmtls):
            mol.OBMol.GetAtomById(iiat).SetAtomicNum(mtlsnums[i])
        mol.convert2mol3D()
        del forcefield, constr, OBMol
    else:
        # initialize constraints
        constr = openbabel.OBFFConstraints()
        # add atom constraints
        for catom in connected:
            constr.AddAtomConstraint(catom+1)  # indexing babel
        # set up forcefield
        forcefield = openbabel.OBForceField.FindForceField(ff)
        # if len(connected) < 2:
        # mol.OBMol.localopt('mmff94',100) # add hydrogens and coordinates
        OBMol = mol.OBMol  # convert to OBMol
        _ = forcefield.Setup(OBMol, constr)
        # force field optimize structure
        if OBMol.NumHvyAtoms() > 10:
            if debug:
                print('doing 50 steps')
            forcefield.ConjugateGradients(50)
        else:
            if debug:
                print('doing 200 steps')
            forcefield.ConjugateGradients(200)
        forcefield.GetCoordinates(OBMol)
        en = forcefield.Energy()
        mol.OBMol = OBMol
        mol.convert2mol3D()
        del forcefield, constr, OBMol
    return mol, en


def xtb_opt(ff: str, mol: mol3D, connected, constopt, frozenats, frozenangles,
            mlbonds, nsteps: Union[int, str], spin: int = 1, inertial: bool = False,
            debug: bool = False) -> Tuple[mol3D, float]:
    """ XTB optimization. Writes an input file (xtb.in) containing
    all the constraints and parameters to a temporary folder,
    executes the XTB program using the subprocess module and parses
    the output.

    Parameters
    ----------
        ff : str
            Name force field to use. Only option for now is XTB.
        mol : mol3D
            mol3D instance of molecule to be optimized.
        connected : list
            List of indices of connection atoms to metal.
        constopt : int
            Flag for constrained optimization -
            0: unconstrained,
            1: fixed connecting atom positions,
            2: fixed connecting atom distances.
        frozenats : list
            List of frozen atom indices.
        frozenangles : bool
            Flag for frozen angles, equivalent to constopt==1.
        mlbonds : list
            List of M-L bonds for distance constraints.
        nsteps : int
            Number of steps to take.
        spin: int
            Spin multiplicity
        inertial: bool
            Flag for the fast inertial relaxation engine (FIRE)
        debug : bool
            Flag to print extra info to debug.

    Returns
    -------
        mol : mol3D
            Optimized molecule mol3D instance.
        en : float
            Forcefield energy of optimized molecule.

    """
    logger.debug(f'xtbopt() called with {mol.natoms} atoms '
                 f'constopt: {constopt}, frozenats: {frozenats}, '
                 f'frozenangles: {frozenangles}, nsteps: {nsteps}, '
                 f'spin {spin}, inertial {inertial}')
    if nsteps == 'Adaptive':
        # While a similar concept to adaptive would be to set nsteps = 0
        # which corresponds to "automatic" mode in xtb, here the maximum
        # number of steps is just restricted to the same maximum used in
        # adaptive mode: 20*50 = 1000
        nsteps = 1000
    # Initialize defailed input file with optimization parameters.
    input_lines = ['$opt\n', f'maxcycle={nsteps}\n']
    if inertial:
        # engine=inertial is selected in cases if the generation of approximate
        # Hessian coordinates (AHC) fails e.g.: for highly symmetric systems.
        input_lines.append('engine=inertial\n')
    # Arguments for the commandline call of the xtb program
    cmdl_args = ['--opt', 'normal', '--input', 'xtb.inp']
    if ff.lower() == 'gfnff':
        cmdl_args.append('--gfnff')

    # Extract charge (and spin)
    if mol.charge != 0:
        input_lines.append(f'$chrg {mol.charge}\n')
        # xtb uses number of unpaired electrons (Nalpha - Nbeta) instead
        # of multiplicity to define the spin state.
        input_lines.append(f'$spin {spin-1}\n')

    if constopt > 0:  # constrained optimization:
        # List of user selected frozen atoms
        frozen_atoms = frozenats
        # Add all metal atoms
        for i, atom in enumerate(mol.getAtoms()):
            if atom.ismetal():
                frozen_atoms.append(i)

        if constopt == 1 or frozenangles:  # Freeze connecting atoms
            frozen_atoms += connected
        else:  # Contrain bond lengths
            raise NotImplementedError(
                'Bond length constraint XTB optimization '
                'not yet implemented')
        input_lines.append('$fix\n')
        # xtb uses indices starting from 1
        ids = ','.join([str(i+1) for i in frozen_atoms])
        input_lines.append(f'atoms: {ids}\n')

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write detailed input file
        with open(os.path.join(tmpdir, 'xtb.inp'), 'w') as fout:
            fout.writelines(input_lines)
            fout.write('$end\n')
        # Write .xyz file
        mol.writexyz(os.path.join(tmpdir, 'tmp.xyz'))
        # Run xtb using the cmdl args and capture the stdout
        try:
            output = subprocess.run(
                ['xtb'] + cmdl_args + ['tmp.xyz'],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except FileNotFoundError:
            raise ChildProcessError('Could not find subprocess xtb. Ensure xtb'
                                    ' is installed and properly configured.')
        if output.returncode != 0:
            if b'ANC generation failed!' in output.stdout:
                print('Switching xtb_opt to inertial engine.')
                return xtb_opt(ff, mol, connected, constopt, frozenats,
                               frozenangles, mlbonds, nsteps, spin=spin,
                               inertial=True, debug=debug)
            else:
                print(output)
                raise ChildProcessError('XTB calculation failed')
        # Parse geometry, inspired by mol3D.convert2mol3D()
        original_graph = mol.graph
        mol.initialize()
        mol.graph = original_graph
        mol.readfromxyz(os.path.join(tmpdir, 'xtbopt.xyz'))
        # Parse energy from .xyz file comment line
        with open(os.path.join(tmpdir, 'xtbopt.xyz'), 'r') as fout:
            output_lines = fout.readlines()
        en = float(output_lines[1].split()[1])
    return mol, en


def getconnection(core, cidx, BL):
    """Finds the optimum attachment point for an atom/group to a central atom given the desired bond length.
    Objective function maximizes the minimum distance between attachment point and other groups bonded to the central atom.

    Parameters
    ----------
        core : mol3D
            mol3D class instance of the core.
        cidx : int
            Core connecting atom index.
        BL : float
            Optimal core-ligand bond length.

    Returns
    -------
        cpoint : list
            Coordinates of attachment point.

    """
    groups = core.getBondedAtoms(cidx)
    ccoords = core.getAtom(cidx).coords()
    # brute force search
    cpoint = []
    objopt = 0
    for itheta in range(1, 359, 1):
        for iphi in range(1, 179, 1):
            P = PointTranslateSph(ccoords, ccoords, [BL, itheta, iphi])
            dists = []
            for ig in groups:
                dists.append(distance(core.getAtomCoords(ig), P))
            obj = min(dists)
            if obj > objopt:
                objopt = obj
                cpoint = P
    return cpoint


def findsmarts(lig3D, smarts, catom):
    """Checks if connecting atom of lig3D is part of SMARTS pattern.

    Parameters
    ----------
        lig3D : OBMol
            OBMol class instance of ligand. Use convert2OBMol mol3D bound method to obtain it.
        smarts : list
            List of SMARTS patterns (strings).
        catom : int
            connecting atom of lig3D (zero based numbering).

    Returns
    -------
        SMARTS_flag : bool
            SMARTS match flag. True if found, False if not.

    """
    mall = []
    for smart in smarts:
        # initialize SMARTS matcher
        sm = openbabel.OBSmartsPattern()
        sm.Init(smart)
        sm.Match(lig3D)
        matches = list(sm.GetUMapList())
        # unpack tuple
        matches = [i for sub in matches for i in sub]
        for m in matches:
            if m not in mall:
                mall.append(m)
    if catom+1 in mall:
        return True
    else:
        return False


def align_lig_centersym(corerefcoords, lig3D, atom0, core3D, EnableAutoLinearBend):
    """Aligns a ligand's center of symmetry along the metal-connecting atom axis

    Parameters
    ----------
        corerefcoords : list
            Core reference coordinates.
        lig3D : mol3D
            mol3D class instance of the ligand.
        atom0 : int
            Ligand connecting atom index.
        core3D : mol3D
            mol3D instance of partially built complex.
        EnableAutoLinearBend : bool
            Flag for enabling automatic bending of linear ligands (e.g. superoxo).

    Returns
    -------
        lig3D_aligned : mol3D
            mol3D class instance of aligned ligand.

    """
    # rotate to align center of symmetry
    r0 = corerefcoords
    r1 = lig3D.getAtom(atom0).coords()
    lig3Db = mol3D()
    lig3Db.copymol3D(lig3D)
    auxmol = mol3D()
    for at in lig3D.getBondedAtoms(atom0):
        auxmol.addAtom(lig3D.getAtom(at))
    r2 = auxmol.centersym()
    theta, u = rotation_params(r0, r1, r2)
    # rotate around axis and get both images
    lig3D = rotate_around_axis(lig3D, r1, u, theta)
    lig3Db = rotate_around_axis(lig3Db, r1, u, theta-180)
    # compare shortest distances to core reference coordinates
    d2 = distance(r0, lig3D.centersym())
    d1 = distance(r0, lig3Db.centersym())
    lig3D = lig3D if (d1 < d2) else lig3Db  # pick best one
    # additional rotation for bent terminal connecting atom:
    if auxmol.natoms == 1:
        if (distance(auxmol.getAtomCoords(0), lig3D.getAtomCoords(atom0))
                > 0.8*(auxmol.getAtom(0).rad + lig3D.getAtom(atom0).rad)
                and EnableAutoLinearBend):
            print('bending of linear terminal ligand')
            # warning: force field might overwrite this
            # warning: skipping this part because
            # we no longer understand it
            if False:
                globs = globalvars()
                r1 = lig3D.getAtom(atom0).coords()
                r2 = auxmol.getAtom(0).coords()
                theta, u = rotation_params([1, 1, 1], r1, r2)
                lig3D = rotate_around_axis(
                    lig3D, r1, u, -1*globs.linearbentang)
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned


def align_linear_pi_lig(corerefcoords, lig3D, atom0, ligpiatoms):
    """Aligns a linear pi ligand's connecting point to the metal-ligand axis.

    Parameters
    ----------
        corerefcoords : list
            Core reference coordinates.
        lig3D : mol3D
            mol3D class instance of the ligand.
        atom0 : int
            Ligand connecting atom index.
        ligpiatoms : list
            List of ligand pi-connecting atom indices.

    Returns
    -------
        lig3D_aligned : mol3D
            mol3D class instance of aligned ligand.

    """
    # first rotate in the metal plane to ensure perpendicularity
    r0 = corerefcoords
    r1 = lig3D.getAtom(ligpiatoms[0]).coords()
    r2 = lig3D.getAtom(ligpiatoms[1]).coords()
    theta, u = rotation_params(r0, r1, r2)
    objfuncopt = 90
    # thetaopt = 0
    for theta in range(0, 360, 1):
        lig3D_tmp = mol3D()
        lig3D_tmp.copymol3D(lig3D)
        lig3D_tmp = rotate_around_axis(
            lig3D_tmp, lig3D_tmp.getAtom(atom0).coords(), u, theta)
        # objfunc = abs(vecangle(vecdiff(lig3D_tmp.getAtom(atom0).coords(),corerefcoords),
        #                        vecdiff(lig3D_tmp.getAtom(ligpiatoms[0]).coords(),
        #                                lig3D_tmp.getAtom(ligpiatoms[1]).coords())) - 90)
        objfunc = abs(distance(lig3D_tmp.getAtom(ligpiatoms[0]).coords(
        ), corerefcoords) - distance(lig3D_tmp.getAtom(ligpiatoms[1]).coords(), corerefcoords))
        if objfunc < objfuncopt:
            # thetaopt = theta
            objfuncopt = objfunc
            lig3Dopt = mol3D()  # lig3Dopt = lig3D_tmp DOES NOT WORK!!!
            lig3Dopt.copymol3D(lig3D_tmp)
    lig3D = lig3Dopt
    # then rotate 90 degrees about the bond axis to further reduce steric repulsion
    r1 = lig3D.getAtom(ligpiatoms[0]).coords()
    r2 = lig3D.getAtom(ligpiatoms[1]).coords()
    u = vecdiff(r1, r2)
    lig3D_tmpa = mol3D()
    lig3D_tmpa.copymol3D(lig3D)
    lig3D_tmpa = rotate_around_axis(
        lig3D_tmpa, lig3D_tmpa.getAtom(atom0).coords(), u, 90)
    lig3D_tmpb = mol3D()
    lig3D_tmpb.copymol3D(lig3D)
    lig3D_tmpb = rotate_around_axis(
        lig3D_tmpb, lig3D_tmpb.getAtom(atom0).coords(), u, -90)
    d1 = distance(corerefcoords, lig3D_tmpa.centermass())
    d2 = distance(corerefcoords, lig3D_tmpb.centermass())
    # lig3D = lig3D if (d1 < d2)  else lig3Db
    # pick the better structure
    lig3D = lig3D_tmpa if (d1 > d2) else lig3D_tmpb
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned


def rotation_objective_func(rotations, lig3D, atom0, ligpiatoms, metal_lig_vec, directional_vectors):
    """Objective function for finding rotations that make an aromatic ring perpendicular to the metal-ligand vector.

    Parameters
    ----------
        rotations : list
            Floats that indicate angles by which to rotate the ligand. Length is 3.
        lig3D : mol3D
            mol3D class instance of the ligand.
        atom0 : int
            Ligand connecting atom index. Here, refers to the fictitious atom in the center of the aromatic ring.
        ligpiatoms : list
            List of ligand pi-connecting atom indices.
        metal_lig_vec : np.array
            Vector from the metal to the fictitious atom in the center of the aromatic ring. Shape is (3,)
        directional_vectors : list
            Numpy arrays of the x-axis vector, y-axis vector, and z-axis vector. Length is 3.

    Returns
    -------
        lig3D_aligned : mol3D
            mol3D class instance of aligned ligand.

    """
    lig3D_tmp = mol3D()
    lig3D_tmp.copymol3D(lig3D)

    for _i in range(3):  # 3 axes of rotation
        # Three rotations
        rotate_around_axis(lig3D_tmp, lig3D_tmp.getAtom(atom0).coords(), directional_vectors[_i], rotations[_i])

    # Get the best fit plane for the aromatic atoms.
    aromatic_coordinates = np.zeros((3, len(ligpiatoms)))
    for idx, _i in enumerate(ligpiatoms):  # Iterate over the aromatic atoms
        current_coordinates = lig3D_tmp.getAtom(_i).coords()

        for _j in range(3):  # Iterate over the three dimensions of space
            aromatic_coordinates[_j, _i] = current_coordinates[_j]

    normal_vector_plane = best_fit_plane(aromatic_coordinates)  # plane formed by the aromatic ring atoms

    # The roots for this objective function are to be found
    normalized_metal_lig_vec = metal_lig_vec / np.linalg.norm(metal_lig_vec)
    normalized_normal_vector_plane = normal_vector_plane / np.linalg.norm(normal_vector_plane)
    return normalized_metal_lig_vec - normalized_normal_vector_plane


def align_pi_ring_lig(corerefcoords, lig3D, atom0, ligpiatoms, u):
    """Rotates the ligand such that the aromatic ring that bonds to the central metal
    is perpendicular to the vector from the metal to the fictitous atom in the center
    of the ring.

    Parameters
    ----------
        corerefcoords : list
            Core reference coordinates. These are the coordinates of the central metal.
        lig3D : mol3D
            mol3D class instance of the ligand.
        atom0 : int
            Ligand connecting atom index. Here, refers to the fictitious atom in the
            center of the aromatic ring, since we have a ligand that coordinates
            through an aromatic ring.
        ligpiatoms : list
            List of ligand pi-connecting atom indices.
        u : list
            Vector from the metal to the fictitious atom in the center of the aromatic
            ring. Length is 3.

    Returns
    -------
        lig3D : mol3D
            mol3D class instance of aligned ligand.

    """
    x_vec = np.array([1, 0, 0])
    y_vec = np.array([0, 1, 0])
    z_vec = np.array([0, 0, 1])
    directional_vectors = [x_vec, y_vec, z_vec]

    # use a solver to find the rotations around the three vectors (x, y, z) required such that
    # the plane formed by the aromatic ring atoms is perpendicular to u
    from scipy.optimize import fsolve
    initial_guess = [0, 0, 0]
    rotations = fsolve(rotation_objective_func, initial_guess,
                       args=(lig3D, atom0, ligpiatoms, np.array(u), directional_vectors))

    # rotate lig3D by the three rotations found
    for _i in range(3):  # 3 axes of rotation
        rotate_around_axis(lig3D, lig3D.getAtom(atom0).coords(), directional_vectors[_i], rotations[_i])

    return lig3D


def check_rotate_linear_lig(corerefcoords, lig3D, atom0):
    """Checks if ligand has a linear coordination environment (e.g., OCO) and ensures perpendicularity to M-L axis

    Parameters
    ----------
        corerefcoords : list
            Core reference coordinates.
        lig3D : mol3D
            mol3D class instance of the ligand.
        atom0 : int
            Ligand connecting atom index.

    Returns
    -------
        lig3D_aligned : mol3D
            mol3D class instance of rotated ligand.

    """
    auxm = mol3D()
    lig3D_aligned = mol3D()
    for at in lig3D.getBondedAtoms(atom0):
        auxm.addAtom(lig3D.getAtom(at))
    if auxm.natoms > 1:
        r0 = lig3D.getAtom(atom0).coords()
        r1 = auxm.getAtom(0).coords()
        r2 = auxm.getAtom(1).coords()
        if checkcolinear(r1, r0, r2):
            # rotate so that O-C-O bond is perpendicular to M-L axis
            theta, urot = rotation_params(r1, corerefcoords, r2)
            theta = vecangle(vecdiff(r0, corerefcoords), urot)
            lig3D = rotate_around_axis(lig3D, r0, urot, theta)
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned


def check_rotate_symm_lig(corerefcoords, lig3D, atom0, core3D):
    """Aligns a ligand's center of symmetry along the metal-connecting atom axis

    Parameters
    ----------
        corerefcoords : list
            Core reference coordinates.
        lig3D : mol3D
            mol3D class instance of the ligand.
        atom0 : int
            Ligand connecting atom index.
        core3D : mol3D
            mol3D instance of partially built complex.

    Returns
    -------
        lig3D_aligned : mol3D
            mol3D class instance of rotated ligand.

    """
    if distance(lig3D.getAtom(atom0).coords(), lig3D.centersym()) < 8.0e-2:
        at = lig3D.getBondedAtoms(atom0)
        r0 = lig3D.getAtom(atom0).coords()
        r1 = lig3D.getAtom(at[0]).coords()
        r2 = lig3D.getAtom(at[1]).coords()
        theta, u = rotation_params(r0, r1, r2)
        theta = vecangle(u, vecdiff(r0, corerefcoords))
        urot = np.cross(u, vecdiff(r0, corerefcoords))
        # rotate around axis and get both images
        lig3Db = mol3D()
        lig3Db.copymol3D(lig3D)
        lig3D = rotate_around_axis(lig3D, r0, urot, theta)
        lig3Db = rotate_around_axis(lig3Db, r0, urot, -theta)
        # compute shortest distances to core
        d2 = lig3D.mindist(core3D)
        d1 = lig3Db.mindist(core3D)
        lig3D = lig3D if (d1 < d2) else lig3Db  # pick best one
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned


def rotate_MLaxis_minimize_steric(corerefcoords, lig3D, atom0, core3D):
    """Rotates aligned ligand about M-L axis to minimize steric clashes with rest of complex

    Parameters
    ----------
        corerefcoords : list
            Core reference coordinates.
        lig3D : mol3D
            mol3D class instance of the ligand.
        atom0 : int
            Ligand connecting atom index.
        core3D : mol3D
            mol3D instance of partially built complex.

    Returns
    -------
        lig3D_aligned : mol3D
            mol3D class instance of rotated ligand.

    """
    r1 = lig3D.getAtom(atom0).coords()
    u = vecdiff(r1, corerefcoords)
    dtheta = 2
    optmax = -9999
    totiters = 0
    lig3Db = mol3D()
    lig3Db.copymol3D(lig3D)
    # maximize a combination of minimum distance between atoms and center of mass distance
    while totiters < 180:
        lig3D = rotate_around_axis(lig3D, r1, u, dtheta)
        d0 = lig3D.mindist(core3D)  # shortest distance
        d0cm = lig3D.distance(core3D)  # center of mass distance
        iteropt = d0cm+10*np.log(d0)
        if (iteropt > optmax):  # if better conformation, keep
            lig3Db = mol3D()
            lig3Db.copymol3D(lig3D)
            optmax = iteropt
        totiters += 1
    lig3D = lig3Db
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned


def rotate_catom_fix_Hs(lig3D, catoms, n, mcoords, core3D):
    """Rotates a connecting atom of a multidentate ligand to improve H atom placement.
    There are separate routines for terminal connecting atoms and intermediate connecting atoms.

    Parameters
    ----------
        lig3D : mol3D
            mol3D class instance of the ligand.
        catoms : list
            List of ligand connecting atom indices.
        n : int
            Index of connecting atom.
        mcoords : list
            Coordinates of a core reference (usually a metal).
        core3D : mol3D
            mol3D of partially built complex.

    Returns
    -------
        lig3D_aligned : mol3D
            mol3D class instance of rotated ligand.

    """
    # isolate fragment to be rotated
    confrag3D = mol3D()
    confragatomlist = []
    danglinggroup = []
    catoms_other = catoms[:]
    catoms_other.pop(n)
    # add connecting atom
    confrag3D.addAtom(lig3D.getAtom(catoms[n]))
    confragatomlist.append(catoms[n])
    # add all Hs bound to connecting atom
    for ii in lig3D.getHsbyIndex(catoms[n]):
        confrag3D.addAtom(lig3D.getAtom(ii))
        confragatomlist.append(ii)
    # add dangling groups
    anchoratoms = []
    for atom in lig3D.getBondedAtomsnotH(catoms[n]):
        subm = lig3D.findsubMol(atom, catoms[n])
        if len(list(set(subm).intersection(catoms_other))) == 0:
            danglinggroup = subm
        else:
            # bridginggroup = subm
            if list(set(subm).intersection(lig3D.getBondedAtoms(catoms[n])))[0] not in anchoratoms:
                anchoratoms.append(list(set(subm).intersection(
                    lig3D.getBondedAtoms(catoms[n])))[0])

    if not len(anchoratoms) == 1:
        for atom in danglinggroup:
            confrag3D.addAtom(lig3D.getAtom(atom))
            confragatomlist.append(atom)
        if confrag3D.natoms > 1:
            # terminal connecting atom
            confrag3Dtmp = mol3D()
            confrag3Dtmp.copymol3D(confrag3D)
            if len(anchoratoms) == 1:
                anchoratom = anchoratoms[0]
                anchor = lig3D.getAtomCoords(anchoratom)
                if not checkcolinear(anchor, confrag3D.getAtomCoords(0), confrag3D.getAtomCoords(1)):
                    refpt = confrag3D.getAtomCoords(0)
                    u = vecdiff(refpt, anchor)
                    dtheta = 5
                    # objs = []
                    objopt = 0
                    thetaopt = 0
                    # localmaxs = []
                    thetas = list(range(0, 360, dtheta))
                    for theta in thetas:
                        confrag3Dtmp = rotate_around_axis(
                            confrag3Dtmp, refpt, u, dtheta)
                        auxmol1 = mol3D()
                        auxmol1.addAtom(confrag3Dtmp.getAtom(0))
                        for at in confrag3Dtmp.getBondedAtoms(0):
                            auxmol1.addAtom(confrag3Dtmp.getAtom(at))
                        auxmol1.addAtom(lig3D.getAtom(anchoratom))
                        auxmol2 = mol3D()
                        auxmol2.copymol3D(confrag3Dtmp)
                        # objs.append(distance(mcoords,auxmol.centersym()))
                        if auxmol2.natoms > 3:
                            obj = auxmol2.mindisttopoint(mcoords)
                        else:
                            obj = distance(mcoords, auxmol1.centersym())
                        if obj > objopt:
                            objopt = obj
                            thetaopt = theta
                    # for i,obj in enumerate(objs):
                        # try:
                            # if objs[i] > objs[i-1] and objs[i] > objs[i+1]:
                            # localmaxs.append(thetas[i])
                        # except IndexError:
                            # pass
                    # in future, compare multiple local maxima
                    # if localmaxs == []:
                    #     localmaxs = [0]
                    confrag3D = rotate_around_axis(confrag3D, refpt, u, thetaopt)
                # confrag3D = rotate_around_axis(confrag3D,refpt,u,localmaxs[0])
            # non-terminal connecting atom
            elif len(anchoratoms) == 2:
                refpt = confrag3D.getAtomCoords(0)
                anchorcoords1 = lig3D.getAtomCoords(anchoratoms[0])
                anchorcoords2 = lig3D.getAtomCoords(anchoratoms[1])
                u = vecdiff(anchorcoords1, anchorcoords2)
                dtheta = 5
                objs = []
                localmaxs = []
                thetas = list(range(0, 360, dtheta))
                for _ in thetas:
                    confrag3Dtmp = rotate_around_axis(
                        confrag3Dtmp, refpt, u, dtheta)
                    newHcoords = confrag3Dtmp.getAtomCoords(1)
                    objs.append(distance(newHcoords, anchorcoords1)+distance(
                        newHcoords, anchorcoords2)+distance(newHcoords, mcoords))
                for i, obj in enumerate(objs):
                    try:
                        if objs[i] > objs[i-1] and objs[i] > objs[i+1]:
                            localmaxs.append(thetas[i])
                    except IndexError:
                        pass
                if localmaxs == []:
                    localmaxs = [0]
                confrag3D = rotate_around_axis(
                    confrag3D, refpt, u, localmaxs[0])
            for i, atom in enumerate(confragatomlist):
                lig3D.getAtom(atom).setcoords(confrag3D.getAtomCoords(i))
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned


def rotate_catoms_fix_Hs(lig3D: mol3D, catoms: List[int], mcoords, core3D: mol3D) -> mol3D:
    """Rotates connecting atoms of multidentate ligands to improve H atom placement.
    Loops over rotate_catom_fix_Hs().

    Parameters
    ----------
        lig3D : mol3D
            mol3D class instance of the ligand.
        catoms : list
            List of ligand connecting atom indices.
        mcoords : list
            Coordinates of a core reference (usually a metal).
        core3D : mol3D
            mol3D of partially built complex.

    Returns
    -------
        lig3D_aligned : mol3D
            mol3D class instance of rotated ligand.

    """
    for i, n in enumerate(catoms):
        # if len(lig3D.getHsbyIndex(n)) > 0:
        lig3D = rotate_catom_fix_Hs(lig3D, catoms, i, mcoords, core3D)
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned


def get_MLdist(metal: atom3D, oxstate: str, spin: str, lig3D: mol3D,
               atom0: int, ligand: str, MLb: List[str], i: int,
               ANN_flag: bool, ANN_bondl: float, this_diag: run_diag,
               MLbonds: dict, debug: bool = False) -> float:
    """Gets target M-L distance from desired source (custom, sum cov rad or ANN).
    Aligns a monodentate ligand to core connecting atom coordinates.

    Parameters
    ----------
        args : Namespace
            Namespace of arguments.
        lig3D : mol3D
            mol3D class instance of the ligand.
        atom0 : int
            Ligand connecting atom index.
        ligand : str
            Name of ligand for dictionary lookup.
        metal : atom3D
            atom3D class instance of the first atom (usually a metal).
        MLb : float
            Custom M-L bond length (if any).
        i : int
            Ligand index number.
        ANN_flag : bool
            Flag for ANN activation.
        ANN_bondl : float
            ANN predicted M-L bond length.
        this_diag : run_diag
            run_diag instance for ANN diagnostic object.
        MLbonds : dict
            M-L bond dictionary.

    Returns
    -------
        bondl : float
            M-L bond length in angstroms.

    """
    # first check for user-specified distances and use them
    # print(MLb, MLb[i])
    if (MLb and MLb[i]) and ("F" not in MLb[i]):
        print('using user-specified M-L distances')
        if 'c' in MLb[i].lower():
            bondl = metal.rad + lig3D.getAtom(atom0).rad
        else:
            bondl = float(MLb[i])
    else:
        # otherwise, check for exact DB match
        bondl, exact_match = get_MLdist_database(
            metal, oxstate, spin, lig3D, atom0, ligand, MLbonds, debug)
        try:
            this_diag.set_dict_bl(bondl)
        except AttributeError:
            pass
        if not exact_match and ANN_flag:
            # if no exact match found and ANN enabled, use it
            if debug:
                print('no exact M-L match in DB, using ANN')
            bondl = ANN_bondl
        elif exact_match:
            print('using exact M-L match from DB')
        else:
            print('Warning: ANN not active and exact M-L match not found in '
                  'DB, distance may not be accurate')
            print(f'using partial DB match distance of {bondl}')
    return bondl


def get_MLdist_database(metal: atom3D, oxstate: str, spin: str, lig3D: mol3D,
                        atom0: int, ligand: str, MLbonds: dict,
                        debug=False) -> Tuple[float, bool]:
    """Gets target M-L distance from desired source (custom, sum cov rad or ANN).
    Aligns a monodentate ligand to core connecting atom coordinates.

    Parameters
    ----------
        metal : atom3D
            atom3D class instance of the first atom (usually a metal).
        oxstate:  str:
            oxidation state
        spin : str
            spin state
        lig3D : mol3D
            mol3D class instance of the ligand.
        atom0 : int
            Ligand connecting atom index.
        ligand : str
            Name of ligand for dictionary lookup.
        MLbonds : dict
            M-L bond dictionary.

    Returns
    -------
        bondl : float
            M-L bond length in angstroms.
        exact_match : bool
            Flag for database match.
    """
    # check for roman letters in oxstate
    if oxstate:  # if defined put oxstate in keys
        if oxstate in romans.keys():
            oxs = romans[oxstate]
        else:
            oxs = oxstate
    else:
        oxs = '-'
    # check for spin multiplicity
    spin = spin if spin else '-'
    # Build possible keys in descending order of specificity
    key = []
    key.append((metal.sym, oxs, spin, lig3D.getAtom(atom0).sym, ligand))
    # disregard exact ligand
    key.append((metal.sym, oxs, spin, lig3D.getAtom(atom0).sym, '-'))
    # disregard oxstate/spin
    key.append((metal.sym, '-', '-', lig3D.getAtom(atom0).sym, ligand))
    # else just consider bonding atom
    key.append((metal.sym, '-', '-', lig3D.getAtom(atom0).sym, '-'))
    exact_match = False
    # search for data
    for kk in key:
        if kk in MLbonds.keys():  # if exact key in dictionary
            bondl = float(MLbonds[kk])
            if (kk == ((metal.sym, oxs, spin, lig3D.getAtom(atom0).sym, ligand))):  # exact match
                exact_match = True
            break
    else:  # If no match in dict (no break encountered):
        # last resort sum of covalent radii
        bondl = metal.rad + lig3D.getAtom(atom0).rad
    if debug:
        print(f'ms default distance is {bondl}')
    return bondl, exact_match


def get_batoms(args, batslist, ligsused):
    """Get backbone atoms from template.

    Parameters
    ----------
        args : Namespace
            Namespace of arguments.
        batslist : list
            List of backbone connecting atoms for each ligand.
        ligsused : int
            Number of ligands placed.

    Returns
    -------
        batoms : list
            Backbone connecting atoms for ligand.

    """
    batoms = batslist[ligsused]
    if len(batoms) < 1:
        emsg = 'Connecting all ligands is not possible. Check your input!'
        if args.gui:
            from Classes.mWidgets import mQDialogWarn
            qqb = mQDialogWarn('Warning', emsg)
            qqb.setParent(args.gui.wmain)
    return batoms


def align_dent2_catom2_coarse(args, lig3D, core3D, catoms, r1, r0, m3D, batoms, corerefcoords):
    """Crude rotations to improve alignment of the 2nd connecting atom of a bidentate substrate.

    Parameters
    ----------
        args : Namespace
            Namespace of arguments.
        lig3D : mol3D
            mol3D class instance of the ligand.
        core3D : mol3D
            mol3D class instance of partially built complex.
        catoms : list
            List of ligand connecting atom indices.
        r1 : list
            Coordinates of ligand first connecting atom.
        r0 : list
            Coordinates of core reference point.
        m3D : mol3D
            mol3D class instance of backbone template.
        batoms : list
            List of backbone atom indices.
        corerefcoords : list
            Coordinates of core reference atom.

    Returns
    -------
        lig3D_aligned : mol3D
            mol3D class instance of aligned ligand.
        r1b : list
            Coordinates of second backbone point.

    """
    r21 = [a-b for a, b in zip(lig3D.getAtom(catoms[1]).coords(), r1)]
    r21n = [a-b for a, b in zip(m3D.getAtom(batoms[1]).coords(), r1)]
    if (norm(r21)*norm(r21n)) > 1e-8:
        theta = 180*np.arccos(np.dot(r21, r21n)/(norm(r21)*norm(r21n)))/np.pi
    else:
        theta = 0.0
    u = np.cross(r21, r21n)
    lig3Db = mol3D()
    lig3Db.copymol3D(lig3D)
    # rotate around axis and get both images
    lig3D = rotate_around_axis(lig3D, r1, u, theta)
    lig3Db = rotate_around_axis(lig3Db, r1, u, theta-180)
    d1 = distance(lig3D.getAtom(
        catoms[1]).coords(), m3D.getAtom(batoms[1]).coords())
    d2 = distance(lig3Db.getAtom(
        catoms[1]).coords(), m3D.getAtom(batoms[1]).coords())
    lig3D = lig3D if (d1 < d2) else lig3Db  # pick best one
    # flip if overlap
    r0l = lig3D.getAtom(catoms[0]).coords()
    r1l = lig3D.getAtom(catoms[1]).coords()
    md = min(distance(r0l, corerefcoords), distance(r1l, corerefcoords))
    if lig3D.mindist(core3D) < md:
        lig3D = rotate_around_axis(lig3D, r0l, vecdiff(r1l, r0l), 180.0)
    # correct plane
    # r0b = m3D.getAtom(batoms[0]).coords()
    r1b = m3D.getAtom(batoms[1]).coords()
    r0l = lig3D.getAtom(catoms[0]).coords()
    r1l = lig3D.getAtom(catoms[1]).coords()
    # rm = lig3D.centermass()
    urot = vecdiff(r1l, r0l)
    # theta,ub = rotation_params(corerefcoords,r0b,r1b)
    # theta,ul = rotation_params(rm,r0l,r1l)
    # if (norm(ub)*norm(ul)) > 1e-8:
    #     theta = 180*np.arccos(np.dot(ub,ul)/(norm(ub)*norm(ul)))/pi-180.0
    # else:
    #     theta = 0.0
    # rotate around axis
    objopt = 0
    for theta in range(0, 360, 5):
        lig3D_tmp = mol3D()
        lig3D_tmp.copymol3D(lig3D)
        lig3D_tmp = rotate_around_axis(lig3D_tmp, r1, urot, theta)
        lig3D_tmp2 = mol3D()
        lig3D_tmp2.copymol3D(lig3D_tmp)
        H1 = lig3D_tmp2.getBondedAtomsH(catoms[1])
        H2 = lig3D_tmp2.getBondedAtomsH(catoms[0])
        lig3D_tmp2.deleteatoms([catoms[1]]+[catoms[0]]+H1+H2)
        obj = lig3D_tmp2.mindisttopoint(corerefcoords)
        if obj > objopt:
            objopt = obj
            lig3Dopt = mol3D()
            lig3Dopt.copymol3D(lig3D_tmp)
    lig3D = mol3D()
    lig3D.copymol3D(lig3Dopt)
    tmp3D = mol3D()
    tmp3D.copymol3D(m3D)
    tmp3D.combine(lig3D)
    # tmp3D.writexyz('new')
    # lig3Db = mol3D()
    # lig3Db.copymol3D(lig3D)
    # lig3D = rotate_around_axis(lig3D,r1,urot,theta)
    # lig3Db = rotate_around_axis(lig3Db,r1,urot,-theta)
    # select best

    # The following block was commented because ub is undefinded after
    # someone previously commented out other parts of this function.
    # Note: this is not a "fix" of the problem and just the simplest
    # solution to stay consistent with previous behavior.
    # try:
    #     rm0, rm1 = lig3D.centermass(), lig3Db.centermass()
    #     theta, ul0 = rotation_params(rm0, r0l, r1l)
    #     theta, ul1 = rotation_params(rm1, r0l, r1l)
    #     th0 = 180*np.arccos(np.dot(ub, ul0)/(norm(ub)*norm(ul0)))/pi
    #     th0 = min(abs(th0), abs(180-th0))
    #     th1 = 180*np.arccos(np.dot(ub, ul1)/(norm(ub)*norm(ul1)))/pi
    #     th1 = min(abs(th1), abs(180-th1))
    #     lig3D = lig3D if th0 < th1 else lig3Db
    # except:
    #     pass
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned, r1b


def align_dent2_catom2_refined(args, lig3D, catoms, bondl, r1, r0, core3D, rtarget, coreref, MLoptbds):
    """Aligns second connecting atom of a bidentate ligand to balance ligand strain and the desired coordination environment.

    Parameters
    ----------
        args : Namespace
            Namespace of arguments.
        lig3D : mol3D
            mol3D class instance of the ligand.
        catoms : list
            List of ligand connecting atom indices.
        bondl : float
            Target M-L bond length.
        r1 : list
            Coordinates of ligand first connecting atom.
        r0 : list
            Coordinates of core reference point.
        core3D : mol3D
            mol3D class instance of partially built complex.
        rtarget : list
            Coordinates of target point for second connecting atom.
        coreref : atom3D
            atom3D of core reference atom.
        MLoptbds : list
            List of final M-L bond lengths.

    Returns
    -------
        lig3D_aligned : mol3D
            mol3D class instance of aligned ligand.

    """
    # compute starting ligand FF energy for later comparison
    corerefcoords = coreref.coords()
    dr = vecdiff(rtarget, lig3D.getAtom(catoms[1]).coords())
    cutoff = 5  # energy threshold for ligand strain, kcal/mol
    lig3Dtmp = mol3D()
    lig3Dtmp.copymol3D(lig3D)
    lig3Dtmp, en_start = ffopt(
        args.ff, lig3Dtmp, [], 1, [], False, [], 200, debug=args.debug)
    # take steps between current ligand position and ideal position on backbone
    nsteps = 20
    ddr = [di/nsteps for di in dr]
    ens = []
    finished = False
    relax = False
    while True:
        lig3Dtmp = mol3D()
        lig3Dtmp.copymol3D(lig3D)
        for ii in range(0, nsteps):
            lig3Dtmp, enl = ffopt(args.ff, lig3Dtmp, [], 1, [
                                  catoms[0], catoms[1]], False, [], 'Adaptive', debug=args.debug)
            ens.append(enl)
            lig3Dtmp.getAtom(catoms[1]).translate(ddr)
            # once the ligand strain energy becomes too high, stop and accept ligand position
            # or if the ideal coordinating point is reached without reaching the strain energy cutoff, stop
            if (ens[-1] - ens[0] > cutoff) or (ii == nsteps-1):
                r0, r1 = lig3Dtmp.getAtomCoords(
                    catoms[0]), lig3Dtmp.getAtomCoords(catoms[1])
                r01 = distance(r0, r1)
                try:
                    # but if ligand still cannot be aligned, instead force
                    # alignment with a huge cutoff and then relax later
                    theta1 = 180*np.arccos(0.5*r01/bondl)/np.pi
                except AssertionError:
                    # To whoever encounters this: Please replace AssertionError
                    # with whatever we are actually trying to except. I am
                    # pretty sure that np.arccos does not raise Exceptions.
                    # RM 2022/02/17
                    print('Forcing alignment...')
                    cutoff += 5000000
                    relax = True
                    break
                theta2 = vecangle(vecdiff(r1, r0), vecdiff(corerefcoords, r0))
                dtheta = theta2-theta1
                theta, urot = rotation_params(corerefcoords, r0, r1)
                # rotate so that it matches bond
                lig3Dtmp = rotate_around_axis(lig3Dtmp, r0, urot, -dtheta)
                finished = True
                break
        if finished:
            break
    # for long linear ligand chains, this procedure might produce the wrong ligand curvature. If so, reflect about M-L plane
    lig3Dtmpb = mol3D()
    lig3Dtmpb.copymol3D(lig3Dtmp)
    lig3Dtmpb = reflect_through_plane(lig3Dtmpb, vecdiff(midpt(lig3Dtmpb.getAtom(catoms[0]).coords(
    ), lig3Dtmpb.getAtom(catoms[1]).coords()), corerefcoords), lig3Dtmpb.getAtom(catoms[0]).coords())
    lig3Dtmp = lig3Dtmpb if lig3Dtmp.mindist(
        core3D) < lig3Dtmpb.mindist(core3D) else lig3Dtmp
    if relax:
        # Relax the ligand
        lig3Dtmp, enl = ffopt(args.ff, lig3Dtmp, [catoms[1]], 2, [
                              catoms[0]], False, MLoptbds[-2:-1], 200, debug=args.debug)
        lig3Dtmp.deleteatom(lig3Dtmp.natoms-1)
    lig3Dtmp, en_final = ffopt(
        args.ff, lig3Dtmp, [], 1, [], False, [], 0, debug=args.debug)
    if en_final - en_start > 20:
        print(('Warning: Complex may be strained. Ligand strain energy (kcal/mol) = ' +
              str(en_final - en_start)))
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3Dtmp)
    return lig3D_aligned


def align_dent1_lig(args, cpoint, core3D, coreref, ligand, lig3D, catoms,
                    rempi=False, ligpiatoms=[], MLb=[], ANN_flag=False,
                    ANN_bondl: float = np.nan, this_diag=0, MLbonds=dict(),
                    MLoptbds: Optional[List[float]] = None, i: int = 0,
                    EnableAutoLinearBend=True) -> Tuple[mol3D, List[float]]:
    """Aligns a monodentate ligand to core connecting atom coordinates.

    Parameters
    ----------
        args : Namespace
            Namespace of arguments.
        cpoint : atom3D
            atom3D class instance containing backbone connecting point.
        core3D : mol3D
            mol3D class instance of partially built complex.
        coreref : atom3D
            atom3D of core reference atom.
        ligand : str
            Name of ligand for dictionary lookup.
        lig3D : mol3D
            mol3D class instance of the ligand.
        catoms : list
            List of ligand connecting atom indices.
        rempi : bool, optional
            Flag for pi-coordinating ligand. Default is False.
        ligpiatoms : list, optional
            List of pi-coordinating atom indices in ligand. Default is empty.
        MLb : list, optional
            Custom M-L bond length (if any). Default is empty.
        ANN_flag : bool, optional
            Flag for ANN activation. Default is False.
        this_diag : run_diag, optional
            ANN run_diag class instance. Default is 0.
        MLbonds : dict, optional
            M-L bond dictionary. Default is empty.
        MLoptbds : list, optional
            List of final M-L bond lengths. Default is None.
        i : int, optional
            Ligand serial number. Default is 0.
        EnableAutoLinearBend : bool, optional
            Flag for enabling automatic bending of linear ligands (e.g. superoxo).

    Returns
    -------
        lig3D_aligned : mol3D
            mol3D class instance of aligned ligand.
        MLoptbds : list
            Updated list of metal ligand bonds.

    """
    if MLoptbds is None:
        MLoptbds = []

    corerefcoords = coreref.coords()
    # connection atom in lig3D
    atom0 = catoms[0]
    # translate ligand to overlap with backbone connecting point
    lig3D.alignmol(lig3D.getAtom(atom0), cpoint)
    # determine bond length (database/cov rad/ANN)
    bondl = get_MLdist(coreref, args.oxstate, args.spin, lig3D, atom0, ligand,
                       MLb, i, ANN_flag, ANN_bondl, this_diag, MLbonds, args.debug)
    MLoptbds.append(bondl)
    # align ligand to correct M-L distance
    u = vecdiff(cpoint.coords(), corerefcoords)
    lig3D = aligntoaxis2(lig3D, cpoint.coords(), corerefcoords, u, bondl)
    if rempi:  # pi-coordinating ligand
        if len(ligpiatoms) == 2:
            # align linear (non-arom.) pi-coordinating ligand
            lig3D = align_linear_pi_lig(corerefcoords, lig3D, atom0, ligpiatoms)
        else:  # 5 and 6 membered rings dealt with here
            lig3D = align_pi_ring_lig(corerefcoords, lig3D, atom0, ligpiatoms, u)
    elif lig3D.natoms > 1:
        # align ligand center of symmetry
        lig3D = align_lig_centersym(
            corerefcoords, lig3D, atom0, core3D, EnableAutoLinearBend)
        if lig3D.natoms > 2:
            # check for linear molecule and align
            lig3D = check_rotate_linear_lig(corerefcoords, lig3D, atom0)
            # check for symmetric molecule
            lig3D = check_rotate_symm_lig(corerefcoords, lig3D, atom0, core3D)
        # rotate around M-L axis to minimize steric repulsion
        lig3D = rotate_MLaxis_minimize_steric(
            corerefcoords, lig3D, atom0, core3D)
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned, MLoptbds


def align_dent2_lig(args, cpoint, batoms, m3D, core3D, coreref, ligand, lig3D,
                    catoms, MLb, ANN_flag, ANN_bondl: float, this_diag, MLbonds,
                    MLoptbds: List[float], frozenats: List[int], i: int
                    ) -> Tuple[mol3D, List[int], List[float]]:
    """Aligns a bidentate ligand to core connecting atom coordinates.

    Parameters
    ----------
        args : Namespace
            Namespace of arguments.
        cpoint : atom3D
            atom3D class instance containing backbone connecting point.
        batoms : list
            List of backbone atom indices.
        m3D : mol3D
            mol3D of backbone template.
        core3D : mol3D
            mol3D class instance of partially built complex.
        coreref : atom3D
            atom3D of core reference atom.
        ligand : str
            Name of ligand for dictionary lookup.
        lig3D : mol3D
            mol3D class instance of the ligand.
        catoms : list
            List of ligand connecting atom indices.
        MLb : list
            Custom M-L bond length (if any).
        ANN_flag : bool
            Flag for ANN activation.
        ANN_bondl : list
            List of ANN predicted bond lengths.
        this_diag : run_diag
            ANN run_diag class instance.
        MLbonds : dict
            M-L bond dictionary.
        MLoptbds : list
            List of final M-L bond lengths.
        frozenats : list
            List of atoms frozen in FF optimization.
        i : int, optional
            Ligand serial number. Default is 0.

    Returns
    -------
        lig3D_aligned : mol3D
            mol3D class instance of aligned ligand.
        frozenats : list
            List of frozen atoms.
        MLoptbds : list
            Updated list of metal ligand bonds.

    """
    corerefcoords = coreref.coords()
    r0 = corerefcoords
    # get cis conformer by rotating rotatable bonds
    # lig3D = find_rotate_rotatable_bond(lig3D,catoms)
    # connection atom
    atom0 = catoms[0]
    # translate ligand to match first connecting atom to backbone connecting point
    lig3D.alignmol(lig3D.getAtom(atom0), cpoint)
    r1 = lig3D.getAtom(atom0).coords()
    # Crude rotations to bring the 2nd connecting atom closer to its ideal location
    lig3D, r1b = align_dent2_catom2_coarse(
        args, lig3D, core3D, catoms, r1, r0, m3D, batoms, corerefcoords)
    # get bond length
    bondl = get_MLdist(coreref, args.oxstate, args.spin, lig3D, atom0, ligand,
                       MLb, i, ANN_flag, ANN_bondl, this_diag, MLbonds, args.debug)
    MLoptbds.append(bondl)
    MLoptbds.append(bondl)
    lig3D, dxyz = setPdistance(lig3D, r1, r0, bondl)
    # get target point for 2nd connecting atom
    rtarget = getPointu(corerefcoords, bondl, vecdiff(
        r1b, corerefcoords))  # get second point target
    if args.ff:
        # align 2nd connecting atom while balancing the desired location and ligand strain
        lig3D = align_dent2_catom2_refined(
            args, lig3D, catoms, bondl, r1, r0, core3D, rtarget, coreref, MLoptbds)
    else:
        print('Warning: Ligand FF optimization is inactive.')
    # rotate connecting atoms to align Hs properly
    lig3D = rotate_catoms_fix_Hs(lig3D, catoms, corerefcoords, core3D)
    # freeze local geometry
    lats = lig3D.getBondedAtoms(catoms[0])+lig3D.getBondedAtoms(catoms[1])
    for lat in list(set(lats)):
        frozenats.append(lat+core3D.natoms)
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned, frozenats, MLoptbds


def align_dent3_lig(args, cpoint, batoms, m3D, core3D, coreref, ligand, lig3D,
                    catoms, MLb, ANN_flag, ANN_bondl, this_diag, MLbonds,
                    MLoptbds: List[float], frozenats: List[int], i: int
                    ) -> Tuple[mol3D, List[int], List[float]]:
    """Aligns a tridentate ligand to core connecting atom coordinates

    Parameters
    ----------
        args : Namespace
            Namespace of arguments.
        cpoint : atom3D
            atom3D class instance containing backbone connecting point.
        batoms : list
            List of backbone atom indices.
        m3D : mol3D
            mol3D of backbone template.
        core3D : mol3D
            mol3D class instance of partially built complex.
        coreref : atom3D
            atom3D of core reference atom.
        ligand : str
            Name of ligand for dictionary lookup.
        lig3D : mol3D
            mol3D class instance of the ligand.
        catoms : list
            List of ligand connecting atom indices.
        MLb : list
            Custom M-L bond length (if any).
        ANN_flag : bool
            Flag for ANN activation.
        ANN_bondl : list
            List of ANN predicted bond lengths.
        this_diag : run_diag
            ANN run_diag class instance.
        MLbonds : dict
            M-L bond dictionary.
        MLoptbds : list
            List of final M-L bond lengths.
        frozenats : list
            List of atoms frozen in FF optimization.
        i : int, optional
            Ligand serial number. Default is 0.

    Returns
    -------
        lig3D_aligned : mol3D
            mol3D class instance of aligned ligand.
        frozenats : list
            List of frozen atoms.
        MLoptbds : list
            Updated list of metal ligand bonds.

    """
    atom0 = catoms[1]
    corerefcoords = coreref.coords()
    # align molecule according to connection atom and shadow atom
    lig3D.alignmol(lig3D.getAtom(atom0), m3D.getAtom(batoms[1]))
    # 1. align ligand connection atoms center of symmetry
    auxm = mol3D()
    auxm.addAtom(lig3D.getAtom(catoms[0]))
    auxm.addAtom(lig3D.getAtom(catoms[2]))
    r0 = core3D.getAtom(0).coords()
    lig3Db = mol3D()
    lig3Db.copymol3D(lig3D)
    theta, urot = rotation_params(
        r0, lig3D.getAtom(atom0).coords(), auxm.centersym())
    lig3D = rotate_around_axis(
        lig3D, lig3D.getAtom(atom0).coords(), urot, theta)
    # 2. align with correct plane
    rl0, rl1, rl2 = lig3D.getAtom(catoms[0]).coords(), lig3D.getAtom(
        catoms[1]).coords(), lig3D.getAtom(catoms[2]).coords()
    rc0, rc1, rc2 = m3D.getAtom(batoms[0]).coords(), m3D.getAtom(
        batoms[1]).coords(), m3D.getAtom(batoms[2]).coords()
    theta0, ul = rotation_params(rl0, rl1, rl2)
    theta1, uc = rotation_params(rc0, rc1, rc2)
    urot = vecdiff(rl1, corerefcoords)
    theta = vecangle(ul, uc)
    lig3Db = mol3D()
    lig3Db.copymol3D(lig3D)
    lig3D = rotate_around_axis(lig3D, rl1, urot, theta)
    lig3Db = rotate_around_axis(lig3Db, rl1, urot, 180-theta)
    rl0, rl1, rl2 = lig3D.getAtom(catoms[0]).coords(), lig3D.getAtom(
        catoms[1]).coords(), lig3D.getAtom(catoms[2]).coords()
    rl0b, rl1b, rl2b = lig3Db.getAtom(catoms[0]).coords(), lig3Db.getAtom(
        catoms[1]).coords(), lig3Db.getAtom(catoms[2]).coords()
    rc0, rc1, rc2 = m3D.getAtom(batoms[0]).coords(), m3D.getAtom(
        batoms[1]).coords(), m3D.getAtom(batoms[2]).coords()
    theta, ul = rotation_params(rl0, rl1, rl2)
    theta, ulb = rotation_params(rl0b, rl1b, rl2b)
    theta, uc = rotation_params(rc0, rc1, rc2)
    d1 = norm(np.cross(ul, uc))
    d2 = norm(np.cross(ulb, uc))
    lig3D = lig3D if (d1 < d2) else lig3Db  # pick best one
    # 3. correct if not symmetric
    theta0, urotaux = rotation_params(lig3D.getAtom(catoms[0]).coords(
    ), lig3D.getAtom(catoms[1]).coords(), core3D.getAtom(0).coords())
    theta1, urotaux = rotation_params(lig3D.getAtom(catoms[2]).coords(
    ), lig3D.getAtom(catoms[1]).coords(), core3D.getAtom(0).coords())
    dtheta = 0.5*(theta1-theta0)
    if abs(dtheta) > 0.5:
        lig3D = rotate_around_axis(
            lig3D, lig3D.getAtom(atom0).coords(), urot, dtheta)
    # 4. flip for correct stereochemistry
    urot = vecdiff(lig3D.getAtom(
        catoms[1]).coords(), core3D.getAtom(0).coords())
    lig3Db = mol3D()
    lig3Db.copymol3D(lig3D)
    lig3Db = rotate_around_axis(
        lig3Db, lig3Db.getAtom(catoms[1]).coords(), urot, 180)
    d1 = min(distance(lig3D.getAtom(catoms[2]).coords(), m3D.getAtom(batoms[2]).coords(
    )), distance(lig3D.getAtom(catoms[2]).coords(), m3D.getAtom(batoms[0]).coords()))
    d2 = min(distance(lig3Db.getAtom(catoms[2]).coords(), m3D.getAtom(batoms[2]).coords(
    )), distance(lig3Db.getAtom(catoms[2]).coords(), m3D.getAtom(batoms[0]).coords()))
    lig3D = lig3D if (d1 < d2) else lig3Db  # pick best one
    # 5. flip to align 1st and 3rd connection atoms
    lig3Db = mol3D()
    lig3Db.copymol3D(lig3D)
    theta, urot = rotation_params(lig3Db.getAtom(catoms[0]).coords(), lig3Db.getAtom(
        catoms[1]).coords(), lig3Db.getAtom(catoms[2]).coords())
    lig3Db = rotate_around_axis(
        lig3Db, lig3Db.getAtom(catoms[1]).coords(), urot, 180)
    d1 = min(distance(lig3D.getAtom(catoms[2]).coords(), m3D.getAtom(batoms[2]).coords(
    )), distance(lig3D.getAtom(catoms[2]).coords(), m3D.getAtom(batoms[0]).coords()))
    d2 = min(distance(lig3Db.getAtom(catoms[2]).coords(), m3D.getAtom(batoms[2]).coords(
    )), distance(lig3Db.getAtom(catoms[2]).coords(), m3D.getAtom(batoms[0]).coords()))
    lig3D = lig3D if d1 < d2 else lig3Db
    bondl = get_MLdist(m3D.getAtom(0), args.oxstate, args.spin, lig3D, atom0,
                       ligand, MLb, i, ANN_flag, ANN_bondl, this_diag, MLbonds,
                       args.debug)
    for iib in range(0, 3):
        MLoptbds.append(bondl)
    # set correct distance
    setPdistance(lig3D, lig3D.getAtom(atom0).coords(),
                 m3D.getAtom(0).coords(), bondl)
    # rotate connecting atoms to align Hs properly
    lig3D = rotate_catoms_fix_Hs(
        lig3D, [catoms[0], catoms[1], catoms[2]], m3D.getAtom(0).coords(), core3D)
    # freeze local geometry
    lats = lig3D.getBondedAtoms(catoms[0])+lig3D.getBondedAtoms(catoms[1])
    for lat in list(set(lats)):
        frozenats.append(lat+core3D.natoms)
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned, frozenats, MLoptbds


def mcomplex(args, ligs, ligoc, licores, globs):  # -> Tuple[mol3D, List[mol3D], str, run_diag, List, List]:
    """Main ligand placement routine

    Parameters
    ----------
        args : Namespace
            Namespace of arguments.
        ligs : list
            List of ligand names.
        ligoc : list
            List of ligand occupations.
        licores : dict
            Ligand dictionary as in molSimplify.
        globs : dict
            Globalvars dictionary. molSimplify Class.

    Returns
    -------
        core3D : mol3D
            mol3D class instance for core.
        complex3D : mol3D
            mol3D class instance for built complex.
        emsg : str
            Flag for error. String if error, with error message.
        this_diag: run_diag
            run_diag class instance of the complex.
        subcatoms_ext : list
            Substrate connection atoms from TSGen. Deprecated.
        mligcatoms_ext : list
            Ligand connection atoms from TSGen. Deprecated.

    """
    this_diag = run_diag()
    if globs.debug:
        print(('\nGenerating complex with ligands and occupations:', ligs, ligoc))
    if args.gui:
        args.gui.iWtxt.setText('\nGenerating complex with core:'+args.core +
                               ' and ligands: ' + ' '.join(ligs)+'\n'+args.gui.iWtxt.toPlainText())
        args.gui.app.processEvents()
        # import gui options
        from Classes.mWidgets import mQDialogWarn
    # initialize variables
    emsg = ''
    complex3D: List[mol3D] = []
    occs0 = []      # occurrences of each ligand
    toccs = 0       # total occurrence count (number of ligands)
    smilesligs = 0  # count how many smiles strings
    cats0 = []      # connection atoms for ligands
    dentl = []      # denticity of ligands
    connected = []  # indices in core3D of ligand atoms connected to metal
    frozenats = []  # atoms to be frozen in optimization
    freezeangles = False  # custom angles imposed
    MLoptbds = []   # list of bond lengths
    rempi = False   # remove dummy pi orbital center of mass atom
    backbatoms = []
    batslist = []
    bats = []
    ffoption_list = []  # for each ligand, keeps track of what the forcefield option is.
    # load bond data
    MLbonds = loaddata('/Data/ML.dat')
    # calculate occurrences, denticities etc for all ligands
    for i, ligname in enumerate(ligs):
        # if not in cores -> smiles/file
        if ligname not in list(licores.keys()):
            if args.smicat and len(args.smicat) >= (smilesligs+1):
                if 'pi' in args.smicat[smilesligs]:
                    cats0.append(['c'])
                else:
                    cats0.append(args.smicat[smilesligs])
            else:
                cats0.append([0])
            dent_i = len(cats0[-1])
            smilesligs += 1
        else:
            cats0.append(False)
        # otherwise get denticity from ligands dictionary
            if 'pi' in licores[ligname][2]:
                dent_i = 1
            else:
                if isinstance(licores[ligname][2], str):
                    dent_i = 1
                else:
                    dent_i = int(len(licores[ligname][2]))
        # get occurrence for each ligand if specified (default 1)
        oc_i = int(ligoc[i]) if i < len(ligoc) else 1
        occs0.append(0)         # initialize occurrences list
        dentl.append(dent_i)    # append denticity to list
        # loop over occurrence of ligand i to check for max coordination
        for j in range(0, oc_i):
            occs0[i] += 1
            toccs += dent_i

        if 'l' in args.ffoption:  # ligands.dict control for force field option
            if ligname in list(licores.keys()):  # ligand is in the ligands dictionary
                forcefield_option_current_ligand = licores[ligname][4][0].lower()
                ffoption_list.append(forcefield_option_current_ligand)
            else:  # default to 'ba' for ligands not in ligands.dict
                ffoption_list.append('ba')

    if 'l' in args.ffoption:  # ligands.dict control for force field option
        # Setting args.ffoption to the least-action option among the ligands.
        # So, if at least one of the ligands has 'n' as the forcefield option, no optimization.
        # Otherwise, if there is any mixture of 'b' and 'a' among the ligands, go with 'n' for consistency.
        # Otherwise, if there is a mixture of 'b' and 'ba', go with 'b'.
        # Otherwise, if there is a mixture of 'a' and 'ba', go with 'a'.
        # Only if all ligands have 'ba' as the forcefield option do we go with 'ba'.
        ffoption_set = set(ffoption_list)
        if 'n' in ffoption_set:
            args.ffoption = 'n'
        elif 'b' in ffoption_set and 'a' in ffoption_set:
            args.ffoption = 'n'
        elif 'b' in ffoption_set:
            args.ffoption = 'b'
        elif 'a' in ffoption_set:
            args.ffoption = 'a'
        else:
            args.ffoption = 'ba'

    # sort by descending denticity (needed for adjacent connection atoms)
    ligandsU, occsU, dentsU = ligs, occs0, dentl  # save unordered lists
    indcs = smartreorderligs(ligs, dentl, args.ligalign)
    ligands = [ligs[i] for i in indcs]  # sort ligands list
    occs = [occs0[i] for i in indcs]    # sort occurrences list
    tcats = [cats0[i] for i in indcs]   # sort connections list
    dents = [dentl[i] for i in indcs]   # sort denticities list
    # if using decorations, make repeatable list
    if args.decoration:
        if not args.decoration_index:
            print('Warning, no decoration index given, assuming first ligand')
            args.decoration_index = [[0]]
        if len(args.decoration_index) != len(ligs):
            new_decoration_index = []
            new_decorations = []
            for i in range(0, len(ligs)):
                if len(args.decoration_index) > i:
                    new_decoration_index.append(args.decoration_index[i])
                    new_decorations.append(args.decoration[i])
                else:
                    new_decoration_index.append([])
                    new_decorations.append(False)
            if args.debug:
                print('setting decoration:')
                print(new_decoration_index)
                print(new_decorations)
            args.decoration = new_decorations
            args.decoration_index = new_decoration_index
        args.decoration_index = [args.decoration_index[i]
                                 for i in indcs]   # sort decorations list
        args.decoration = [args.decoration[i]
                           for i in indcs]   # sort decorations list
    # sort keepHs list and unpack into list of tuples representing each connecting atom###
    keepHs = [k for k in args.keepHs]
    keepHs = [keepHs[i] for i in indcs]
    for i, keepH in enumerate(keepHs):
        keepHs[i] = [keepHs[i]] * dents[i]
    # sort M-L bond list
    MLb = False
    if args.MLbonds:
        MLb = [k for k in args.MLbonds]
        for j in range(len(args.MLbonds), len(ligs)):
            MLb.append(False)
        MLb = [MLb[i] for i in indcs]  # sort MLbonds list
    # sort ligands custom angles
    pangles = False
    if args.pangles:
        pangles = []
        for j in range(len(args.pangles), len(ligs)):
            pangles.append(False)
        pangles = [args.pangles[i] for i in indcs]  # sort custom langles list

    # compute number of connecting points required
    cpoints_required = 0
    for i, ligand in enumerate(ligands):
        for j in range(0, occs[i]):
            cpoints_required += dents[i]

    # load core and initialize template
    m3D, core3D, geom, backbatoms, coord, corerefatoms = init_template(
        args, cpoints_required, globs)
    # Get connection points for all the ligands
    # smart alignment and forced order

    # if geom:
    if args.ligloc and args.ligalign:
        batslist0 = []
        for i, ligand in enumerate(ligandsU):
            for j in range(0, occsU[i]):
                # get correct atoms
                bats, backbatoms = getnupdateb(backbatoms, dentsU[i])
                batslist0.append(bats)
        # reorder according to smart reorder
        for i in indcs:
            offset = 0
            for ii in range(0, i):
                offset += (occsU[ii]-1)
            for j in range(0, occsU[i]):
                batslist.append(batslist0[i+j+offset])  # sort connections list
    else:
        for i, ligand in enumerate(ligands):
            for j in range(0, occs[i]):
                # get correct atoms
                bats, backbatoms = getnupdateb(backbatoms, dents[i])
                batslist.append(bats)
    if not geom:
        for comb in batslist:
            for i in comb:
                if i == 1:
                    batslist[comb][i] = m3D.natoms - coord + 1
    # initialize ANN
    ANN_flag, ANN_bondl, ANN_reason, ANN_attributes, catalysis_flag = init_ANN(
        args, ligands, occs, dents, batslist, tcats, licores)

    this_diag.set_ANN(ANN_flag, ANN_reason, ANN_attributes, catalysis_flag)

    # freeze core
    for i in range(0, core3D.natoms):
        frozenats.append(i)

    # loop over ligands and begin functionalization
    # loop over ligands
    totlig = 0  # total number of ligands added
    ligsused = 0  # total number of ligands used
    subcatoms_ext = []
    mligcatoms_ext = []
    if args.mligcatoms:
        for i in range(len(args.mligcatoms)):
            mligcatoms_ext.append(0)
    for i, ligand in enumerate(ligands):
        if args.debug:
            print('************')
            print(('loading ligand '+str(ligand) + ', number  ' +
                   str(i) + ' of ' + str(len(ligands))))
        if not (ligand == 'x' or ligand == 'X'):

            # load ligand
            lig, emsg = lig_load(ligand)
            # add decorations to ligand
            if args.decoration and args.decoration_index:
                if len(args.decoration) > i and len(args.decoration_index) > i:
                    if args.decoration[i]:
                        if args.debug:
                            print(('decorating ' + str(ligand) + ' with ' + str(
                                args.decoration[i]) + ' at sites ' + str(args.decoration_index)))
                        lig = decorate_ligand(
                            ligand, args.decoration[i], args.decoration_index[i], args.debug)
                    else:
                        # keeps ligands that are not being decorated
                        lig, emsg = lig_load(ligand)
            lig.convert2mol3D()
            # initialize ligand
            lig3D, rempi, ligpiatoms = init_ligand(args, lig, tcats, keepHs, i)
            if emsg:
                return core3D, complex3D, emsg, this_diag, subcatoms_ext, mligcatoms_ext
        # Skip = False
        for j in range(0, occs[i]):  # The number of occurrences of the current ligand.
            if args.debug:
                print(('loading copy '+str(j) + ' of ligand ' +
                       ligand + ' with dent ' + str(dents[i])))
                print(('totlig is ' + str(totlig)))
                print(('ligused is ' + str(ligsused)))
                print(('target BL is ' + str(ANN_bondl[ligsused])))
                print('******')
            denticity = dents[i]

            if not (ligand == 'x' or ligand == 'X') and (totlig-1+denticity < coord):
                # add atoms to connected atoms list
                catoms = lig.cat  # connection atoms
                initatoms = core3D.natoms  # initial number of atoms in core3D
                if args.debug:
                    print((ligand.lower()))
                    print((args.mlig))
                    print((args.core.lower()))
                    print(('mligcatoms_ext is ' + str(mligcatoms_ext)))
                for at in catoms:
                    connected.append(initatoms+at)
                # initialize variables
                # metal coordinates in backbone
                mcoords = core3D.getAtom(0).coords()
                atom0 = 0  # initialize variables
                coreref = corerefatoms.getAtom(totlig)
                # connecting point in backbone to align ligand to
                batoms = get_batoms(args, batslist, ligsused)
                cpoint = m3D.getAtom(batoms[0])
                # attach ligand depending on the denticity
                # optimize geometry by minimizing steric effects
                if args.debug:
                    print(('backbone atoms: ' + str(batoms)))
                if (denticity == 1):
                    lig3D, MLoptbds = align_dent1_lig(
                        args, cpoint, core3D, coreref, ligand, lig3D, catoms,
                        rempi, ligpiatoms, MLb, ANN_flag, ANN_bondl[ligsused],
                        this_diag, MLbonds, MLoptbds, i)
                    if args.debug:
                        print(('adding monodentate at distance: ' + str(
                            ANN_bondl[totlig]) + '/'+str(MLb) + '/'+' at catoms ' + str(catoms)))
                        print('printing ligand information')
                        print((lig3D.printxyz()))
                elif (denticity == 2):
                    lig3D, frozenats, MLoptbds = align_dent2_lig(
                        args, cpoint, batoms, m3D, core3D, coreref, ligand,
                        lig3D, catoms, MLb, ANN_flag, ANN_bondl[ligsused],
                        this_diag, MLbonds, MLoptbds, frozenats, i)
                elif (denticity == 3):
                    lig3D, frozenats, MLoptbds = align_dent3_lig(
                        args, cpoint, batoms, m3D, core3D, coreref, ligand,
                        lig3D, catoms, MLb, ANN_flag, ANN_bondl[ligsused],
                        this_diag, MLbonds, MLoptbds, frozenats, i)
                elif (denticity == 4):
                    # note: catoms for ligand should be specified clockwise
                    # connection atoms in backbone
                    if args.antigeoisomer:
                        print('anti geometric isomer requested.')
                        catoms = catoms[::-1]
                    batoms = batslist[ligsused]
                    if len(batoms) < 1:
                        if args.gui:
                            emsg = 'Connecting all ligands is not possible. Check your input!'
                            qqb = mQDialogWarn('Warning', emsg)
                            qqb.setParent(args.gui.wmain)
                        break
                    # connection atom
                    atom0 = catoms[0]
                    # align ligand center of symmetry to core reference atom
                    auxmol_lig = mol3D()
                    auxmol_m3D = mol3D()
                    for iiax in range(0, 4):
                        auxmol_lig.addAtom(lig3D.getAtom(catoms[iiax]))
                        auxmol_m3D.addAtom(m3D.getAtom(batoms[iiax]))
                    lig3D.alignmol(
                        atom3D('C', auxmol_lig.centersym()), m3D.getAtom(0))
                    # necessary to prevent lig3D from being overwritten
                    lig3Dtmp = mol3D()
                    lig3Dtmp.copymol3D(lig3D)
                    # compute average metal-ligand distance
                    auxmol_lig = mol3D()
                    auxmol_m3D = mol3D()
                    sum_MLdists = 0
                    for iiax in range(0, 4):
                        auxmol_lig.addAtom(lig3Dtmp.getAtom(catoms[iiax]))
                        auxmol_m3D.addAtom(m3D.getAtom(batoms[iiax]))
                        sum_MLdists += distance(m3D.getAtomCoords(0),
                                                auxmol_lig.getAtomCoords(iiax))
                    avg_MLdists = sum_MLdists/4
                    # scale template by average M-L distance
                    auxmol_m3D.addAtom(m3D.getAtom(0))
                    # TODO BCM definition slightly modified. Keep an eye for unexpected structures
                    for iiax in range(0, 4):
                        auxmol_m3D.BCM(iiax, 4, avg_MLdists)
                    auxmol_m3D.deleteatom(4)
                    # align lig3D to minimize RMSD from template
                    auxmol_lig, U, d0, d1 = kabsch(auxmol_lig, auxmol_m3D)
                    lig3D.translate(d0)
                    lig3D = rotate_mat(lig3D, U)

                    bondl = get_MLdist(m3D.getAtom(0), args.oxstate, args.spin,
                                       lig3D, atom0, ligand, MLb, i, ANN_flag,
                                       ANN_bondl[ligsused], this_diag, MLbonds,
                                       args.debug)
                    for iib in range(0, 4):
                        MLoptbds.append(bondl)
                elif (denticity == 5):
                    # connection atoms in backbone
                    batoms = batslist[ligsused]
                    if len(batoms) < 1:
                        if args.gui:
                            qqb = mQDialogWarn('Warning', emsg)
                            qqb.setParent(args.gui.wmain)
                        emsg = 'Connecting all ligands is not possible. Check your input!'
                        break
                    # get center of mass
                    ligc = mol3D()
                    for c_i in range(0, 4):  # 5 is the non-planar atom
                        ligc.addAtom(lig3D.getAtom(catoms[c_i]))
                    # translate ligand to the middle of octahedral
                    lig3D.translate(vecdiff(mcoords, ligc.centersym()))
                    # get plane
                    r0c = m3D.getAtom(batoms[0]).coords()
                    r2c = m3D.getAtom(batoms[1]).coords()
                    r1c = mcoords
                    r0l = lig3D.getAtom(catoms[0]).coords()
                    r2l = lig3D.getAtom(catoms[1]).coords()
                    r1l = mcoords
                    # normal vector to backbone plane
                    theta, uc = rotation_params(r0c, r1c, r2c)
                    # normal vector to ligand plane
                    theta, ul = rotation_params(r0l, r1l, r2l)
                    theta = vecangle(uc, ul)
                    u = np.cross(uc, ul)
                    lig3Db = mol3D()
                    lig3Db.copymol3D(lig3D)
                    # rotate around axis to match planes
                    lig3D = rotate_around_axis(lig3D, mcoords, u, theta)
                    lig3Db = rotate_around_axis(lig3Db, mcoords, u, 180+theta)
                    d1 = distance(lig3D.getAtom(
                        catoms[4]).coords(), m3D.getAtom(batoms[-1]).coords())
                    d2 = distance(lig3Db.getAtom(
                        catoms[4]).coords(), m3D.getAtom(batoms[-1]).coords())
                    lig3D = lig3D if (d2 < d1) else lig3Db  # pick best one
                    # rotate around center axis to match backbone atoms
                    r0l = vecdiff(lig3D.getAtom(catoms[0]).coords(), mcoords)
                    r1l = vecdiff(m3D.getAtom(totlig+1).coords(), mcoords)
                    u = np.cross(r0l, r1l)
                    theta = 180*np.arccos(np.dot(r0l, r1l)/(norm(r0l)*norm(r1l)))/np.pi
                    lig3Db = mol3D()
                    lig3Db.copymol3D(lig3D)
                    lig3D = rotate_around_axis(lig3D, mcoords, u, theta)
                    lig3Db = rotate_around_axis(lig3Db, mcoords, u, theta-90)
                    d1 = distance(lig3D.getAtom(
                        catoms[0]).coords(), m3D.getAtom(batoms[0]).coords())
                    d2 = distance(lig3Db.getAtom(
                        catoms[0]).coords(), m3D.getAtom(batoms[0]).coords())
                    lig3D = lig3D if (d1 < d2) else lig3Db  # pick best one
                    bondl, exact_match = get_MLdist_database(
                        core3D.getAtom(0), args.oxstate, args.spin, lig3D,
                        catoms[0], ligand, MLbonds, args.debug)
                    # flip if necessary
                    if len(batslist) > ligsused:
                        nextatbats = batslist[ligsused]
                    auxm = mol3D()
                    if len(nextatbats) > 0:
                        for at in nextatbats:
                            auxm.addAtom(m3D.getAtom(at))
                        if lig3D.overlapcheck(auxm, True):  # if overlap flip
                            urot = vecdiff(m3D.getAtomCoords(
                                batoms[1]), m3D.getAtomCoords(batoms[0]))
                            lig3D = rotate_around_axis(
                                lig3D, mcoords, urot, 180)
                    for iib in range(0, 5):
                        MLoptbds.append(bondl)
                elif (denticity == 6):
                    # connection atoms in backbone
                    batoms = batslist[ligsused]
                    if len(batoms) < 1:
                        if args.gui:
                            qqb = mQDialogWarn('Warning', emsg)
                            qqb.setParent(args.gui.wmain)
                        emsg = 'Connecting all ligands is not possible. Check your input!'
                        break
                    # get center of mass
                    ligc = mol3D()
                    for c_i in range(0, 6):
                        ligc.addAtom(lig3D.getAtom(catoms[c_i]))
                    # translate metal to the middle of octahedral
                    core3D.translate(vecdiff(ligc.centersym(), mcoords))
                    bondl, exact_match = get_MLdist_database(
                        core3D.getAtom(0), args.oxstate, args.spin, lig3D,
                        catoms[0], ligand, MLbonds, args.debug)
                    for iib in range(0, 6):
                        MLoptbds.append(bondl)
                auxm = mol3D()
                auxm.copymol3D(lig3D)
                complex3D.append(auxm)
                if 'a' not in lig.ffopt.lower():
                    for latdix in range(0, lig3D.natoms):
                        if args.debug:
                            print(('a is not ff.lower, so adding atom:  ' +
                                   str(latdix+core3D.natoms) + ' to freeze'))
                        frozenats.append(latdix+core3D.natoms)
                # combine molecules
                core3D = core3D.combine(lig3D)
                core3D.convert2OBMol()
                core3D.convert2mol3D()
                # remove dummy cm atom if requested
                if rempi:
                    # remove the fictitious center atom, for aromatic-bonding ligands like benzene
                    core3D.deleteatom(core3D.natoms-1)
                if args.debug:
                    print(('number of atoms in lig3D is ' + str(lig3D.natoms)))
                if lig3D.natoms < 3:
                    frozenats += list(range(core3D.natoms-2, core3D.natoms))
                    if args.debug:
                        print(
                            (str(list(range(core3D.natoms-2, core3D.natoms))) + ' are frozen.'))
                if args.calccharge:
                    core3D.charge += lig3D.charge
                    if args.debug:
                        print(('core3D charge is ' + str(core3D.charge)))
                # perform FF optimization if requested
                if args.debug:
                    print(('saving a copy of the complex named complex_' +
                           str(i)+'_'+str(j) + '.xyz'))
                    core3D.writexyz('complex_'+str(i)+'_'+str(j) + '.xyz')

                if 'a' in args.ffoption:
                    if args.debug:
                        print('FF optimizing molecule after placing ligand')
                        print(
                            ('in the relax function, passing connected atoms list: ' + str(connected)))
                    # (ff,mol,connected,constopt,frozenats,frozenangles,mlbonds,nsteps,debug=False):
                    core3D, enc = ffopt(ff=args.ff,
                                        mol=core3D,
                                        connected=connected,
                                        constopt=1,
                                        frozenats=frozenats,
                                        frozenangles=freezeangles,
                                        mlbonds=MLoptbds,
                                        nsteps='Adaptive',
                                        spin=int(args.spin),
                                        debug=args.debug)
                    if args.debug:
                        print(
                            ('saving a copy of the complex named complex_'+str(i)+'_'+str(j) + '_ff.xyz'))
                        core3D.writexyz('complex_'+str(i) +
                                        '_'+str(j) + '_ff.xyz')
                if args.debug:
                    print(('done with pair of inds '+str(i)+' and '+str(j)))
                    print('**************************')
            totlig += denticity
            ligsused += 1
    # perform FF optimization if requested
    if 'a' in args.ffoption or args.ff_final_opt:
        if args.debug:
            print('Performing final FF opt')
        # idxes
        midx = core3D.findMetal()[0]
        # If args.ff_final_opt is None (default) use args.ff
        ff = args.ff_final_opt or args.ff
        core3D, enc = ffopt(ff=ff,
                            mol=core3D,
                            connected=connected,
                            constopt=1,
                            frozenats=connected + [midx],
                            frozenangles=freezeangles,
                            mlbonds=MLoptbds,
                            nsteps='Adaptive',
                            spin=int(args.spin),
                            debug=args.debug)

        if args.debug:
            print(
                'saving a final debug copy of the complex named complex_after_final_ff.xyz')
            core3D.writexyz('complex_after_final_ff.xyz')
        # core3D,enc = ffopt(args.ff,core3D,connected,1,frozenats,freezeangles,MLoptbds,'Adaptive',args.debug)

    return core3D, complex3D, emsg, this_diag, subcatoms_ext, mligcatoms_ext


def generate_report(args, ligands, ligoc, licores, globs):
    ligs = []
    cons = []
    # Bunch of unused variables?? RM 2022/02/17
    # mono_inds = []
    # bi_inds = []
    # tri_inds = []
    # tetra_inds = []
    # penta_inds = []
    emsg, complex3D = False, []
    occs0 = []      # occurrences of each ligand
    toccs = 0       # total occurrence count (number of ligands)
    # catsmi = []     # SMILES ligands connection atoms
    smilesligs = 0  # count how many smiles strings
    cats0 = []      # connection atoms for ligands
    dentl = []      # denticity of ligands
    # connected = []  # indices in core3D of ligand atoms connected to metal
    # frozenats = []  # atoms to be frozen in optimization
    # freezeangles = False  # custom angles imposed
    # MLoptbds = []   # list of bond lengths
    # rempi = False   # remove dummy pi orbital center of mass atom
    backbatoms = []
    batslist = []
    bats = []
    print('ONLY report wanted. Not building structure.')
    metal_mol = mol3D()
    metal_mol.addAtom(atom3D(args.core))
    # CURRENTLY only works for molsimplify ligands...
    for i, name in enumerate(ligands):
        this_mol, emsg = lig_load(name)
        this_mol.convert2mol3D()
        this_lig = ligand_class(mol3D(), [], this_mol.denticity)
        this_lig.mol = this_mol
        this_con = this_mol.cat
        ligs.append(this_lig)
        cons.append(this_con)
        if name not in list(licores.keys()):  # ligand is not in the ligands dictionary
            if args.smicat and len(args.smicat) >= (smilesligs+1):
                if 'pi' in args.smicat[smilesligs]:
                    cats0.append(['c'])
                else:
                    cats0.append(args.smicat[smilesligs])
            else:
                cats0.append([0])
            dent_i = len(cats0[-1])
            smilesligs += 1
        else:
            cats0.append(False)
        # otherwise get denticity from ligands dictionary
            if 'pi' in licores[name][2]:
                dent_i = 1
            else:
                if isinstance(licores[name][2], str):
                    dent_i = 1
                else:
                    dent_i = int(len(licores[name][2]))
        oc_i = int(ligoc[i]) if i < len(ligoc) else 1
        occs0.append(0)         # initialize occurrences list
        dentl.append(dent_i)    # append denticity to list
        # loop over occurrence of ligand i to check for max coordination
        for j in range(0, oc_i):
            occs0[i] += 1
            toccs += dent_i
    # sort by descending denticity (needed for adjacent connection atoms)
    ligandsU, occsU, dentsU = ligs, occs0, dentl  # save unordered lists
    indcs = smartreorderligs(ligs, dentl, args.ligalign)
    occs = [occs0[i] for i in indcs]    # sort occurrences list
    tcats = [cats0[i] for i in indcs]   # sort connections list
    dents = [dentl[i] for i in indcs]   # sort denticities list

    cpoints_required = 0
    for i, ligand_val in enumerate(ligands):
        for j in range(0, occs[i]):
            cpoints_required += dents[i]

    # load core and initialize template
    m3D, core3D, geom, backbatoms, coord, corerefatoms = init_template(
        args, cpoints_required, globs)
    # Get connection points for all the ligands
    # smart alignment and forced order

    # if geom:
    if args.ligloc and args.ligalign:
        batslist0 = []
        for i, ligand_val in enumerate(ligandsU):
            for j in range(0, occsU[i]):
                # get correct atoms
                bats, backbatoms = getnupdateb(backbatoms, dentsU[i])
                batslist0.append(bats)
        # reorder according to smart reorder
        for i in indcs:
            offset = 0
            for ii in range(0, i):
                offset += (occsU[ii]-1)
            for j in range(0, occsU[i]):
                # sort connections list
                batslist.append(batslist0[i+j+offset])
    else:
        for i, ligand_val in enumerate(ligands):
            for j in range(0, occs[i]):
                # get correct atoms
                bats, backbatoms = getnupdateb(backbatoms, dents[i])
                batslist.append(bats)
    if not geom:
        for comb in batslist:
            for i in comb:
                if i == 1:
                    batslist[comb][i] = m3D.natoms - coord + 1
    ANN_flag, ANN_bondl, ANN_reason, ANN_attributes, catalysis_flag = init_ANN(
        args, ligands, occs, dents, batslist, tcats, licores)

    this_diag = run_diag()
    this_diag.set_ANN(ANN_flag, ANN_reason, ANN_attributes, catalysis_flag)

    # Unused return arguments
    subcatoms_ext = []
    mligcatoms_ext = []
    if args.mligcatoms:
        for i in range(len(args.mligcatoms)):
            mligcatoms_ext.append(0)

    return core3D, complex3D, emsg, this_diag, subcatoms_ext, mligcatoms_ext


def structgen(args, rootdir, ligands, ligoc, globs, sernum, write_files=True) -> Tuple[List[str], str, run_diag]:
    """Main structure generation routine - multiple structures

    Parameters
    ----------
        args : Namespace
            Namespace of arguments.
        rootdir : str
            Directory of current run to generate complex.
        ligands : list
            List of ligand names.
        ligoc : list
            List of ligand occupations.
        globs : dict
            Globalvars dictionary. molSimplify Class.
        sernum : str
            Serial number of complex for naming.
        write_files : bool, optional
            Flag to write files. Default is True. False for pythonic generation.


    Returns
    -------
        strfiles : str
            List of XYZ files.
        emsg : str
            Error message for structure generation. If True, has string.
        this_diag : run_diag
            run_diag class instance containing properties of structure.

    """
    emsg = ''

    strfiles: List[str] = []
    # load ligand dictionary
    licores = getlicores()
    # build structure
    sanity = False
    this_diag = run_diag()
    if (ligands):
        if args.reportonly:
            core3D, complex3D, emsg, this_diag, subcatoms_ext, mligcatoms_ext = generate_report(
                args, ligands, ligoc, licores, globs)
            if emsg:
                return strfiles, emsg, this_diag
        else:
            core3D, complex3D, emsg, this_diag, subcatoms_ext, mligcatoms_ext = mcomplex(
                args, ligands, ligoc, licores, globs)
            if args.debug:
                print(('subcatoms_ext are ' + str(subcatoms_ext)))
            if emsg:
                return strfiles, emsg, this_diag
    else:
        print('You specified no ligands. The whole mcomplex is read from the core.')
        # read mol3D from core
        core3D = mol3D()
        if '.xyz' in args.core:
            core3D.readfromxyz(args.core)
        else:
            atom = atom3D(Sym=args.core, xyz=[0, 0, 0])
            core3D.addAtom(atom)

    name_core = args.core

    if args.calccharge:
        args.charge = core3D.charge
        if args.debug:
            print(('setting charge to be ' + str(args.charge)))
    # check for molecule sanity
    if args.reportonly:
        this_diag.set_sanity(False, 'graph')
    else:
        sanity, d0 = core3D.sanitycheck(True)
        if sanity:
            print(('WARNING: Generated complex is not good! Minimum distance between atoms:' +
                  "{0:.2f}".format(d0)+'A\n'))
            if args.gui:
                ssmsg = 'Generated complex in folder '+rootdir + \
                    ' is no good! Minimum distance between atoms:' + \
                        "{0:.2f}".format(d0)+'A\n'
                from Classes.mWidgets import mQDialogWarn
                qqb = mQDialogWarn('Warning', ssmsg)
                qqb.setParent(args.gui.wmain)
        if args.debug:
            print(('setting sanity diag, min dist at ' +
                   str(d0) + ' (higher is better)'))
        this_diag.set_sanity(sanity, d0)
    # generate file name
    fname = name_complex(rootdir, name_core, args.geometry,
                         ligands, ligoc, sernum, args, 1, sanity)
    if args.debug:
        print(('fname is ' + fname))

    # write xyz file
    if (not args.reportonly) and (write_files):
        core3D.writexyz(fname)
    strfiles.append(fname)
    if write_files:
        # write report file
        this_diag.set_mol(core3D)
        this_diag.write_report(fname+'.report')
        # write input file from command line arguments
        getinputargs(args, fname)
    # (Possibly breaking change) Copy core3D into this_diag
    core3D_copy = mol3D()
    core3D_copy.copymol3D(core3D)
    this_diag.set_mol(core3D_copy)

    del core3D  # Legacy code, unsure if needed

    pfold = rootdir.split('/', 1)[-1]
    if args.gui:
        args.gui.iWtxt.setText('In folder '+pfold+' generated ' +
                               '1 structure!\n'+args.gui.iWtxt.toPlainText())
        args.gui.app.processEvents()
    print(('\nIn folder '+rootdir+' generated 1 structure!'))

    return strfiles, emsg, this_diag
