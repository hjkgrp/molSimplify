# @file tsgen.py
#  Generates transition state guess structures based on a user-specified complex (core), substrate and automatically identified reaction class.
#
#  Written by Terry Gani for HJK Group
#
#  Dpt of Chemical Engineering, MIT

from numpy import log

from molSimplify.Classes.mol3D import (mol3D)
from molSimplify.Classes.atom3D import atom3D
from molSimplify.Classes.rundiag import (run_diag)
from molSimplify.Scripts.geometry import (alignPtoaxis,
                                          distance,
                                          rotate_around_axis,
                                          rotation_params,
                                          vecangle,
                                          vecdiff)
from molSimplify.Scripts.io import getinputargs
from molSimplify.Scripts.structgen import (PointTranslateSph,
                                           align_lig_centersym,
                                           check_rotate_linear_lig,
                                           check_rotate_symm_lig,
                                           ffopt,
                                           getconnection,
                                           rotate_MLaxis_minimize_steric)

# XY sum cov rad coefficient
XYcoeff = 1.1
# Mode 1 (oxidative addition) MXY angle
MXYang = 135
# Mode 1 (oxidative addition) MX sum cov rad coefficient
MXdistcoeff = 0.9
# Mode 3 (abstraction) ABX angle
ABXang = 140

# Gets all possible substrate connecting points (X in A-B...X-Y).
#
#  Given a fixed ABX angle and AB distance, these lie on a circle.
#
#  Due to limitations of geometry routines, we sample the sphere and keep points within ~1 deg of the circle.
#  @param core mol3D of core
#  @param catom Connecting atom to core
#  @param Midx Core reference metal index
#  @param BL Target B-X distance
#  @param ABXang Target A-B-X angle
#  @return List of discretized possible connecting points


def getconnections(core, catom, Midx, BL, ABXang):
    Ocoords = core.getAtom(catom).coords()
    Mcoords = core.getAtom(Midx).coords()
    backbcoords = alignPtoaxis(Ocoords, Ocoords, vecdiff(Ocoords, Mcoords), BL)
    am = atom3D('C', backbcoords)
    connPts = []
    for iphi in range(1, 359, 10):
        for itheta in range(1, 179, 1):
            P = PointTranslateSph(Ocoords, backbcoords, [BL, iphi, itheta])
            am.setcoords(P)
            ang = 180-vecangle(vecdiff(Ocoords, Mcoords), vecdiff(P, Ocoords))
            if abs(ang - ABXang) < 1:
                connPts.append(P)
    return connPts

# Cheap method for determining optimal connecting point by empirically maximizing distances.
#  @param core mol3D of core
#  @param connPts List of possible connecting points
#  @param catom Connecting atom to core
#  @return Optimal substrate connecting point


def substplacecheap(core, connPts, catom):
    corerem = mol3D()
    corerem.copymol3D(core)
    corerem.deleteatom(catom)  # complex less O atom
    mdist = -1
    cpoint = [0, 0, 0]
    for P in connPts:
        if corerem.mindisttopoint(P) < 1:  # auto-reject if too close
            d0 = -2
        else:
            d0 = distance(core.centermass(), P)+0.5 * \
                log(corerem.mindisttopoint(P)-1)
        if d0 > mdist:
            mdist = d0
            cpoint = P
    return cpoint

# Aligns substrate to connecting point while minimizing sterics (abstraction version)
#  @param core mol3D of core
#  @param substr mol3D of substrate
#  @param substreact Index of substrate reacting atom
#  @param compreact Index of core reacting atom
#  @param cpoint Coordinates of connecting point
#  @param args Namespace of arguments
#  @param connected List of connecting atoms for FF optimization
#  @param frozenats Indices of atoms frozen in FF optimization
#  @return Name of files generated, error messages, run diagnostic info


def substplaceff_mode3(core, substr, substreact, compreact, cpoint, args, connected, frozenats):
    enc = 0
    # align substrate according to connection atom and shadow atom
    substr.alignmol(substr.getAtom(substreact), atom3D('H', cpoint))
    # perform rotations
    Bcoords = core.getAtomCoords(compreact)
    if substr.natoms > 1:
        # align ligand center of symmetry
        substr = align_lig_centersym(Bcoords, substr, substreact, core, False)
    if substr.natoms > 2:
        # check for linear molecule and align
        substr = check_rotate_linear_lig(Bcoords, substr, substreact)
        # check for symmetric molecule
        substr = check_rotate_symm_lig(Bcoords, substr, substreact, core)
        # rotate around M-L axis to minimize steric repulsion
        substr = rotate_MLaxis_minimize_steric(
            Bcoords, substr, substreact, core)
    # distort substrate molecule
    adjsidx = substr.getBondedAtoms(substreact)[0]
    XYBL = XYcoeff*(substr.getAtom(substreact).rad +
                    substr.getAtom(adjsidx).rad)
    substr.BCM(adjsidx, substreact, XYBL)
    # combine molecules
    ts3D = mol3D()
    ts3D.copymol3D(core)
    ts3D = ts3D.combine(substr)
    ts3D.charge += substr.charge
    if 'a' in args.ffoption or args.substplaceff:
        ts3D, enc = ffopt(args.ff, ts3D, connected, 1,
                          frozenats, False, [], 'Adaptive')
    return ts3D, enc

# Aligns substrate to connecting point while minimizing sterics (oxidative addition version)
#  @param core mol3D of core
#  @param substr mol3D of substrate
#  @param substreact Index of substrate reacting atom
#  @param compreact Index of core reacting atom
#  @param cpoint Coordinates of distal ligand atom connecting point
#  @param ligalignpt Coordinates of proximal ligand atom connecting point
#  @param args Namespace of arguments
#  @param connected List of connecting atoms for FF optimization
#  @param frozenats Indices of atoms frozen in FF optimization
#  @return Name of files generated, error messages, run diagnostic info


def substplaceff_mode1(core, substr, substreact, compreact, cpoint, ligalignpt, args, connected, frozenats):
    enc = 0
    Mcoords = core.getAtomCoords(compreact)
    adjsidx = substr.getBondedAtoms(substreact)[0]
    theta, u = rotation_params(
        substr.getAtomCoords(adjsidx), cpoint, ligalignpt)
    substr = rotate_around_axis(substr, cpoint, u, 180-theta)
    if substr.natoms > 2:
        # same as aligning an equilibrium ligand
        substr = check_rotate_linear_lig(Mcoords, substr, substreact)
        substr = check_rotate_symm_lig(Mcoords, substr, substreact, core)
        substr = rotate_MLaxis_minimize_steric(
            Mcoords, substr, substreact, core)
    ts3D = mol3D()
    ts3D.copymol3D(core)
    ts3D = ts3D.combine(substr)
    if 'a' in args.ffoption or args.substplaceff:
        print('FF optimizing remainder of substrate')
        ts3D, enc = ffopt(args.ff, ts3D, connected, 1,
                          frozenats, False, [], 'Adaptive')
    return ts3D, enc

# Main transition state generation routine
#  @param mode TS generation mode (rungen.py)
#  @param args Namespace of arguments
#  @param rootdir Root directory
#  @param core mol3D of core
#  @param substr mol3D of substrate
#  @param compreact Index of core reacting atom
#  @param substreact Index of substrate reacting atom
#  @param globs Global variables
#  @return Name of files generated, error messages, run diagnostic info


def tsgen(mode, args, rootdir, core, substr, compreact, substreact, globs):
    emsg = False
    ts3D = mol3D()
    this_diag = run_diag()
    strfiles = []
    adjsidx = substr.getBondedAtoms(substreact)[0]
    adjcidx = core.getBondedAtoms(compreact)[0]
    # initialize connecting and frozen atoms for FF opt
    frozenats = []
    for i in range(0, core.natoms):
        frozenats.append(i)
    # also freeze the abstracted atom and the heavy atom bonded to it
    frozenats.append(core.natoms+substreact)
    frozenats.append(core.natoms+adjsidx)
    connected = [core.natoms+substreact]
    # START FUNCTIONALIZING
    sanity = False
    if mode == 2:
        emsg = 'Sorry, this mode is not supported yet. Exiting...'
        return strfiles, emsg, this_diag
    elif mode == 1:  # oxidative addition of a single group
        # get first connecting point
        MXBL = MXdistcoeff*(core.getAtom(compreact).rad +
                            substr.getAtom(substreact).rad)
        cpoint = getconnection(core, compreact, MXBL)
        # distort substrate molecule
        XYBL = XYcoeff*(substr.getAtom(substreact).rad +
                        substr.getAtom(adjsidx).rad)
        substr.BCM(adjsidx, substreact, XYBL)
        # align substrate molecule
        substr.alignmol(substr.getAtom(substreact), atom3D('H', cpoint))
        tmp3D = mol3D()
        tmp3D.copymol3D(core)
        tmp3D.addAtom(atom3D('Cl', cpoint))
        ligalignpts = getconnections(
            tmp3D, tmp3D.natoms-1, compreact, XYBL, MXYang)
        if args.substplaceff:
            # full FF substrate placement
            print('Full FF-based substrate placement specified.')
            en_min = 1e6
            for n, P in enumerate(ligalignpts):
                print(('Evaluating FF energy of point ' +
                       str(n+1)+' of '+str(len(ligalignpts))))
                coretmp = mol3D()
                coretmp.copymol3D(core)
                substrtmp = mol3D()
                substrtmp.copymol3D(substr)
                ts3Dtmp, enc = substplaceff_mode1(
                    coretmp, substrtmp, substreact, compreact, cpoint, P, args, connected, frozenats)
                if enc < en_min:
                    en_min = enc
                    ts3D = mol3D()
                    ts3D.copymol3D(ts3Dtmp)
        else:
            # cheap substrate placement
            print('Cheap substrate placement')
            ligalignpt = substplacecheap(core, ligalignpts, compreact)
            ts3D, enc = substplaceff_mode1(
                core, substr, substreact, compreact, cpoint, ligalignpt, args, connected, frozenats)
    elif mode == 3:  # abstraction
        # distort A-B bond
        ABBL = distance(core.getAtomCoords(compreact), core.getAtomCoords(
            adjcidx)) + 0.05*(core.getAtom(compreact).rad + core.getAtom(adjcidx).rad)
        core.BCM(compreact, adjcidx, ABBL)
        # set B-X distance
        BXBL = 1.1*(substr.getAtom(substreact).rad +
                    core.getAtom(compreact).rad)
        # get possible connecting points
        connPts = getconnections(core, compreact, adjcidx, BXBL, ABXang)
        if args.substplaceff:
            # full FF substrate placement
            print('Full FF-based substrate placement specified.')
            en_min = 1e6
            for n, P in enumerate(connPts):
                print(('Evaluating FF energy of point ' +
                       str(n+1)+' of '+str(len(connPts))))
                coretmp = mol3D()
                coretmp.copymol3D(core)
                substrtmp = mol3D()
                substrtmp.copymol3D(substr)
                ts3Dtmp, enc = substplaceff_mode3(
                    coretmp, substrtmp, substreact, compreact, P, args, connected, frozenats)
                if enc < en_min:
                    en_min = enc
                    ts3D = mol3D()
                    ts3D.copymol3D(ts3Dtmp)
        else:
            # cheap substrate placement
            print('Cheap substrate placement')
            cpoint = substplacecheap(core, connPts, compreact)
            ts3D, enc = substplaceff_mode3(
                core, substr, substreact, compreact, cpoint, args, connected, frozenats)
            if 'a' in args.ffoption:
                print('FF optimized remainder of substrate')
    ts3D.charge += substr.charge
    # END FUNCTIONALIZING
    fname = name_TS(rootdir, args.core, substr, args,  # noqa F821 function deprecated
                    bind=args.bind, bsmi=args.nambsmi)
    ts3D.writexyz(fname)
    strfiles.append(fname)
    getinputargs(args, fname)
    pfold = rootdir.split('/', 1)[-1]
    # check for molecule sanity
    sanity, d0 = ts3D.sanitycheck(True)
    if args.debug:
        print(('setting sanity diag, min dist at ' +
               str(d0) + ' (higher is better)'))
    this_diag.set_sanity(sanity, d0)
    this_diag.set_mol(ts3D)
    this_diag.write_report(fname+'.report')
    del ts3D
    if sanity:
        print(('WARNING: Generated complex is not good! Minimum distance between atoms:' +
              "{0:.2f}".format(d0)+'A\n'))
    print(('\nIn folder '+pfold+' generated 1 structure(s)!'))
    return strfiles, emsg, this_diag
