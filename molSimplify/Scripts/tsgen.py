#!/usr/bin/env python
# Written by Terry Gani for HJK Group
# Dpt of Chemical Engineering, MIT

# This script generates TS guesses based on a user-specified complex (core), substrate and reaction.

# Currently, only the hydrogen abstraction reaction is supported. 

# import custom modules
from molSimplify.Scripts.geometry import *
from molSimplify.Scripts.structgen import *
from molSimplify.Scripts.io import *
from molSimplify.Scripts.nn_prep import *
from molSimplify.Classes.globalvars import *
from molSimplify.Classes.rundiag import *
from molSimplify.Classes import globalvars
from molSimplify.Classes import mol3D
from molSimplify.Informatics.decoration_manager import*
# import standard modules
import os, sys
import openbabel
from pkg_resources import resource_filename, Requirement
import random, itertools
from numpy import log, arccos, cross, dot, pi

# geometric parameters
ABXang = 170 # mode 3
XYcoeff = 1.1
MXYang = 135 # mode 1
MXdistcoeff = 0.9 # mode 1

def getconnections(core,catom,Midx,BL,ABXang):
    # Gets all possible substrate connecting points (X in A-B...X-Y).
    # Given a fixed ABX angle and AB distance, these lie on a circle.
    # Due to limitations of geometry routines, we sample the sphere and 
    # keep points within ~1 deg of the circle.
    
    Ocoords = core.getAtom(catom).coords()
    Mcoords = core.getAtom(Midx).coords()
    backbcoords = alignPtoaxis(Ocoords,Ocoords,vecdiff(Ocoords,Mcoords),BL)
    am = atom3D('C',backbcoords)
    connPts = []
    for iphi in range(1,359,10):
        for itheta in range(1,179,1):
            P = PointTranslateSph(Ocoords,backbcoords,[BL,iphi,itheta])
            am.setcoords(P)
            ang = 180-vecangle(vecdiff(Ocoords,Mcoords),vecdiff(P,Ocoords))
            if abs(ang - ABXang) < 1:
                connPts.append(P)
    return connPts

def substplacecheap(core,connPts,catom):
    # Cheap method for determining optimal connecting point by empirically maximizing distances.

    corerem = mol3D()
    corerem.copymol3D(core)
    corerem.deleteatom(catom) # complex less O atom
    mdist = -1
    cpoint = [0,0,0]
    for P in connPts:
        if corerem.mindisttopoint(P) < 1: # auto-reject if too close
            d0 = -2
        else:
            d0 = distance(core.centermass(),P)+0.5*log(corerem.mindisttopoint(P)-1)
        if d0 > mdist:
            mdist = d0
            cpoint = P
    return cpoint 

def substplaceff_mode3(core,substr,substreact,compreact,cpoint,args,connected,frozenats):
    # Expensive method for determining optimal connecting point by trial and error FF optimization/energy evals.
    enc = 0
    # align substrate according to connection atom and shadow atom
    substr.alignmol(substr.getAtom(substreact),atom3D('H',cpoint))
    # perform rotations
    Bcoords = core.getAtomCoords(compreact)
    adjsidx = substr.getBondedAtoms(substreact)[0]
    if substr.natoms > 1:
        # align ligand center of symmetry
        substr = align_lig_centersym(Bcoords,substr,substreact,core,False)
    if substr.natoms > 2:
        # check for linear molecule and align
        substr = check_rotate_linear_lig(Bcoords,substr,substreact)
        # check for symmetric molecule
        substr = check_rotate_symm_lig(Bcoords,substr,substreact,core)
        # rotate around M-L axis to minimize steric repulsion
        substr = rotate_MLaxis_minimize_steric(Bcoords,substr,substreact,core)
    # distort substrate molecule
    adjsidx = substr.getBondedAtoms(substreact)[0]
    XYBL = XYcoeff*(substr.getAtom(substreact).rad + substr.getAtom(adjsidx).rad)
    substr.BCM(adjsidx,substreact,XYBL)
    # combine molecules
    ts3D = mol3D()
    ts3D.copymol3D(core)
    ts3D = ts3D.combine(substr)
    ts3D.charge += substr.charge
    if 'a' in args.ffoption or args.substplaceff:
        ts3D,enc = ffopt(args.ff,ts3D,connected,1,frozenats,False,[],'Adaptive')
    return ts3D,enc

def substplaceff_mode1(core,substr,substreact,compreact,cpoint,ligalignpt,args,connected,frozenats):
    # Expensive method for determining optimal connecting point by trial and error FF optimization/energy evals.
    # Slightly different routine between modes
    enc = 0
    Mcoords = core.getAtomCoords(compreact)
    adjsidx = substr.getBondedAtoms(substreact)[0]
    theta,u = rotation_params(substr.getAtomCoords(adjsidx),cpoint,ligalignpt)
    substr = rotate_around_axis(substr,cpoint,u,180-theta)

    if substr.natoms > 2:
        # same as aligning an equilibrium ligand
        substr = check_rotate_linear_lig(Mcoords,substr,substreact)
        substr = check_rotate_symm_lig(Mcoords,substr,substreact,core)
        substr = rotate_MLaxis_minimize_steric(Mcoords,substr,substreact,core)
    ts3D = mol3D()
    ts3D.copymol3D(core)
    ts3D = ts3D.combine(substr)
    if 'a' in args.ffoption or args.substplaceff:
        print('FF optimizing remainder of substrate')
        ts3D,enc = ffopt(args.ff,ts3D,connected,1,frozenats,False,[],'Adaptive')
    return ts3D,enc

def tsgen(mode,args,rootdir,core,substr,compreact,substreact,globs):
    # INPUT
    #   - mode: TS generation mode (see rungen.py)
    #   - args: placeholder for input arguments
    #   - rootdir: directory of current run
    #   - core: core mol3D (generated in rungen.py)
    #   - substr: substrate mol3D
    #   - compreact: complex reacting atom(s) (see rungen)
    #   - substreact: substrate reacting atom(s) (see rungen)
    #   - globs: class with global variables
    # OUTPUT
    #   - strfiles: list of xyz files generated
    #   - emsg: error messages
    emsg = False
    this_diag = run_diag()
    strfiles = []
    adjsidx = substr.getBondedAtoms(substreact)[0]
    adjcidx = core.getBondedAtoms(compreact)[0]
    # initialize connecting and frozen atoms for FF opt
    frozenats = []
    for i in range(0,core.natoms):
        frozenats.append(i)
    # also freeze the abstracted atom and the heavy atom bonded to it
    frozenats.append(core.natoms+substreact)
    frozenats.append(core.natoms+adjsidx)
    connected = [core.natoms+substreact]
    ########## START FUNCTIONALIZING ##########
    sanity = False
    if mode == 2:
        emsg = 'Sorry, this mode is not supported yet. Exiting...'
        return strfiles, emsg, this_diag
    elif mode == 1: # oxidative addition of a single group
		# get first connecting point
        MXBL = MXdistcoeff*(core.getAtom(compreact).rad + substr.getAtom(substreact).rad)
        cpoint = getconnection(core,compreact,MXBL)
        # distort substrate molecule
        XYBL = XYcoeff*(substr.getAtom(substreact).rad + substr.getAtom(adjsidx).rad)
        substr.BCM(adjsidx,substreact,XYBL)
        # align substrate molecule
        substr.alignmol(substr.getAtom(substreact),atom3D('H',cpoint))
        tmp3D = mol3D()
        tmp3D.copymol3D(core)
        tmp3D.addAtom(atom3D('Cl',cpoint))
        ligalignpts = getconnections(tmp3D,tmp3D.natoms-1,compreact,XYBL,MXYang)
        if args.substplaceff:
            # full FF substrate placement
            print('Full FF-based substrate placement specified.')
            en_min = 1e6
            for n,P in enumerate(ligalignpts):
                print('Evaluating FF energy of point '+str(n+1)+' of '+str(len(ligalignpts)))
                coretmp = mol3D()
                coretmp.copymol3D(core)
                substrtmp = mol3D()
                substrtmp.copymol3D(substr)
                ts3Dtmp,enc = substplaceff_mode1(coretmp,substrtmp,substreact,compreact,cpoint,P,args,connected,frozenats)
                if enc < en_min:
                    en_min = enc
                    ts3D = mol3D()
                    ts3D.copymol3D(ts3Dtmp)
        else:
            # cheap substrate placement
            print('Cheap substrate placement')
            ligalignpt = substplacecheap(core,ligalignpts,compreact)
            ts3D,enc = substplaceff_mode1(core,substr,substreact,compreact,cpoint,ligalignpt,args,connected,frozenats)
    elif mode == 3: # abstraction
        # distort A-B bond
        ABBL = distance(core.getAtomCoords(compreact),core.getAtomCoords(adjcidx)) + 0.05*(core.getAtom(compreact).rad + core.getAtom(adjcidx).rad)
        core.BCM(compreact,adjcidx,ABBL)
        # set B-X distance 
        BXBL = 1.1*(substr.getAtom(substreact).rad + core.getAtom(compreact).rad)
        # get possible connecting points
        connPts = getconnections(core,compreact,adjcidx,BXBL,ABXang)
        if args.substplaceff:
            # full FF substrate placement
            print('Full FF-based substrate placement specified.')
            en_min = 1e6
            for n,P in enumerate(connPts):
                print('Evaluating FF energy of point '+str(n+1)+' of '+str(len(connPts)))
                coretmp = mol3D()
                coretmp.copymol3D(core)
                substrtmp = mol3D()
                substrtmp.copymol3D(substr)
                ts3Dtmp,enc = substplaceff_mode3(coretmp,substrtmp,substreact,compreact,P,args,connected,frozenats)
                if enc < en_min:
                    en_min = enc
                    ts3D = mol3D()
                    ts3D.copymol3D(ts3Dtmp)
        else:
            # cheap substrate placement
            print('Cheap substrate placement')
            cpoint = substplacecheap(core,connPts,compreact)
            ts3D,enc = substplaceff_mode3(core,substr,substreact,compreact,cpoint,args,connected,frozenats)
            if 'a' in args.ffoption:
                print('FF optimized remainder of substrate')
    ts3D.charge += substr.charge
    ############ END FUNCTIONALIZING ###########
    fname = name_TS(rootdir,args.core,substr,args,bind=args.bind,bsmi=args.nambsmi)
    ts3D.writexyz(fname)
    strfiles.append(fname)
    getinputargs(args,fname)
    pfold = rootdir.split('/',1)[-1]
    # check for molecule sanity
    sanity,d0 = ts3D.sanitycheck(True)
    if args.debug:
        print('setting sanity diag, min dist at ' +str(d0) + ' (higher is better)')
    this_diag.set_sanity(sanity,d0)
    this_diag.set_mol(ts3D)
    this_diag.write_report(fname+'.report')
    del ts3D
    if sanity:
        print 'WARNING: Generated complex is not good! Minimum distance between atoms:'+"{0:.2f}".format(d0)+'A\n'
    print '\nIn folder '+pfold+' generated 1 structure(s)!'
    return strfiles, emsg, this_diag
