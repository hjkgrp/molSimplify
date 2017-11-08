# Written by Tim Ioannidis for HJK Group
# Extended by JP Janet
# Revised by Terry Gani
# Dpt of Chemical Engineering, MIT

##########################################################
######## This script generates a collection  #############
#####  of randomly placed binding species around   #######
########   a functionalized ferrocene core   #############
######## and then creates the required input #############
######## and job files for running terachem  #############
########  calculations on these structures   #############
##########################################################

# import custom modules
from molSimplify.Scripts.geometry import *
from molSimplify.Scripts.io import *
from molSimplify.Scripts.nn_prep import *
from molSimplify.Classes.globalvars import *
from molSimplify.Classes.rundiag import *
from molSimplify.Classes import globalvars
from molSimplify.Classes import mol3D
from molSimplify.Informatics.decoration_manager import*
# import standard modules
import os, sys
from pkg_resources import resource_filename, Requirement
import openbabel, random, itertools
from numpy import log, arccos, cross, dot, pi
numpy.seterr(all='raise')

def norm(u):
    # Euclidean norm
    # INPUT
    #   - u: n-element list
    # OUTPUT
    #   - d: Euclidean norm
    d = 0.0
    for u0 in u:
        d += (u0*u0)
    d = sqrt(d)
    return d

def setdiff(a,b):
    # gets the elements in a that are not in b
    # INPUT
    #   - a: list with elements
    #   - b: list with elements
    # OUTPUT
    #   - aa: elements in a that are not in b
    b = set(b)
    return [aa for aa in a if aa not in b]

##########################################
#### gets all possible combinations   ####
#### for connection atoms in geometry ####
####  in the case of forced order  #######
########   or unknown geometry   #########
##########################################
def getbackbcombsall(nums):
    bbcombs = []
    for i in range(1,len(nums)+1):
        bbcombs += list(itertools.combinations(nums,i)) 
    for i,tup in enumerate(bbcombs):
        bbcombs[i] = list(tup)
    return bbcombs

def getnupdateb(backbatoms,denticity):
    # gets a combination that satisfies denticity and updates dictionary
    dlist = []
    batoms = []
    # find matching combination
    for b in backbatoms:
        if len(b)==denticity:
            batoms = b
            break
    # loop and find elements to delete
    for b in batoms:
        for i,bcomb in enumerate(backbatoms):
            if b in bcomb and i not in dlist:
                dlist.append(i)
    dlist.sort(reverse=True) # sort
    # delete used points
    for i in dlist:
        del backbatoms[i]
    if len(batoms) < 1:
        print 'No more connecting points available..'
    return batoms,backbatoms

def getsmilescat(args,indsmi):
    # gets connection atoms of smiles string
    # INPUT
    #   - args: placeholder for input arguments
    #   - nosmiles: number of ligands defined via SMILES
    #   - indsmi: index of SMILES string ligand (like counter)
    # OUTPUT
    #   - tt: list of connection atoms
    tt= []  # initialize list of connection atoms
    if args.smicat and len(args.smicat)>indsmi: # get connection atom(s)
        tt = args.smicat[indsmi] # default value
    else:
        tt = [0] # default value 0 connection atom
    return tt

def getsmident(args,indsmi):
    # gets denticity of smiles string
    # INPUT
    #   - args: placeholder for input arguments
    #   - nosmiles: number of ligands defined via SMILES
    #   - indsmi: index of SMILES string ligand (like counter)
    # OUTPUT
    #   - SMILES ligand denticity (int)
    ### check for denticity specification in input ###
    # if denticity is specified return this
    if args.smicat and len(args.smicat) > indsmi:
        return int(len(args.smicat[indsmi]))
    # otherwise return default
    else:
        return 1

def init_ANN(args,ligands,occs,dents,batslist,tcats,licores):
    # initialize ANN
    ANN_attributes = dict()
    if args.skipANN:
         print('Skipping ANN')
         ANN_flag = False
         ANN_bondl = 0
         ANN_reason = 'ANN skipped by user'
    else:
         try:
             ANN_flag,ANN_reason,ANN_attributes = ANN_preproc(args,ligands,occs,dents,batslist,tcats,licores)
             if ANN_flag:
                 ANN_bondl = ANN_attributes['ANN_bondl']
             else:
                 ANN_bondl = 0
                 if args.debug:
                     print("ANN called failed with reason: " + ANN_reason)
         except:
             print("ANN call rejected")
             ANN_reason = 'uncaught exception'
             ANN_flag = False
             ANN_bondl = 0
    return ANN_flag,ANN_bondl,ANN_reason,ANN_attributes 

def init_template(args,cpoints_required):
    # Initializes core (core3D) and template (m3D)
    # Input
    # args: namespace
    # cpoints_required: number of connecting points required
    # Output
    # core3D: mol3D of core
    # m3D: mol3D of template (core + connecting point placeholders)

    # initialize core and template
    core3D = mol3D()
    m3D = mol3D()
    # container for ordered list of core reference atoms
    corerefatoms = mol3D()         
    
    # geometry load flag
    geom = False
    backbatoms = []
    coord = 0 
    # build mode
    if args.geometry and not args.ccatoms:
        # determine geometry
        coord = int(args.coord)
        # get available geometries
        coords,geomnames,geomshorts,geomgroups = getgeoms()
        maxcoord = len(geomgroups)
        # get list of possible combinations for connecting points
        bbcombsdict = bbcombs_mononuc
        # get a default geometry
        geom = geomgroups[coord-1][0]
        # check if geometry is defined and overwrite
        if args.geometry in geomshorts:
            geom = args.geometry
        else:
            emsg = "Requested geometry not available."+"Defaulting to "+geomgroups[coord-1][0]
            if args.gui:
                qqb = mQDialogWarn('Warning',emsg)
                qqb.setParent(args.gui.wmain)
            print emsg
        # load predefined backbone coordinates
        corexyz = loadcoord(geom)
        # load backbone atom combinations
        if geom in bbcombsdict.keys() and not args.ligloc:
            backbatoms = bbcombsdict[geom]
        else:
            nums = range(1,len(corexyz))
            backbatoms = getbackbcombsall(nums)
        # distort if requested
        if args.pangles:
            corexyz = modifybackbonep(corexyz,args.pangles) # point distortion
        if args.distort:
            corexyz = distortbackbone(corexyz,args.distort) # random distortion
        # add center atom
        if args.core[0].upper()+args.core[1:] in elementsbynum:    
            centeratom = args.core[0].upper()+args.core[1:]  
        else:
            print('WARNING: Core is not an element. Defaulting to Fe')
            centeratom = 'Fe'
        core3D.addAtom(atom3D(centeratom,corexyz[0]))
        m3D.copymol3D(core3D)
        # add connecting points to template
        for m in range(1,coord+1):
            m3D.addAtom(atom3D('X',corexyz[m]))
            corerefatoms.addAtom(core3D.getAtom(0))
            #corerefatoms.append(0)
        
    # functionalize mode
    else:
        # check ccatoms
        if not args.ccatoms:
            emsg = 'Connection atoms for custom core not specified. Defaulting to 1!\n'
            print emsg
            if args.gui:
                qqb = mQDialogWarn('Warning',emsg)
                qqb.setParent(args.gui.wmain)
        ccatoms = args.ccatoms if args.ccatoms else [0]
        coord = len(ccatoms)
        if args.debug:
            print('setting ccatoms ' + str(ccatoms))
         
        # load core
        core,emsg = core_load(args.core)
        if emsg:
            return False,emsg
        core.convert2mol3D()
        core3D.copymol3D(core)
        m3D.copymol3D(core3D)
        for i in range(cpoints_required):
            if not args.replig:
                # not replacing ligands: add Xs to ccatoms
                # NOTE: ccatoms should be a list with # elements = cpoints_required
                cpoint = getconnection(m3D,ccatoms[i],2)
                # store core reference atom
                conatom3D = atom3D(core3D.getAtom(ccatoms[i]).sym,core3D.getAtom(ccatoms[i]).coords())
                corerefatoms.addAtom(conatom3D)
                #corerefatoms.append(ccatoms[i])
                # add connecting points to template
                m3D.addAtom(atom3D(Sym='X',xyz=cpoint))
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
                        delatoms = core3D.findsubMol(ccatoms[i],cat)
                        if len(delatoms) < minmol: # check for smallest
                            mindelats = delatoms
                            minmol = len(delatoms) # size
                            atclose = cat # connection atom
                        # if same atoms in ligand get shortest distance
                        elif len(delatoms)==minmol:
                            d0 = core3D.getAtom(ccatoms[i]).distance(core3D.getAtom(cat))
                            d1 = core3D.getAtom(ccatoms[i]).distance(core3D.getAtom(mindelats[0]))
                            if d0 < d1:
                                mindelats = delatoms
                                atclose = cat
                    # store core reference atom
                    conatom3D = atom3D(core3D.getAtom(atclose).sym,core3D.getAtom(atclose).coords())
                    corerefatoms.addAtom(conatom3D)
                    #corerefatoms.append(atclose)
                    delatoms = mindelats
                    # for multidentate ligands: if a submolecule contains multiple ccatoms, add all of them to the template
                    for atomidx in delatoms:
                        if atomidx in ccatoms[i+1:]:
                            # add connecting points to template
                            m3D.addAtom(atom3D(Sym='X',xyz=core3D.getAtom(atomidx).coords()))
                            ccatoms.remove(atomidx)
                            corerefatoms.addAtom(conatom3D)
                    # update remaining ccatoms according to deleted atoms
                    if len(ccatoms) > i+1:
                        for cccat in range(i+1,len(ccatoms)):
                            lshift = len([a for a in delatoms if a < ccatoms[cccat]])
                            ccatoms[cccat] -= lshift
                    # delete submolecule        
                    core3D.deleteatoms(delatoms)
                    m3D.deleteatoms(delatoms)
                    # add connecting points to template
                    m3D.addAtom(atom3D(Sym='X',xyz=cpoint))
                except IndexError:
                    pass        
            nums = m3D.findAtomsbySymbol('X')
            backbatoms = getbackbcombsall(nums)
    # set charge from oxidation state if desired
    if args.calccharge:
        if args.oxstate:
            if args.oxstate in romans.keys():
                core3D.charge = int(romans[args.oxstate])
            else:
                core3D.charge = int(args.oxstate)
    return m3D,core3D,geom,backbatoms,coord,corerefatoms

def init_ligand(args,lig,tcats,keepHs,i):
    # initializes ligand 3D geometry and properties
    globs = globalvars()
    rempi = False
    ## check if ligand should decorated
    if args.decoration and args.decoration_index:
        if len(args.decoration) > i and len(args.decoration_index) > i:
            if args.decoration[i]:
                if args.debug:
                    print('decorating ' + str(ligand) + ' with ' +str(args.decoration[i]) + ' at sites '  + str(args.decoration_index))
                lig = decorate_ligand(args,ligand,args.decoration[i],args.decoration_index[i])
    # if SMILES string
    if not lig.cat and tcats[i]:
        if 'c' in tcats[i]:
            lig.cat = [lig.natoms]
        else:
            lig.cat = tcats[i]
    lig3D = mol3D()
    lig3D.copymol3D(lig)
    lig3D.convert2OBMol()
    # check for pi-coordinating ligand
    ligpiatoms = []
    if 'pi' in lig.cat:
        lig3Dpiatoms = mol3D()
        for k in lig.cat[:-1]:
            lig3Dpiatoms.addAtom(lig3D.getAtom(k))
            lig3Dpiatoms.addAtom(lig3D.getAtom(k))
        ligpiatoms = lig.cat[:-1]
        lig3D.addAtom(atom3D('C',lig3Dpiatoms.centermass()))
        lig.cat = [lig3D.natoms-1]
        rempi = True
    # perform FF optimization if requested (not supported for pi-coordinating ligands)
    if args.ff and 'b' in args.ffoption and not rempi:
        if 'b' in lig.ffopt.lower():
            print 'FF optimizing ligand'
            lig,enl = ffopt(args.ff,lig,lig.cat,0,[],False,[],100)
    # skip hydrogen removal for pi-coordinating ligands    
    if not rempi: 
        # check smarts match
        if 'auto' in keepHs[i]:
            for j,catom in enumerate(lig.cat):
                match = findsmarts(lig3D.OBMol,globs.remHsmarts,catom)
                if match:
                    keepHs[i][j] = False
                else:
                    keepHs[i][j] = True
        # remove one hydrogen from each connecting atom with keepH false
        for j,cat in enumerate(lig.cat):
            Hs = lig3D.getHsbyIndex(cat)
            if len(Hs) > 0 and not keepHs[i][j]:
                if args.debug:
                    print('modifying charge down from ' + str(lig3D.charge))
                    try:
                        print('debug keepHs check, removing? ' + str(keepHs) + ' i = ' +str(i)+ 
                    ' , j = ' +str(j) + ' lig = ' + str(lig.coords()) + ' is keephs[i] ' + str(keepHs[i] ) +
                     ' length of keepHs list  '+ str(len(keepHs)))
                    except:
                        pass 
                # check for cats indices
                if cat > Hs[0]:
                    lig.cat[j] -= 1
                lig3D.deleteatom(Hs[0])
                lig3D.charge = lig3D.charge - 1
    return lig3D,rempi,ligpiatoms

def modifybackbonep(backb, pangles):
    # modifies backbone according to pangles
    # INPUT
    #   - backb: list with points comprising the backbone
    #   - pangles: angles for distorting corresponding backbone points  (pairs of theta/phi)
    # OUTPUT
    #   - backb: list with modified points comprising the backbone
    for i,ll in enumerate(pangles):
        if ll:
            theta = pi*float(ll.split('/')[0])/180.0
            phi = pi*float(ll.split('/')[-1])/180.0
            backb[i+1] = PointTranslateSph(backb[0],backb[i+1],[distance(backb[0],backb[i+1]),theta,phi])
    return backb

def distortbackbone(backb, distort):
    # randomly distorts backbone
    # INPUT
    #   - backb: list with points comprising the backbone
    #   - distort: % distortion of the backbone
    # OUTPUT
    #   - backb: list with modified points comprising the backbone
    for i in range(1,len(backb)):
            theta = random.uniform(0.0,0.01*int(distort)) # *0.5
            phi = random.uniform(0.0,0.01*int(distort)*0.5) # *0.5
            backb[i] = PointTranslateSph(backb[0],backb[i],[distance(backb[0],backb[i]),theta,phi])
    return backb

def smartreorderligs(args,ligs,dentl,licores):
    # reorder ligands
    globs = globalvars()
    # INPUT
    #   - args: placeholder for input arguments
    #   - ligs: list of ligands
    #   - dents: ligand denticities
    # OUTPUT
    #   - indcs: reordering indices
    # check for forced order
    if not args.ligalign:
        indcs = range(0,len(ligs))
        return indcs
    lsizes = []
    for ligand in ligs:
        lig,emsg = lig_load(ligand) # load ligand
        lig.convert2mol3D()
        lsizes.append(lig.natoms)
    # group by denticities
    dents = list(set(dentl))
    ligdentsidcs = [[] for a in dents]
    for i,dent in enumerate(dentl):
        ligdentsidcs[dents.index(dent)].append(i)
    # sort by highest denticity first
    ligdentsidcs = list(reversed(ligdentsidcs))
    indcs = []
    # within each group sort by size (smaller first)
    for ii,dd in enumerate(ligdentsidcs):
        locs = [lsizes[i] for i in dd]
        locind = [i[0] for i in sorted(enumerate(locs), key=lambda x:x[1])]
        for l in locind:
            indcs.append(ligdentsidcs[ii][l])
    return indcs

def ffopt(ff,mol,connected,constopt,frozenats,frozenangles,mlbonds,nsteps):
    # Main constrained FF opt routine
    # INPUT ffopt(args.ff,core3D,connected,2,frozenats,freezeangles,MLoptbds)
    #   - ff: force field to use, available MMFF94, UFF, Ghemical, GAFF
    #   - mol: mol3D to be ff optimized
    #   - connected: indices of connection atoms to metal
    #   - constopt: flag for constrained optimization
    #   - nsteps: 'Adaptive': run only enough steps to stop clashing, number: run that many steps (default 200), 0: report energy only)
    # OUTPUT
    #   - mol: force field optimized mol3D
    globs = globalvars()
    metals = range(21,31)+range(39,49)+range(72,81)
    ### check requested force field
    ffav = 'mmff94, uff, ghemical, gaff, mmff94s' # force fields
    if ff.lower() not in ffav:
        print 'Requested force field not available. Defaulting to MMFF94'
        ff = 'mmff94'
    # perform constrained ff optimization if requested after #
    if (constopt > 0):
        ### get metal
        midx = mol.findMetal()
        ### convert mol3D to OBMol
        mol.convert2OBMol()
        OBMol = mol.OBMol
        # initialize force field
        forcefield = openbabel.OBForceField.FindForceField(ff)
        ### initialize constraints
        constr = openbabel.OBFFConstraints()      
        ### openbabel indexing starts at 1 ### !!!
        # convert metals to carbons for FF
        indmtls = []
        mtlsnums = []
        for iiat,atom in enumerate(openbabel.OBMolAtomIter(OBMol)):
            if atom.GetAtomicNum() in metals:
                indmtls.append(iiat)
                mtlsnums.append(atom.GetAtomicNum())
                atom.SetAtomicNum(6)
        # freeze and ignore metals
        for midxm in indmtls:
            constr.AddAtomConstraint(midxm+1) # indexing babel           
        # add coordinating atom constraints
        for ii,catom in enumerate(connected):
            if constopt==1 or frozenangles:
                constr.AddAtomConstraint(catom+1) # indexing babel
            else:
                constr.AddDistanceConstraint(midx+1,catom+1,mlbonds[ii]) # indexing babel
        bridgingatoms = []        
        # identify bridging atoms in the case of bimetallic cores
        for i in range(mol.natoms):
            nbondedmetals = len([idx for idx in range(len(mol.getBondedAtomsOct(i))) if mol.getAtom(mol.getBondedAtomsOct(i)[idx]).ismetal()])
            if nbondedmetals > 1:
                bridgingatoms.append(i)
        # ensure fake carbons have correct valence for FF setup
        for m in indmtls:
            i = 0
            # too many bonds: delete bonds
            if OBMol.GetAtom(m+1).GetValence() > 4:
                while OBMol.GetAtom(m+1).GetValence() > 4:
                    # delete bond only if it is a real metal-terminal ligand bond
                    if OBMol.GetBond(m+1,mol.getBondedAtomsOct(m)[i]+1) is not None and mol.getBondedAtomsOct(m)[i] not in bridgingatoms:
                        OBMol.DeleteBond(OBMol.GetBond(m+1,mol.getBondedAtomsOct(m)[i]+1))
                    i += 1
            # too few bonds: add bonds
            elif OBMol.GetAtom(m+1).GetValence() == 0:
                for i in mol.getBondedAtomsOct(m+1):
                    if OBMol.GetAtom(m+1).GetValence() < 4:
                        OBMol.AddBond(m+1,i+1,1)               
        ### freeze small ligands
        for cat in frozenats:
            constr.AddAtomConstraint(cat+1) # indexing babel
        ### set up forcefield
        s = forcefield.Setup(OBMol,constr)
        if s == False:
            print('FF setup failed')
        ### force field optimize structure
        elif nsteps == 'Adaptive':
            i = 0
            while i < 20:
                forcefield.ConjugateGradients(50)
                forcefield.GetCoordinates(OBMol)
                mol.OBMol = OBMol
                mol.convert2mol3D()              
                overlap,mind = mol.sanitycheck(True)
                if not overlap:
                    break
                i += 1
        elif nsteps != 0:
            try:
                n = nsteps
            except:
                n = 100    
            forcefield.ConjugateGradients(n)
            forcefield.GetCoordinates(OBMol)
            mol.OBMol = OBMol
            mol.convert2mol3D()
        else:
            forcefield.GetCoordinates(OBMol)
        en = forcefield.Energy()
        mol.OBMol = OBMol
        # reset atomic number to metal
        for i,iiat in enumerate(indmtls):
            mol.OBMol.GetAtomById(iiat).SetAtomicNum(mtlsnums[i])
        mol.convert2mol3D()
        del forcefield, constr, OBMol
    else:
        ### initialize constraints
        constr = openbabel.OBFFConstraints()
        ### add atom constraints
        for catom in connected:
            constr.AddAtomConstraint(catom+1) # indexing babel
        ### set up forcefield
        forcefield = openbabel.OBForceField.FindForceField(ff)
        #if len(connected) < 2:
            #mol.OBMol.localopt('mmff94',100) # add hydrogens and coordinates
        OBMol = mol.OBMol # convert to OBMol
        s = forcefield.Setup(OBMol,constr)   
        ### force field optimize structure
        if OBMol.NumHvyAtoms() > 10:
            forcefield.ConjugateGradients(50)
        else:
            forcefield.ConjugateGradients(200)
        forcefield.GetCoordinates(OBMol)
        en = forcefield.Energy()
        mol.OBMol = OBMol
        mol.convert2mol3D()
        del forcefield, constr, OBMol
    return mol,en

#def getconnection(core,catom):
    ## Use FF to estimate optimum backbone positioning (Tim)
    #cm = core.centermass()
    #ff = 'UFF'
    #metals = range(21,31)+range(39,49)+range(72,81)
    #### get hydrogens
    #Hlist = core.getHs()
    #### add fake atoms for catoms
    #ncore = core.natoms
    ## add fake atom in local centermass axis
    #coords = core.getAtom(catom).coords()
    #dd = distance(coords,core.centermass())
    #backbcoords = alignPtoaxis(coords,coords,vecdiff(coords,core.centermass()),1.5)
    #bopt = backbcoords
    ## manually find best positioning
    #am = mol3D()
    #am.addAtom(atom3D('C',backbcoords))
    ##for ii in range(0,toconnect-1):
    #ii = 0
    #P = PointTranslateSph(coords,am.atoms[ii].coords(),[1.5,45,30])
    #am.addAtom(atom3D('C',P))
    #setopt = []
    #mdist = -1
    ##for ii in range(0,toconnect):
    #ii = 0
    #for itheta in range(0,360,3):
        #for iphi in range(0,180,2):
            #P = PointTranslateSph(coords,backbcoords,[1.5,itheta,iphi])
            #am.atoms[ii].setcoords(P)
            #dd = 0
            ##for idx in range(0,toconnect):
            #dd += distance(cm,am.atoms[0].coords())
            #if (am.mindistmol() > 0.0):
                #d0 = dd+0.5*(log(core.mindist(am)*am.mindistmol()))
                #print d0
            #if d0 > mdist:
                #mdist = d0
                #setopt = am.coordsvect()
    ##for ii in range(0,toconnect):
    #core.addAtom(atom3D('C',setopt[0]))
    ##connPts = []
    ##for ii in range(0,toconnect):
    ##connPts.append(core.getAtom(ncore+ii).coords())
    #print core.getAtom(ncore).coords()
    #return core.getAtom(ncore).coords()

def getconnection(core,cidx,BL):
    # finds the optimum attachment point for an atom/group to a central atom given the desired bond length
    # objective function maximizes the minimum distance between attachment point and other groups bonded to the central atom
    ncore = core.natoms
    groups = core.getBondedAtoms(cidx)
    ccoords = core.getAtom(cidx).coords()
    # brute force search
    cpoint = []
    objopt = 0
    for itheta in range(1,359,1):
        for iphi in range(1,179,1):
            P = PointTranslateSph(ccoords,ccoords,[BL,itheta,iphi])
            dists = []
            for ig in groups:
                dists.append(distance(core.getAtomCoords(ig),P))
            obj = min(dists)
            if obj > objopt:
                objopt = obj
                cpoint = P
    return cpoint

def findsmarts(lig3D,smarts,catom):
    #returns true if connecting atom of lig3D is part of SMARTS pattern
    #lig3D: OBMol of mol3D
    #smarts: list of SMARTS patterns (strings)
    #catom: connecting atom of lig3D (zero based numbering)
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

def align_lig_centersym(corerefcoords,lig3D,atom0,core3D,EnableAutoLinearBend):
    # Aligns a ligand's center of symmetry along the metal-connecting atom axis.
    # corerefcoords: core reference coordinates
    # lig3D: mol3D of ligand to be rotated
    # atom0: ligand connecting atom
    # core3D: core mol3D
    # RETURNS
    # lig3D_aligned: mol3D of aligned ligand
    # rotate to align center of symmetry
    globs = globalvars()
    r0 = corerefcoords
    r1 = lig3D.getAtom(atom0).coords()
    lig3Db = mol3D()
    lig3Db.copymol3D(lig3D)
    auxmol = mol3D()
    for at in lig3D.getBondedAtoms(atom0):
        auxmol.addAtom(lig3D.getAtom(at))
    r2 = auxmol.centersym()
    theta,u = rotation_params(r0,r1,r2)
    # rotate around axis and get both images
    lig3D = rotate_around_axis(lig3D,r1,u,theta)
    lig3Db = rotate_around_axis(lig3Db,r1,u,theta-180)
    # compare shortest distances to core reference coordinates
    d2 = distance(r0,lig3D.centersym())
    d1 = distance(r0,lig3Db.centersym())
    lig3D = lig3D if (d1 < d2)  else lig3Db # pick best one
    # additional rotation for bent terminal connecting atom:
    if auxmol.natoms == 1:
        if distance(auxmol.getAtomCoords(0),lig3D.getAtomCoords(atom0)) > 0.8*(auxmol.getAtom(0).rad + lig3D.getAtom(atom0).rad) and EnableAutoLinearBend: 
            print('bending of linear terminal ligand')
            ##warning: force field might overwrite this
            r1 = lig3D.getAtom(atom0).coords()
            r2 = auxmol.getAtom(0).coords()
            theta,u = rotation_params([1,1,1],r1,r2)
            lig3D = rotate_around_axis(lig3D,r1,u,globs.linearbentang) 
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned    

def align_linear_pi_lig(mcoords,lig3D,atom0,ligpiatoms):
    # Aligns a linear pi ligand's connecting point to the metal-ligand axis.
    # mcoords: metal coordinates
    # lig3D: ligand mol3D
    # atom0: ligand center of mass placeholder (where is this written?)
    # ligpiatoms: ligand pi-connecting atoms
    # RETURNS
    # lig3D_aligned: mol3D of aligned ligand
    
    # first rotate in the metal plane to ensure perpendicularity
    r0 = mcoords
    r1 = lig3D.getAtom(ligpiatoms[0]).coords()
    r2 = lig3D.getAtom(ligpiatoms[1]).coords()
    theta,u = rotation_params(r0,r1,r2)
    objfuncopt = 90
    thetaopt = 0
    for theta in range(0,360,1):
        lig3D_tmp = mol3D()
        lig3D_tmp.copymol3D(lig3D)
        lig3D_tmp = rotate_around_axis(lig3D_tmp, lig3D_tmp.getAtom(atom0).coords(), u, theta)
        #objfunc = abs(vecangle(vecdiff(lig3D_tmp.getAtom(atom0).coords(),mcoords),vecdiff(lig3D_tmp.getAtom(ligpiatoms[0]).coords(),lig3D_tmp.getAtom(ligpiatoms[1]).coords()))-90)
        objfunc = abs(distance(lig3D_tmp.getAtom(ligpiatoms[0]).coords(),mcoords) - distance(lig3D_tmp.getAtom(ligpiatoms[1]).coords(),mcoords))
        if objfunc < objfuncopt:
            thetaopt = theta
            objfuncopt = objfunc
            lig3Dopt = mol3D() # lig3Dopt = lig3D_tmp DOES NOT WORK!!!
            lig3Dopt.copymol3D(lig3D_tmp) 
    lig3D = lig3Dopt
    # then rotate 90 degrees about the bond axis to further reduce steric repulsion
    r1 = lig3D.getAtom(ligpiatoms[0]).coords()
    r2 = lig3D.getAtom(ligpiatoms[1]).coords()
    u = vecdiff(r1,r2)
    lig3D_tmpa = mol3D()
    lig3D_tmpa.copymol3D(lig3D)
    lig3D_tmpa = rotate_around_axis(lig3D_tmpa, lig3D_tmpa.getAtom(atom0).coords(), u, 90)
    lig3D_tmpb = mol3D()
    lig3D_tmpb.copymol3D(lig3D)                        
    lig3D_tmpb = rotate_around_axis(lig3D_tmpb, lig3D_tmpb.getAtom(atom0).coords(), u, -90)
    d1 = distance(mcoords,lig3D_tmpa.centermass())
    d2 = distance(mcoords,lig3D_tmpb.centermass())
    #lig3D = lig3D if (d1 < d2)  else lig3Db 
    lig3D = lig3D_tmpa if (d1 > d2) else lig3D_tmpb # pick the better structure
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned 

def check_rotate_linear_lig(corerefcoords,lig3D,atom0):
    # Checks if ligand has a linear coordination environment (e.g., OCO) and ensures perpendicularity to M-L axis
    # corerefcoords: core reference coordinates
    # lig3D: ligand mol3D
    # atom0: ligand center of mass placeholder (where is this written?)
    # RETURNS
    # lig3D_aligned: mol3D of aligned ligand
    
    auxm = mol3D()
    lig3D_aligned = mol3D()
    for at in lig3D.getBondedAtoms(atom0):
        auxm.addAtom(lig3D.getAtom(at))
    if auxm.natoms > 1:
        r0 = lig3D.getAtom(atom0).coords()
        r1 = auxm.getAtom(0).coords()
        r2 = auxm.getAtom(1).coords()
        if checkcolinear(r1,r0,r2):
            # rotate so that O-C-O bond is perpendicular to M-L axis
            theta,urot = rotation_params(r1,corerefcoords,r2)
            theta = vecangle(vecdiff(r0,corerefcoords),urot)
            lig3D = rotate_around_axis(lig3D,r0,urot,theta)
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned

def check_rotate_symm_lig(corerefcoords,lig3D,atom0,core3D):
    # Checks if ligand has is symmetric about connecting atom (center of symmetry coincides with connecting atom) and minimizes clashes with rest of complex
    # corerefcoords: core reference coordinates
    # lig3D: ligand mol3D
    # atom0: ligand center of mass placeholder (where is this written?)
    # core3D: mol3D of partially built complex
    # RETURNS
    # lig3D_aligned: mol3D of aligned ligand
    
    if distance(lig3D.getAtom(atom0).coords(),lig3D.centersym()) < 8.0e-2:
        at = lig3D.getBondedAtoms(atom0)
        r0 = lig3D.getAtom(atom0).coords()
        r1 = lig3D.getAtom(at[0]).coords()
        r2 = lig3D.getAtom(at[1]).coords()
        theta,u = rotation_params(r0,r1,r2)
        theta = vecangle(u,vecdiff(r0,corerefcoords))
        urot = cross(u,vecdiff(r0,corerefcoords))
        # rotate around axis and get both images
        lig3Db = mol3D()
        lig3Db.copymol3D(lig3D)
        lig3D = rotate_around_axis(lig3D,r0,urot,theta)
        lig3Db = rotate_around_axis(lig3Db,r0,urot,-theta)
        # compute shortest distances to core
        d2 = lig3D.mindist(core3D)
        d1 = lig3Db.mindist(core3D)
        lig3D = lig3D if (d1 < d2)  else lig3Db # pick best one
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned

def rotate_MLaxis_minimize_steric(corerefcoords,lig3D,atom0,core3D):
    # Rotates aligned ligand about M-L axis to minimize steric clashes with rest of complex
    # corerefcoords: core reference coordinates
    # lig3D: ligand mol3D
    # atom0: ligand center of mass placeholder (where is this written?)
    # core3D: mol3D of partially built complex
    # RETURNS
    # lig3D_aligned: mol3D of aligned ligand
    
    r1 = lig3D.getAtom(atom0).coords()
    u = vecdiff(r1,corerefcoords)
    dtheta = 2
    optmax = -9999
    totiters = 0
    lig3Db = mol3D()
    lig3Db.copymol3D(lig3D)
    # maximize a combination of minimum distance between atoms and center of mass distance
    while totiters < 180:
        lig3D = rotate_around_axis(lig3D,r1,u,dtheta)
        d0 = lig3D.mindist(core3D) # shortest distance
        d0cm = lig3D.distance(core3D) # center of mass distance
        iteropt = d0cm+10*log(d0)
        if (iteropt > optmax): # if better conformation, keep
            lig3Db = mol3D()
            lig3Db.copymol3D(lig3D)
            optmax = iteropt
        totiters += 1
    lig3D = lig3Db
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned    

def rotate_catom_fix_Hs(lig3D,catoms,n,mcoords,core3D):
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
    for atom in lig3D.getBondedAtomsnotH(catoms[n]):
        subm = lig3D.findsubMol(atom,catoms[n])
        if len(list(set(subm).intersection(catoms_other))) == 0:
            danglinggroup = subm
        else:
            bridginggroup = subm
            anchoratom = list(set(subm).intersection(lig3D.getBondedAtoms(catoms[n])))[0]  
    for atom in danglinggroup:
        confrag3D.addAtom(lig3D.getAtom(atom))
        confragatomlist.append(atom)             
    anchor = lig3D.getAtomCoords(anchoratom)
    if not checkcolinear(anchor,confrag3D.getAtomCoords(0),confrag3D.getAtomCoords(1)):
        refpt = confrag3D.getAtomCoords(0)
        u = vecdiff(refpt,anchor)
        dtheta = 5
        objs = []
        localmaxs = []
        thetas = range(0,360,dtheta)
        for theta in thetas:
            confrag3D = rotate_around_axis(confrag3D,refpt,u,dtheta)     
            auxmol = mol3D()
            auxmol.addAtom(confrag3D.getAtom(0))
            for at in confrag3D.getBondedAtoms(0):
                auxmol.addAtom(confrag3D.getAtom(at))
            auxmol.addAtom(lig3D.getAtom(anchoratom))  
            objs.append(distance(mcoords,auxmol.centersym()))
        for i,obj in enumerate(objs):
            try:
                if objs[i] > objs[i-1] and objs[i] > objs[i+1]:
                    localmaxs.append(thetas[i])
            except IndexError:
                pass
    # in future, compare multiple local maxima
    confrag3D = rotate_around_axis(confrag3D,refpt,u,localmaxs[0])
    for i,atom in enumerate(confragatomlist):
        lig3D.getAtom(atom).setcoords(confrag3D.getAtomCoords(i))
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned   

def rotate_catoms_fix_Hs(lig3D,catoms,mcoords,core3D):
    # prevent H clashes by rotating connecting atoms
    for i,n in enumerate(catoms):
        if len(lig3D.getHsbyIndex(n)) > 0:
            lig3D = rotate_catom_fix_Hs(lig3D,catoms,i,mcoords,core3D)
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned      

def find_rotate_rotatable_bond(lig3D,catoms):
    # For bidentate ligands, 
    # mcoords: metal coordinates
    # lig3D: ligand mol3D
    # atom0: ligand center of mass placeholder (where is this written?)
    # core3D: mol3D of partially built complex
    # RETURNS
    # lig3D_aligned: mol3D of aligned ligand
    
    bats = list(set(lig3D.getBondedAtomsnotH(catoms[0])) | set(lig3D.getBondedAtomsnotH(catoms[1])))
    rb1 = 1000
    rb2 = 1000
    for ii in range(lig3D.OBMol.NumBonds()):
        bd = lig3D.OBMol.GetBond(ii)
        bst = bd.GetBeginAtomIdx()
        ben = bd.GetEndAtomIdx()
        if bd.IsRotor() and (bst-1 in bats) and (ben-1 in bats):
            print('Rotatable bond found')
            rb1 = bst-1
            rb2 = ben-1
            break
    if (rb1 != 1000) and (rb2 != 1000): # rotatable bond present, execute rotations
        rotfrag3D = mol3D()
        # create submolecule containing atoms to be rotated (the one containing catoms[0] which is aligned first)
        subm1 = lig3D.findsubMol(rb1,rb2)
        subm2 = lig3D.findsubMol(rb2,rb1)
        if catoms[1] in subm1:
            subm = subm1
            anchor = lig3D.getAtomCoords(rb2)
            refpt = lig3D.getAtomCoords(rb1)
        elif catoms[0] in subm1:
            subm = subm2
            anchor = lig3D.getAtomCoords(rb1)
            refpt = lig3D.getAtomCoords(rb2)
        ncoord = 0
        for nii,ii in enumerate(subm):
            rotfrag3D.addAtom(lig3D.getAtom(ii))
            # find coordinating atom in submolecule
            if ii in catoms:
                ncoord = nii
        u = vecdiff(refpt,anchor)
        dtheta = 10
        theta = 0
        thetaopt = 0
        objopt = 1000
        while theta < 360: # minimize distance between connecting atoms
            rotfrag3D = rotate_around_axis(rotfrag3D,anchor,u,dtheta)
            obj = distance(lig3D.getAtomCoords(catoms[1]),rotfrag3D.getAtomCoords(ncoord))
            obj = obj + distance(lig3D.getAtomCoords(catoms[0]),rotfrag3D.getAtomCoords(ncoord))
            if obj < objopt:
                thetaopt = theta
                objopt = obj
            theta = theta + dtheta
        rotfrag3D = rotate_around_axis(rotfrag3D,anchor,u,thetaopt)
        jj = 0
        for ii in subm: # replace coordinates
            lig3D.getAtom(ii).setcoords(rotfrag3D.getAtomCoords(jj))
            jj = jj + 1
    return lig3D
    
def get_MLdist(args,lig3D,atom0,ligand,metal,MLb,i,ANN_flag,ANN_bondl,this_diag,MLbonds):
    # gets target M-L distance from desired source (custom, sum cov rad or ANN)
    # args: argument namespace
    # lig3D: ligand mol3D   
    # atom0: ligand connecting atom
    # ligand: ligand name
    # m3D: metal atom3D
    # MLb: custom M-L bond length (if any)
    # i: index of ligand    
    # ANN_flag: flag determining if ANN is on
    # ANN_bondl: ANN predicted BL
    # this_diag: used for ANN
    # MLbonds: M-L database
    # RETURNS
    # bondl: target M-L distance
            
    # first check for user-specified distances and use them        
    if MLb and MLb[i]:
        print('using user-specified M-L distances')
        if 'c' in MLb[i].lower():
            bondl = metal.rad + lig3D.getAtom(atom0).rad
        else:
            bondl = float(MLb[i])
    else:
    # otherwise, check for exact DB match    
        bondl,exact_match = get_MLdist_database(args,metal,lig3D,atom0,ligand,MLbonds)
        try:
            this_diag.set_dict_bl(bondl)
        except:
            pass
        if not exact_match and ANN_flag:
            # if no exact match found and ANN enabled, use it
            print('no M-L match in DB, using ANN')
            bondl =  ANN_bondl
        elif exact_match:
            print('using exact M-L match from DB')
        else:
            print('Warning: ANN not active and exact M-L match not found in DB, distance may not be accurate')
    return bondl

def get_MLdist_database(args,metal,lig3D,atom0,ligand,MLbonds):
    # loads M-L bond length from database and reports if compound is in DB
    # if completely non-existent, defaults to sum cov rad
    # INPUT
    #   - args: argument namespace
    #   - metal: metal atom3D
    #   - lig3D: ligand mol3D
    #   - atom0: ligand connecting atom
    #   - ligand: name of ligand
    #   - MLbonds: data from database
    # OUTPUT
    #   - bondl: bond length in A
    #   - exact_match: bool, was there an exact match?
    
    # check for roman letters in oxstate
    if args.oxstate: # if defined put oxstate in keys
        if args.oxstate in romans.keys():
            oxs = romans[args.oxstate]
        else:
            oxs = args.oxstate
    else:
        oxs = '-'
    # check for spin multiplicity
    spin = args.spin if args.spin else '-'
    key = []
    key.append((metal.sym,oxs,spin,lig3D.getAtom(atom0).sym,ligand))
    key.append((metal.sym,oxs,spin,lig3D.getAtom(atom0).sym,'-')) # disregard exact ligand
    key.append((metal.sym,'-','-',lig3D.getAtom(atom0).sym,ligand)) # disregard oxstate/spin
    key.append((metal.sym,'-','-',lig3D.getAtom(atom0).sym,'-')) # else just consider bonding atom
    found = False
    exact_match = False
    # search for data
    for kk in key:
        if (kk in MLbonds.keys()): # if exact key in dictionary
            bondl = float(MLbonds[kk])
            found = True
            if (kk == ((metal.sym,oxs,spin,lig3D.getAtom(atom0).sym,ligand))): ## exact match
               exact_match = True
            break
    if not found: # last resort covalent radii
        bondl = metal.rad + lig3D.getAtom(atom0).rad
    if args.debug:
        print('ms default distance is  ' + str(bondl))
    return bondl,exact_match

def get_batoms(args,batslist,ligsused):
    # Get backbone atoms from template
    batoms = batslist[ligsused]
    if len(batoms) < 1 :
        emsg = 'Connecting all ligands is not possible. Check your input!'
        if args.gui:
            qqb = mQDialogWarn('Warning',emsg)
            qqb.setParent(args.gui.wmain)
    return batoms        

def align_dent2_catom2_coarse(args,lig3D,core3D,catoms,r1,r0,m3D,batoms,mcoords):
    # Crude rotations to bring the 2nd connecting atom closer to its ideal location
    # align center of mass to the middle
    r21 = [a-b for a,b in zip(lig3D.getAtom(catoms[1]).coords(),r1)]
    r21n = [a-b for a,b in zip(m3D.getAtom(batoms[1]).coords(),r1)]
    if (norm(r21)*norm(r21n)) > 1e-8:
        theta = 180*arccos(dot(r21,r21n)/(norm(r21)*norm(r21n)))/pi
    else:
        theta = 0.0
    u = cross(r21,r21n)
    lig3Db = mol3D()
    lig3Db.copymol3D(lig3D)
    # rotate around axis and get both images
    lig3D = rotate_around_axis(lig3D,r1,u,theta)
    lig3Db = rotate_around_axis(lig3Db,r1,u,theta-180)
    d1 = distance(lig3D.getAtom(catoms[1]).coords(),m3D.getAtom(batoms[1]).coords())
    d2 = distance(lig3Db.getAtom(catoms[1]).coords(),m3D.getAtom(batoms[1]).coords())
    lig3D = lig3D if (d1 < d2)  else lig3Db # pick best one
    # flip if overlap
    r0l = lig3D.getAtom(catoms[0]).coords()
    r1l = lig3D.getAtom(catoms[1]).coords()
    md = min(distance(r0l,mcoords),distance(r1l,mcoords))
    if lig3D.mindist(core3D) < md:
        lig3D = rotate_around_axis(lig3D,r0l,vecdiff(r1l,r0l),180.0)
    # correct plane
    r0b = m3D.getAtom(batoms[0]).coords()
    r1b = m3D.getAtom(batoms[1]).coords()
    r0l = lig3D.getAtom(catoms[0]).coords()
    r1l = lig3D.getAtom(catoms[1]).coords()
    rm = lig3D.centermass()
    urot = vecdiff(r1l,r0l)
    theta,ub = rotation_params(mcoords,r0b,r1b)
    theta,ul = rotation_params(rm,r0l,r1l)
    if (norm(ub)*norm(ul)) > 1e-8:
        theta = 180*arccos(dot(ub,ul)/(norm(ub)*norm(ul)))/pi-180.0
    else:
        theta = 0.0
    # rotate around axis
    lig3Db = mol3D()       
    lig3Db.copymol3D(lig3D)
    lig3D = rotate_around_axis(lig3D,r1,urot,theta)
    lig3Db = rotate_around_axis(lig3Db,r1,urot,-theta)
    # select best
    #rm0,rm1 = lig3D.centermass(),lig3Db.centermass()
    #theta,ul0 = rotation_params(rm0,r0l,r1l)
    #theta,ul1 = rotation_params(rm1,r0l,r1l)
    #th0 = 180*arccos(dot(ub,ul0)/(norm(ub)*norm(ul0)))/pi
    #th0 = min(abs(th0),abs(180-th0))
    #th1 = 180*arccos(dot(ub,ul1)/(norm(ub)*norm(ul1)))/pi
    #th1 = min(abs(th1),abs(180-th1))
    #lig3D = lig3D if th0 < th1 else lig3Db
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned,r1b     

def align_dent2_catom2_refined(args,lig3D,catoms,bondl,r1,r0,core3D,rtarget,coreref,MLoptbds):
    # Aligns second connecting atom of a bidentate ligand to balance ligand strain and the desired coordination environment.
    # compute starting ligand FF energy for later comparison
    corerefcoords = coreref.coords()
    dr = vecdiff(rtarget,lig3D.getAtom(catoms[1]).coords())
    cutoff = 5 # energy threshold for ligand strain, kcal/mol
    lig3Dtmp = mol3D()
    lig3Dtmp.copymol3D(lig3D)
    lig3Dtmp,en_start = ffopt(args.ff,lig3Dtmp,[],1,[],False,[],200)
    # take steps between current ligand position and ideal position on backbone
    nsteps = 20
    ddr = [di/nsteps for di in dr]
    ens=[]
    finished = False
    relax = False
    while True: 
        lig3Dtmp = mol3D()
        lig3Dtmp.copymol3D(lig3D)
        for ii in range(0,nsteps):
            lig3Dtmp,enl = ffopt(args.ff,lig3Dtmp,[],1,[catoms[0],catoms[1]],False,[],'Adaptive')
            ens.append(enl)
            lig3Dtmp.getAtom(catoms[1]).translate(ddr)
            # once the ligand strain energy becomes too high, stop and accept ligand position 
            # or if the ideal coordinating point is reached without reaching the strain energy cutoff, stop
            if (ens[-1] - ens[0] > cutoff) or (ii == nsteps-1):
                r0,r1 = lig3Dtmp.getAtomCoords(catoms[0]),lig3Dtmp.getAtomCoords(catoms[1])
                r01 = distance(r0,r1)
                try:
                # but if ligand still cannot be aligned, instead force alignment with a huge cutoff and then relax later
                    theta1 = 180*arccos(0.5*r01/bondl)/pi
                except:
                    print('Forcing alignment...')
                    cutoff += 5000000
                    relax = True
                    break
                theta2 = vecangle(vecdiff(r1,r0),vecdiff(corerefcoords,r0))
                dtheta = theta2-theta1
                theta,urot = rotation_params(corerefcoords,r0,r1)  
                lig3Dtmp = rotate_around_axis(lig3Dtmp,r0,urot,-dtheta) # rotate so that it matches bond
                finished = True 
                break
        if finished:
            break
    # for long linear ligand chains, this procedure might produce the wrong ligand curvature. If so, reflect about M-L plane        
    lig3Dtmpb = mol3D()
    lig3Dtmpb.copymol3D(lig3Dtmp)
    lig3Dtmpb = reflect_through_plane(lig3Dtmpb,vecdiff(midpt(lig3Dtmpb.getAtom(catoms[0]).coords(),lig3Dtmpb.getAtom(catoms[1]).coords()),corerefcoords),lig3Dtmpb.getAtom(catoms[0]).coords())
    lig3Dtmp = lig3Dtmpb if lig3Dtmp.mindist(core3D) < lig3Dtmpb.mindist(core3D) else lig3Dtmp
    if relax:
        # Relax the ligand
        lig3Dtmp,enl = ffopt(args.ff,lig3Dtmp,[catoms[1]],2,[catoms[0]],False,MLoptbds[-2:-1],200) 
        lig3Dtmp.deleteatom(lig3Dtmp.natoms-1) 
    lig3Dtmp,en_final = ffopt(args.ff,lig3Dtmp,[],1,[],False,[],0)
    if en_final - en_start > 20:
        print 'Warning: Complex may be strained. Change in ligand MM energy (kcal/mol) = ' + str(en_final - en_start)    
        print 'Consider using our conformer search mode (to be implemented in a future release)'
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3Dtmp)
    return lig3D_aligned                    
   
def align_dent1_lig(args,cpoint,core3D,coreref,ligand,lig3D,catoms,rempi=False,ligpiatoms=[],MLb=[],ANN_flag=False,ANN_bondl=[],this_diag=0,MLbonds=dict(),MLoptbds=[],i=0,EnableAutoLinearBend=True):
    # Aligns a monodentate ligand to core connecting atom coordinates.
    # args: namespace of arguments
    # cpoint: atom3D containing backbone connecting point
    # core3D: mol3D of partially built complex
    # coreref: core reference atom (atom3D), for a single metal atom this would be the center (index 0)
    # ligand: name of ligand
    # lig3D: ligand mol3D
    # catoms: ligand connecting atoms
    # rempi: flag determining if ligand is a pi-bonding ligand
    # ligpiatoms: pi-coordinating atoms in ligand (default 0 if rempi false)
    # MLb: custom M-L bond length (if any)
    # ANN_flag: flag determining if ANN is on
    # ANN_bondl: ANN-predicted bond length
    # this_diag: used for ANN
    # MLbonds: M-L bond database
    # MLoptbds: list of final M-L bond lengths
    # i: index of ligand
    # RETURNS
    # lig3D_aligned: mol3D of aligned ligand
    
    corerefcoords = coreref.coords()

    # connection atom in lig3D
    atom0 = catoms[0]
    # translate ligand to overlap with backbone connecting point
    lig3D.alignmol(lig3D.getAtom(atom0),cpoint)
    # determine bond length (database/cov rad/ANN)
    bondl = get_MLdist(args,lig3D,atom0,ligand,coreref,MLb,i,ANN_flag,ANN_bondl,this_diag,MLbonds)
    MLoptbds.append(bondl)
    # align ligand to correct M-L distance
    u = vecdiff(cpoint.coords(),corerefcoords)
    lig3D = aligntoaxis2(lig3D, cpoint.coords(), corerefcoords, u, bondl)
    if rempi and len(ligpiatoms) == 2:
        # align linear (non-arom.) pi-coordinating ligand
        lig3D = align_linear_pi_lig(corerefcoords,lig3D,atom0,ligpiatoms)
    elif lig3D.natoms > 1:
        # align ligand center of symmetry
        lig3D = align_lig_centersym(corerefcoords,lig3D,atom0,core3D,EnableAutoLinearBend)
        if lig3D.natoms > 2:
            # check for linear molecule and align
            lig3D = check_rotate_linear_lig(corerefcoords,lig3D,atom0)
            # check for symmetric molecule
            lig3D = check_rotate_symm_lig(corerefcoords,lig3D,atom0,core3D)
        # rotate around M-L axis to minimize steric repulsion
        lig3D = rotate_MLaxis_minimize_steric(corerefcoords,lig3D,atom0,core3D)

    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned,MLoptbds

def align_dent2_lig(args,cpoint,batoms,m3D,core3D,coreref,ligand,lig3D,catoms,MLb,ANN_flag,ANN_bondl,this_diag,MLbonds,MLoptbds,frozenats,i):
    # Aligns a bidentate ligand to core connecting atom coordinates.
    corerefcoords = coreref.coords()    
    r0 = corerefcoords
    # get cis conformer by rotating rotatable bonds
    lig3D = find_rotate_rotatable_bond(lig3D,catoms)
    # connection atom
    atom0 = catoms[0]
    # translate ligand to match first connecting atom to backbone connecting point
    lig3D.alignmol(lig3D.getAtom(atom0),cpoint)
    r1 = lig3D.getAtom(atom0).coords()
    # Crude rotations to bring the 2nd connecting atom closer to its ideal location
    lig3D,r1b = align_dent2_catom2_coarse(args,lig3D,core3D,catoms,r1,r0,m3D,batoms,corerefcoords)  
    ## get bond length
    bondl = get_MLdist(args,lig3D,atom0,ligand,coreref,MLb,i,ANN_flag,ANN_bondl,this_diag,MLbonds)
    MLoptbds.append(bondl)
    MLoptbds.append(bondl)
    lig3D = setPdistance(lig3D, r1, r0, bondl)    
    # get target point for 2nd connecting atom
    rtarget = getPointu(corerefcoords, bondl, vecdiff(r1b,corerefcoords)) # get second point target
    if args.ff and 'B' in lig3D.ffopt:
        # align 2nd connecting atom while balancing the desired location and ligand strain
        lig3D = align_dent2_catom2_refined(args,lig3D,catoms,bondl,r1,r0,core3D,rtarget,coreref,MLoptbds)
    else:
        print 'Warning: Ligand FF optimization is inactive.'  
    # rotate connecting atoms to align Hs properly
    lig3D = rotate_catoms_fix_Hs(lig3D,catoms,corerefcoords,core3D)
    # freeze local geometry
    lats = lig3D.getBondedAtoms(catoms[0])+lig3D.getBondedAtoms(catoms[1])
    for lat in list(set(lats)):
        frozenats.append(lat+core3D.natoms)
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned,frozenats,MLoptbds

def align_dent3_lig(args,batslist,ligsused,m3D,core3D,ligand,lig3D,catoms,MLb,ANN_flag,ANN_bondl,this_diag,MLbonds,MLoptbds,frozenats,i):
    # Aligns a tridentate ligand to core connecting atom coordinates. EXPERIMENTAL
    # get metal identity and location
    metal = core3D.getAtom(0).sym
    mcoords = core3D.getAtom(0).coords()    
    r0 = mcoords
    # get cis conformer by rotating rotatable bonds
    lig3D = find_rotate_rotatable_bond(lig3D,catoms)
    # connection atoms in backbone
    batoms = get_batoms(args,batslist,ligsused)
    # connection atom
    atom0 = catoms[0]
    # translate ligand to match first connecting atom to backbone connecting point
    lig3D.alignmol(lig3D.getAtom(atom0),m3D.getAtom(batoms[0]))
    r1 = lig3D.getAtom(atom0).coords()
    # Crude rotations to bring the 2nd connecting atom closer to its ideal location
    lig3D,r1b = align_dent2_catom2_coarse(args,lig3D,core3D,catoms,r1,r0,m3D,batoms,mcoords)
    ## get bond length
    bondl = get_MLdist(args,lig3D,atom0,ligand,m3D.getAtom(0),MLb,i,ANN_flag,ANN_bondl,this_diag,MLbonds)
    MLoptbds.append(bondl)
    MLoptbds.append(bondl)
    lig3D = setPdistance(lig3D, r1, r0, bondl)
    # get target point for 2nd connecting atom
    rtarget = getPointu(mcoords, bondl, vecdiff(r1b,mcoords)) # get second point target
    if not args.ffoption == 'no':
    # align 2nd connecting atom while balancing the desired location and ligand strain
        lig3D = align_dent2_catom2_refined(args,lig3D,catoms,bondl,r1,r0,core3D,rtarget,mcoords,MLoptbds)
    else:
        print 'Warning: You have disabled ligand FF optimization. This may result in poor structures because our routines rely on the FF to get the correct conformer.'    
    # rotate connecting atoms to align Hs properly
    lig3D = rotate_catoms_fix_Hs(lig3D,catoms,mcoords,core3D)
    # freeze local geometry
    lats = lig3D.getBondedAtoms(catoms[0])+lig3D.getBondedAtoms(catoms[1])
    for lat in list(set(lats)):
        frozenats.append(lat+core3D.natoms)
    lig3D_aligned = mol3D()
    lig3D_aligned.copymol3D(lig3D)
    return lig3D_aligned,frozenats,MLoptbds

#################################################
####### functionalizes core with ligands ########
############## for metal complexes ##############
#################################################
def mcomplex(args,ligs,ligoc,licores,globs):
    # INPUT
    #   - args: placeholder for input arguments
    #   - ligs: list of ligands
    #   - ligoc: list of ligand occupations
    #   - licores: dictionary with ligands
    #   - globs: class with global variables
    # OUTPUT
    #   - core3D: built complex
    #   - complex3D: list of all mol3D ligands and core
    #   - emsg: error messages
    this_diag = run_diag()
    if globs.debug:
        print '\nGenerating complex with ligands and occupations:',ligs,ligoc
    if args.gui:
        args.gui.iWtxt.setText('\nGenerating complex with core:'+args.core+' and ligands: '+ ' '.join(ligs)+'\n'+args.gui.iWtxt.toPlainText())
        args.gui.app.processEvents()
    # import gui options
    if args.gui:
        from Classes.mWidgets import mQDialogWarn
    ### initialize variables ###
    emsg, complex3D = False, []
    occs0 = []      # occurrences of each ligand
    toccs = 0       # total occurrence count (number of ligands)
    catsmi = []     # SMILES ligands connection atoms
    smilesligs = 0  # count how many smiles strings
    cats0 = []      # connection atoms for ligands
    dentl = []      # denticity of ligands
    connected = []  # indices in core3D of ligand atoms connected to metal
    frozenats = []  # atoms to be frozen in optimization
    freezeangles = False # custom angles imposed
    MLoptbds = []   # list of bond lengths
    rempi = False   # remove dummy pi orbital center of mass atom
    ### load bond data ###
    MLbonds = loaddata('/Data/ML.dat')
    ### calculate occurrences, denticities etc for all ligands ###
    for i,ligname in enumerate(ligs):
        # if not in cores -> smiles/file
        if ligname not in licores.keys():
            if args.smicat and len(args.smicat)>= (smilesligs+1):
                if 'pi' in args.smicat[smilesligs]:
                    cats0.append(['c'])
                else:
                    cats0.append(args.smicat[smilesligs])
            else:
                cats0.append([1])
            dent_i = len(cats0[-1])
            smilesligs += 1
        else:
            cats0.append(False)
        # otherwise get denticity from ligands dictionary
            if 'pi' in licores[ligname][2]:
                dent_i = 1
            else:
                if isinstance(licores[ligname][2], (str, unicode)):
                    dent_i = 1
                else:
                    dent_i = int(len(licores[ligname][2]))
        # get occurrence for each ligand if specified (default 1)
        oc_i = int(ligoc[i]) if i < len(ligoc) else 1
        occs0.append(0)         # initialize occurrences list
        dentl.append(dent_i)    # append denticity to list
        # loop over occurrence of ligand i to check for max coordination
        for j in range(0,oc_i):
            occs0[i] += 1
            toccs += dent_i
    ### sort by descending denticity (needed for adjacent connection atoms) ###
    ligandsU,occsU,dentsU = ligs,occs0,dentl # save unordered lists
    indcs = smartreorderligs(args,ligs,dentl,licores)
    ligands = [ligs[i] for i in indcs]  # sort ligands list
    occs = [occs0[i] for i in indcs]    # sort occurrences list
    tcats = [cats0[i] for i in indcs]   # sort connections list
    dents = [dentl[i] for i in indcs]   # sort denticities list
    ### if using decorations, make repeatable list
    if args.decoration:
        if not args.decoration_index:
            print('Warning, no deocoration index given, assuming first ligand')
            args.decoration_index = [[0]]
        if len(args.decoration_index) != len(ligs):
            new_decoration_index =  []
            new_decorations = []
            for i in range(0,len(ligs)):
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
            args.decoration_index =  new_decoration_index
        args.decoration_index = [args.decoration_index[i] for i in indcs]   # sort decorations list
        args.decoration = [args.decoration[i] for i in indcs]   # sort decorations list
    # sort keepHs list and unpack into list of tuples representing each connecting atom###
    keepHs = [k for k in args.keepHs]
    keepHs = [keepHs[i] for i in indcs]
    for i,keepH in enumerate(keepHs):
        keepHs[i] = [keepHs[i]] * dents[i]
    ### sort M-L bond list ###
    MLb = False
    if args.MLbonds:
        MLb = [k for k in args.MLbonds]
        for j in range(len(args.MLbonds),len(ligs)):
            MLb.append(False)
        MLb = [MLb[i] for i in indcs] # sort MLbonds list
    ### sort ligands custom angles ###
    pangles = False
    if args.pangles:
        pangles = []
        for j in range(len(args.pangles),len(ligs)):
            pangles.append(False)
        pangles = [args.pangles[i] for i in indcs] # sort custom langles list

    # compute number of connecting points required
    cpoints_required = 0
    for i,ligand in enumerate(ligands):
        for j in range(0,occs[i]):
            cpoints_required += dents[i]

    # load core and initialize template
    m3D,core3D,geom,backbatoms,coord,corerefatoms = init_template(args,cpoints_required)
    #########################################################
    ####### Get connection points for all the ligands #######
    ########### smart alignment and forced order ############
    batslist = []
    #if geom:        
    if args.ligloc and args.ligalign:
        batslist0 = []
        for i,ligand in enumerate(ligandsU):
            for j in range(0,occsU[i]):
                # get correct atoms
                bats,backbatoms = getnupdateb(backbatoms,dentsU[i])
                batslist0.append(bats)
        # reorder according to smart reorder
        for i in indcs:
            offset = 0
            for ii in range(0,i):
                    offset += (occsU[ii]-1)
            for j in range(0,occsU[i]):
                batslist.append(batslist0[i+j+offset])# sort connections list
    else:
        for i,ligand in enumerate(ligands):
            for j in range(0,occs[i]):
                # get correct atoms
                bats,backbatoms = getnupdateb(backbatoms,dents[i])
                batslist.append(bats)
    if not geom:
        for comb in batslist:
            for i in comb:
                if i == 1:
                    batslist[comb][i] = m3D.natoms - coord + 1            
    # initialize ANN
    ANN_flag,ANN_bondl,ANN_reason,ANN_attributes = init_ANN(args,ligands,occs,dents,batslist,tcats,licores)
    this_diag.set_ANN(ANN_flag,ANN_reason,ANN_attributes)
    
    # freeze core
    for i in range(0,core3D.natoms):
        frozenats.append(i) 

    ###############################
    #### loop over ligands and ####
    ### begin functionalization ###
    ###############################
    # loop over ligands
    totlig = 0  # total number of ligands added
    ligsused = 0
    for i,ligand in enumerate(ligands):
        for j in range(0,occs[i]):
            denticity = dents[i]
        
            if not(ligand=='x' or ligand =='X') and (totlig-1+denticity < coord):

                # load ligand
                lig,emsg = lig_load(ligand) # load ligand
                lig.convert2mol3D()
                if emsg:
                    return False,emsg
                lig3D,rempi,ligpiatoms = init_ligand(args,lig,tcats,keepHs,i)

                ### add atoms to connected atoms list
                catoms = lig.cat # connection atoms
                initatoms = core3D.natoms # initial number of atoms in core3D
                for at in catoms:
                    connected.append(initatoms+at)
                ### initialize variables
                mcoords = core3D.getAtom(0).coords() # metal coordinates in backbone
                atom0, r0, r1, r2, r3 = 0, mcoords, 0, 0, 0 # initialize variables
                coreref = corerefatoms.getAtom(totlig)
                # connecting point in backbone to align ligand to
                batoms = get_batoms(args,batslist,ligsused)
                cpoint = m3D.getAtom(batoms[0])
                ####################################################
                ##    attach ligand depending on the denticity    ##
                ## optimize geometry by minimizing steric effects ##
                ####################################################
                if (denticity == 1):
                    lig3D,MLoptbds = align_dent1_lig(args,cpoint,core3D,coreref,ligand,lig3D,catoms,rempi,ligpiatoms,MLb,ANN_flag,ANN_bondl,this_diag,MLbonds,MLoptbds,i)
                elif (denticity == 2):
                    lig3D,frozenats,MLoptbds = align_dent2_lig(args,cpoint,batoms,m3D,core3D,coreref,ligand,lig3D,catoms,MLb,ANN_flag,ANN_bondl,this_diag,MLbonds,MLoptbds,frozenats,i)
                elif (denticity == 3):
                    # connection atoms in backbone
                    batoms = get_batoms(args,batslist,ligsused)
                    # connection atom
                    atom0 = catoms[1]
                    ### align molecule according to connection atom and shadow atom ###
                    lig3D.alignmol(lig3D.getAtom(atom0),m3D.getAtom(batoms[1]))
                    # align with correct plane
                    rl0,rl1,rl2 = lig3D.getAtom(catoms[0]).coords(),lig3D.getAtom(catoms[1]).coords(),lig3D.getAtom(catoms[2]).coords()
                    rc0,rc1,rc2 = m3D.getAtom(batoms[0]).coords(),m3D.getAtom(batoms[1]).coords(),m3D.getAtom(batoms[2]).coords()
                    theta0,ul = rotation_params(rl0,rl1,rl2)
                    theta1,uc = rotation_params(rc0,rc1,rc2)
                    urot = vecdiff(rl1,mcoords)
                    theta = vecangle(ul,uc)
                    ### rotate around primary axis ###
                    lig3Db = mol3D()
                    lig3Db.copymol3D(lig3D)
                    lig3D = rotate_around_axis(lig3D,rl1,urot,theta)
                    lig3Db = rotate_around_axis(lig3Db,rl1,urot,180-theta)
                    rl0,rl1,rl2 = lig3D.getAtom(catoms[0]).coords(),lig3D.getAtom(catoms[1]).coords(),lig3D.getAtom(catoms[2]).coords()
                    rl0b,rl1b,rl2b = lig3Db.getAtom(catoms[0]).coords(),lig3Db.getAtom(catoms[1]).coords(),lig3Db.getAtom(catoms[2]).coords()
                    rc0,rc1,rc2 = m3D.getAtom(batoms[0]).coords(),m3D.getAtom(batoms[1]).coords(),m3D.getAtom(batoms[2]).coords()
                    theta,ul = rotation_params(rl0,rl1,rl2)
                    theta,ulb = rotation_params(rl0b,rl1b,rl2b)
                    theta,uc = rotation_params(rc0,rc1,rc2)
                    d1 = norm(cross(ul,uc))
                    d2 = norm(cross(ulb,uc))
                    lig3D = lig3D if (d1 < d2)  else lig3Db # pick best one
                    ### rotate around secondary axis ###
                    auxm = mol3D()
                    auxm.addAtom(lig3D.getAtom(catoms[0]))
                    auxm.addAtom(lig3D.getAtom(catoms[2]))
                    theta,urot0 = rotation_params(core3D.getAtom(0).coords(),lig3D.getAtom(atom0).coords(),auxm.centermass())
                    theta0,urot = rotation_params(lig3D.getAtom(catoms[0]).coords(),lig3D.getAtom(catoms[1]).coords(),lig3D.getAtom(catoms[2]).coords())
                    # change angle if > 90
                    if theta > 90:
                        theta -= 180
                    lig3Db = mol3D()
                    lig3Db.copymol3D(lig3D)
                    lig3D = rotate_around_axis(lig3D,lig3D.getAtom(atom0).coords(),urot,theta)
                    lig3Db = rotate_around_axis(lig3Db,lig3D.getAtom(atom0).coords(),urot,180-theta)
                    d1 = distance(lig3D.getAtom(catoms[0]).coords(),m3D.getAtom(batoms[0]).coords())
                    d2 = distance(lig3Db.getAtom(catoms[0]).coords(),m3D.getAtom(batoms[0]).coords())
                    lig3D = lig3D if (d1 < d2) else lig3Db
                    # correct if not symmetric
                    theta0,urotaux = rotation_params(lig3D.getAtom(catoms[0]).coords(),lig3D.getAtom(catoms[1]).coords(),core3D.getAtom(0).coords())
                    theta1,urotaux = rotation_params(lig3D.getAtom(catoms[2]).coords(),lig3D.getAtom(catoms[1]).coords(),core3D.getAtom(0).coords())
                    dtheta = 0.5*(theta1-theta0)
                    if abs(dtheta) > 0.5:
                        lig3D = rotate_around_axis(lig3D,lig3D.getAtom(atom0).coords(),urot,dtheta)
                    # flip to align 3rd atom if wrong
                    urot = vecdiff(lig3D.getAtom(catoms[0]).coords(),lig3D.getAtom(catoms[1]).coords())
                    lig3Db = mol3D()
                    lig3Db.copymol3D(lig3D)
                    lig3Db = rotate_around_axis(lig3Db,rc1,urot,180)
                    d1 = distance(lig3D.getAtom(catoms[2]).coords(),m3D.getAtom(batoms[2]).coords())
                    d2 = distance(lig3Db.getAtom(catoms[2]).coords(),m3D.getAtom(batoms[2]).coords())
                    lig3D = lig3D if (d1 < d2)  else lig3Db # pick best one
                    # if overlap flip
                    dm0 = distance(lig3D.getAtom(catoms[0]).coords(),m3D.getAtom(0).coords())
                    dm1 = distance(lig3D.getAtom(catoms[1]).coords(),m3D.getAtom(0).coords())
                    dm2 = distance(lig3D.getAtom(catoms[2]).coords(),m3D.getAtom(0).coords())
                    mind = min([dm0,dm1,dm2])
                    for iiat,atom in enumerate(lig3D.atoms):
                        if iiat not in catoms and distance(atom.coords(),m3D.getAtom(0).coords()) < min([dm0,dm1,dm2]):
                            lig3D = rotate_around_axis(lig3D,rc1,uc,180)
                            break
                    bondl = get_MLdist(args,lig3D,atom0,ligand,m3D.getAtom(0),MLb,i,ANN_flag,ANN_bondl,this_diag,MLbonds)            
                    for iib in range(0,3):
                        MLoptbds.append(bondl)
                    # set correct distance
                    setPdistance(lig3D, lig3D.getAtom(atom0).coords(), m3D.getAtom(0).coords(), bondl)
                elif (denticity == 4):
                    # connection atoms in backbone
                    batoms = batslist[ligsused]
                    if len(batoms) < 1 :
                        if args.gui:
                            emsg = 'Connecting all ligands is not possible. Check your input!'
                            qqb = mQDialogWarn('Warning',emsg)
                            qqb.setParent(args.gui.wmain)
                        break
                    # connection atom
                    atom0 = catoms[0]
                    # align molecule according to symmetry center
                    auxmol = mol3D()
                    for iiax in range(0,4):
                        auxmol.addAtom(lig3D.getAtom(catoms[iiax]))
                    if args.debug:
                        m3D.writexyz('m3d.xyz')
                        auxmol.writexyz('auxmol.xyz')

                    lig3D.alignmol(atom3D('C',auxmol.centermass()),m3D.getAtom(0))
                    # align plane
                    r0c = m3D.getAtom(batoms[0]).coords()
                    r1c = m3D.getAtom(batoms[1]).coords()
                    r2c = m3D.getAtom(batoms[2]).coords()
                    r0l = lig3D.getAtom(catoms[0]).coords()
                    r1l = lig3D.getAtom(catoms[1]).coords()
                    r2l = lig3D.getAtom(catoms[2]).coords()
                    theta,uc = rotation_params(r0c,r1c,r2c) # normal vector to backbone plane
                    theta,ul = rotation_params(r0l,r1l,r2l) # normal vector to ligand plane
                    lig3Db = mol3D()
                    lig3Db.copymol3D(lig3D)
                    theta = 180*arccos(dot(uc,ul)/(norm(uc)*norm(ul)))/pi
                    u = cross(uc,ul)
                    # rotate around axis to match planes
                    theta = 180-theta if theta > 90 else theta
                    lig3D = rotate_around_axis(lig3D,r0l,u,theta)
                    # rotate ar?ound secondary axis to match atoms
                    r0l = lig3D.getAtom(catoms[0]).coords()
                    r1l = lig3D.getAtom(catoms[1]).coords()
                    r2l = lig3D.getAtom(catoms[2]).coords()
                    theta0,ul = rotation_params(r0l,r1l,r2l) # normal vector to ligand plane
                    rm = lig3D.centermass()
                    r1 = vecdiff(r0l,mcoords)
                    r2 = vecdiff(r0c,mcoords)
                    theta = 180*arccos(dot(r1,r2)/(norm(r1)*norm(r2)))/pi
                    lig3Db = mol3D()
                    lig3Db.copymol3D(lig3D)
                    if args.debug:
                        print('normal to tetradentate ligand plane: ',ul)
                        print('lig center of mass ',rm)
                        lig3D.writexyz('lig3d.xyz')
                        lig3Db.writexyz('lig3db.xyz')
                    # rotate around axis and get both images
                    lig3D = rotate_around_axis(lig3D,mcoords,ul,theta)
                    # get distance from bonds table or vdw radii
                    #if MLb and MLb[i]:
                        #if 'c' in MLb[i].lower():
                            #bondl = m3D.getAtom(0).rad + lig3D.getAtom(atom0).rad
                        #else:
                            #bondl = float(MLb[i]) # check for custom
                    #else:
                        #if not ANN_flag:
                            #bondl = getbondlength(args,metal,core3D,lig3D,0,atom0,ligand,MLbonds)
                            #this_diag.set_dict_bl(bondl)
                        #else:
                            #bondl,exact_match = getbondlengthStrict(args,metal,core3D,lig3D,0,atom0,ligand,MLbonds)
                            #this_diag.set_dict_bl(bondl)
                            #if not exact_match :
                                #if args.debug:
                                        #print('No match in DB, using ANN')
                                #bondl =  ANN_bondl
                            #else:
                                #if args.debug:
                                    #print('using exact match from DB at  ' + num2str(bondl))
                                #db_overwrite = True
                    bondl = get_MLdist(args,lig3D,atom0,ligand,m3D.getAtom(0),MLb,i,ANN_flag,ANN_bondl,this_diag,MLbonds)
                    for iib in range(0,4):
                        MLoptbds.append(bondl)
                elif (denticity == 5):
                    # connection atoms in backbone
                    batoms = batslist[ligsused]
                    if len(batoms) < 1 :
                        if args.gui:
                            qqb = mQDialogWarn('Warning',emsg)
                            qqb.setParent(args.gui.wmain)
                        emsg = 'Connecting all ligands is not possible. Check your input!'
                        break
                    # get center of mass
                    ligc = mol3D()
                    for i in range(0,4): #5 is the non-planar atom
                        ligc.addAtom(lig3D.getAtom(catoms[i]))
                    # translate ligand to the middle of octahedral
                    lig3D.translate(vecdiff(mcoords,ligc.centermass()))
                    # get plane
                    r0c = m3D.getAtom(batoms[0]).coords()
                    r2c = m3D.getAtom(batoms[1]).coords()
                    r1c = mcoords
                    r0l = lig3D.getAtom(catoms[0]).coords()
                    r2l = lig3D.getAtom(catoms[1]).coords()
                    r1l = mcoords
                    theta,uc = rotation_params(r0c,r1c,r2c) # normal vector to backbone plane
                    theta,ul = rotation_params(r0l,r1l,r2l) # normal vector to ligand plane
                    theta = vecangle(uc,ul)
                    u = cross(uc,ul)
                    lig3Db = mol3D()
                    lig3Db.copymol3D(lig3D)
                    # rotate around axis to match planes
                    lig3D = rotate_around_axis(lig3D,mcoords,u,theta)
                    lig3Db = rotate_around_axis(lig3Db,mcoords,u,180+theta)
                    d1 = distance(lig3D.getAtom(catoms[4]).coords(),m3D.getAtom(batoms[-1]).coords())
                    d2 = distance(lig3Db.getAtom(catoms[4]).coords(),m3D.getAtom(batoms[-1]).coords())
                    lig3D = lig3D if (d2 < d1)  else lig3Db # pick best one
                    # rotate around center axis to match backbone atoms
                    r0l = vecdiff(lig3D.getAtom(catoms[0]).coords(),mcoords)
                    r1l = vecdiff(m3D.getAtom(totlig+1).coords(),mcoords)
                    u = cross(r0l,r1l)
                    theta = 180*arccos(dot(r0l,r1l)/(norm(r0l)*norm(r1l)))/pi
                    lig3Db = mol3D()
                    lig3Db.copymol3D(lig3D)
                    lig3D = rotate_around_axis(lig3D,mcoords,u,theta)
                    lig3Db = rotate_around_axis(lig3Db,mcoords,u,theta-90)
                    d1 = distance(lig3D.getAtom(catoms[0]).coords(),m3D.getAtom(batoms[0]).coords())
                    d2 = distance(lig3Db.getAtom(catoms[0]).coords(),m3D.getAtom(batoms[0]).coords())
                    lig3D = lig3D if (d1 < d2)  else lig3Db # pick best one
                    bondl,exact_match = get_MLdist_database(args,core3D.getAtom(0),lig3D,catoms[0],ligand,MLbonds)
                    # flip if necessary
                    if len(batslist) > ligsused:
                        nextatbats = batslist[ligsused]
                    auxm = mol3D()
                    if len(nextatbats) > 0:
                        for at in nextatbats:
                            auxm.addAtom(m3D.getAtom(at))
                        if lig3D.overlapcheck(auxm,True): # if overlap flip
                            urot = vecdiff(m3D.getAtomCoords(batoms[1]),m3D.getAtomCoords(batoms[0]))
                            lig3D = rotate_around_axis(lig3D,mcoords,urot,180)
                    for iib in range(0,5):
                        MLoptbds.append(bondl)
                elif (denticity == 6):
                    # connection atoms in backbone
                    batoms = batslist[ligsused]
                    if len(batoms) < 1 :
                        if args.gui:
                            qqb = mQDialogWarn('Warning',emsg)
                            qqb.setParent(args.gui.wmain)
                        emsg = 'Connecting all ligands is not possible. Check your input!'
                        break
                    # get center of mass
                    ligc = mol3D()
                    for i in range(0,6):
                        ligc.addAtom(lig3D.getAtom(catoms[i]))
                    # translate metal to the middle of octahedral
                    core3D.translate(vecdiff(ligc.centermass(),mcoords))
                    bondl,exact_match = get_MLdist_database(args,core3D.getAtom(0),lig3D,catoms[0],ligand,MLbonds)
                    for iib in range(0,6):
                        MLoptbds.append(bondl)
                auxm = mol3D()
                auxm.copymol3D(lig3D)
                complex3D.append(auxm)
                if 'a' not in lig.ffopt.lower():
                    for latdix in range(0,lig3D.natoms):
                        frozenats.append(latdix+core3D.natoms)
                # combine molecules
                core3D = core3D.combine(lig3D)
                # remove dummy cm atom if requested
                if rempi:
                    core3D.deleteatom(core3D.natoms-1)
                if args.calccharge:
                    core3D.charge += lig3D.charge
                # perform FF optimization if requested
                if 'a' in args.ffoption:
                    print('FF optimizing molecule after placing ligand')
                    core3D,enc = ffopt(args.ff,core3D,connected,1,frozenats,freezeangles,MLoptbds,'Adaptive')
            totlig += denticity
            ligsused += 1
    # perform FF optimization if requested
    if 'a' in args.ffoption:
        print('Performing final FF opt')
        core3D,enc = ffopt(args.ff,core3D,connected,1,frozenats,freezeangles,MLoptbds,'Adaptive')
    ###############################

    return core3D,complex3D,emsg,this_diag

##########################################
### main structure generation function ###
##########################################
def structgen(args,rootdir,ligands,ligoc,globs,sernum):

    # INPUT
    #   - args: placeholder for input arguments
    #   - rootdir: directory of current run
    #   - ligands: list of ligands
    #   - ligoc: list of ligand occupations
    #   - globs: class with global variables
    # OUTPUT
    #   - strfiles: list of xyz files generated
    #   - emsg: error messages
    emsg = False
    # import gui options
    if args.gui:
        from Classes.mWidgets import mQDialogWarn
    # load ligand dictionary
    licores = getlicores()
    strfiles = []
    ########## START FUNCTIONALIZING ##########

    sanity = False
    this_diag = run_diag()
    # check if ligands specified for functionalization
    if (ligands):
        core3D,complex3D,emsg,this_diag = mcomplex(args,ligands,ligoc,licores,globs)
        name_core = args.core
        if emsg:
            return False,emsg
    else:
        print('You specified no ligands. Returning the core.')
        core3D = mol3D()
        name_core = core3D
    ############ END FUNCTIONALIZING ###########
    # generate multiple geometric arrangements
    Nogeom = int(args.bindnum) if args.bindnum and args.bind else 1 # number of different combinations
    ligname = '' # name of file
    nosmiles = 0
    # generate name of the file
    for l in ligands:
        if l not in licores.keys():
            if '.xyz' in l or '.mol' in l:
                l = l.split('.')[-1]
                l = l.rsplit('/')[-1]
            else:
                if args.sminame:
                    if globs.nosmiles > 1:
                        ismidx = nosmiles
                    else:
                        ismidx = 0
                    if len(args.sminame) > ismidx:
                        l = args.sminame[ismidx][0:2]
                    else:
                        l = l = 'smi'+str(nosmiles)
                else:
                    l = 'smi'+str(nosmiles)
                nosmiles += 1
        ligname += ''.join("%s" % l[0:2])
    if args.bind:
        # load bind, add hydrogens and convert to mol3D
        bind,bsmi,emsg = bind_load(args.bind)
        if emsg:
            return False,emsg
        bind.convert2mol3D()
        an3D = bind # change name
        # get core size
        mindist = core3D.molsize()
        # assign reference point
        Rp = initcore3D.centermass()
        # Generate base case (separated structures)
        an3Db = mol3D()
        an3Db.copymol3D(an3D)
        base3D = protate(an3Db,Rp,[20*mindist,0.0,0.0])
        mols = []
        if args.bcharge:
            core3D.charge += int(args.bcharge)
        elif args.calccharge:
            core3D.charge += int(an3D.charge)
        # fetch base name
        fname = get_name(args,rootdir,core,ligname,bind,bsmi)
        # check if planar
        conats = core3D.getBondedAtomsnotH(0)
        planar,pos = False, False
        if conats > 3:
            combs = itertools.combinations(conats,4)
            for comb in combs:
                r = []
                for c in comb:
                    r.append(core3D.getAtomCoords(c))
                if checkplanar(r[0],r[1],r[2],r[3]):
                    planar = True
                    th,uax = rotation_params(r[0],r[1],r[2])
                    ueq = vecdiff(r[random.randint(0,3)],core3D.getAtomCoords(0))
                    break
        for i in range(0,Nogeom+1):
            # generate random sequence of parameters for rotate()
            totits = 0
            while True:
                phi = random.uniform(0.0,360.0)
                theta = random.uniform(-180.0,180.0)
                if args.bphi:
                    phi = float(args.bphi)
                if args.btheta:
                    theta = float(args.btheta)
                # if specific angle is requested force angle
                if (args.place and not args.bphi and not args.btheta):
                    if ('a' in args.place):
                        theta = 90.0
                        theta1 = -90.0
                        pos = True
                    elif ('eq' in args.place):
                        theta = 0.0
                        theta1 = 180.0
                        pos = True
                    else:
                        theta = float(args.place)
                thetax = random.uniform(0.0,360.0)
                thetay = random.uniform(0.0,360.0)
                thetaz = random.uniform(0.0,360.0)
                args.btheta = theta
                args.bphi = phi
                # translate
                an3Db = mol3D()
                an3Db.copymol3D(an3D)
                # get mask of reference atoms
                if args.bref:
                    refbP = an3D.getMask(args.bref)
                else:
                    refbP = an3D.centermass()
                if planar and pos:
                    # place axial
                    R = random.uniform(float(args.mind),float(args.maxd))
                    args.mind = R
                    args.maxd = R
                    if 'ax' in args.place:
                        newmol = setPdistanceu(an3D, refbP, core3D.getAtomCoords(0),R,uax)
                    elif 'eq' in args.place:
                        P = getPointu(core3D.getAtomCoords(0),100,ueq)
                        mindist = core3D.getfarAtomdir(P)
                        newmol = setPdistanceu(an3D, refbP, core3D.getAtomCoords(0),R+mindist,ueq)
                else:
                    # get maximum distance in the correct direction
                    Pp0 = PointTranslatetoPSph(core3D.centermass(),[0.5,0.5,0.5],[0.01,theta,phi])
                    cmcore = core3D.centermass()
                    uP = getPointu(cmcore,100,vecdiff(Pp0,cmcore)) # get far away point in space
                    mindist = core3D.getfarAtomdir(uP)
                    maxdist = mindist+float(args.maxd) # Angstrom, distance of non-interaction
                    mindist = mindist+float(args.mind) # Angstrom, distance of non-interaction
                    R = random.uniform(mindist,maxdist) # get random distance, separated for i=0
                    # rotate and place according to distance
                    tr3D = protateref(an3Db, Rp, refbP, [R,theta,phi])
                    # rotate center of mass
                    newmol = rotateRef(tr3D,refbP,[thetax,thetay,thetaz])
                    if ('theta1' in locals()):
                        an3Db = mol3D()
                        an3Db.copymol3D(an3D)
                        tr3D2 = protateref(an3Db, Rp,refbP,[R,theta1,phi])
                        newmol2 = rotateRef(tr3D2,refbP,[thetax,thetay,thetaz])
                        d1 = tr3D.distance(core3D)
                        d2 = tr3D2.distance(core3D)
                        if (d2 > d1):
                            newmol = newmol2
                # check for overlapping
                if not(newmol.overlapcheck(core3D,1)):
                    break
                if totits > 200:
                    print "WARNING: Overlapping in molecules for file "+fname+str(i)
                    break
                totits += 1
            if (i > 0):
                # write separate xyz file
                if args.bsep:
                    core3D.writesepxyz(newmol,fname+str(i))
                else:
                    # write new xyz file
                    newmol.writemxyz(core3D,fname+str(i))
                # append filename
                strfiles.append(fname+str(i))
                getinputargs(args,fname+str(i))
            else:
                # write new xyz file
                core3D.writexyz(fname+'R')
                # append filename
                strfiles.append(fname+'R')
                # write binding molecule file
                an3Db.writexyz(fname+'B')
                strfiles.append(fname+'B')
                del an3Db
                getinputargs(args,fname+'R')
                getinputargs(args,fname+'B')
    else:
        fname = name_complex(rootdir,name_core,ligands,ligoc,sernum,args,bind= False,bsmi=False)
        
        core3D.writexyz(fname)
        strfiles.append(fname)
        getinputargs(args,fname)
    pfold = rootdir.split('/',1)[-1]
    if args.calccharge:
        args.charge = core3D.charge
        print('setting charge to be ' + str(args.charge))
    # check for molecule sanity
    sanity,d0 = core3D.sanitycheck(True)
    if args.debug:
        print('setting sanity diag, min dist at ' +str(d0) + ' (higher is better)')
    this_diag.set_sanity(sanity,d0)
    this_diag.set_mol(core3D)
    this_diag.write_report(fname+'.report')
    del core3D
    if sanity:
        print 'WARNING: Generated complex is not good! Minimum distance between atoms:'+"{0:.2f}".format(d0)+'A\n'
        if args.gui:
            ssmsg = 'Generated complex in folder '+rootdir+' is no good! Minimum distance between atoms:'+"{0:.2f}".format(d0)+'A\n'
            qqb = mQDialogWarn('Warning',ssmsg)
            qqb.setParent(args.gui.wmain)
    if args.gui:
        args.gui.iWtxt.setText('In folder '+pfold+' generated '+str(Nogeom)+' structures!\n'+args.gui.iWtxt.toPlainText())
        args.gui.app.processEvents()
    print '\nIn folder '+pfold+' generated ',Nogeom,' structure(s)!'
    return strfiles, emsg, this_diag

 
