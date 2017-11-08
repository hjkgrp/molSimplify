# Written by Tim Ioannidis 
# and JP Janet
# for HJK Group
# Dpt of Chemical Engineering, MIT

##########################################################
######## Defines class of 3D molecules that  #############
########     will be used to manipulate      #############
########   coordinates and other properties  #############
##########################################################

from math import sqrt
import numpy as np
from molSimplify.Classes.atom3D import atom3D
from molSimplify.Classes.globalvars import globalvars
import openbabel
import sys, time, os, subprocess, random, shutil, unicodedata, inspect, tempfile
from pkg_resources import resource_filename, Requirement
import xml.etree.ElementTree as ET
try:
    import PyQt5
    from molSimplify.Classes.miniGUI import *
    qtflag = True
except ImportError:
    qtflag = False
    pass

#########################################
### Euclidean distance between points ###
#########################################
def distance(R1,R2):
    # INPUT
    #   - R1: 3-element list representing point 1
    #   - R2: 3-element list representing point 2
    # OUTPUT
    #   - d: Euclidean distance
    dx = R1[0] - R2[0]
    dy = R1[1] - R2[1]
    dz = R1[2] - R2[2]
    d = sqrt(dx**2+dy**2+dz**2)
    return d

def mybash(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = []
    while True:
        line = p.stdout.readline()
        stdout.append(line)
        if line == '' and p.poll() != None:
            break
    return ''.join(stdout)

class mol3D:
    """ Class mol3D represents a molecule with its coordinates for
    easy manipulation in 3D space """
    ###################################
    ### constructor for mol3D class ###
    ###################################
    def __init__(self):
        """ Create a new molecule object """
        self.atoms = []
        self.natoms = 0
        self.mass = 0
        self.size = 0
        self.charge = 0
        self.ffopt = 'BA'
        self.name = '' # name of molecule
        self.OBMol = False      # holder for babel molecule
        self.cat = []        # connection atoms
        self.denticity = 0   # denticity
        self.ident = ''      # identifier
        self.globs = globalvars() # holder for global variables
        self.graph = [] # holder for molecular graph

    #######################################
    ### adds a new atom to the molecule ###
    #######################################
    def addAtom(self,atom):
        # INPUT
        #   - atom: atom3D to be added
        self.atoms.append(atom)
        if atom.frozen:
            self.atoms[-1].frozen = True
        self.natoms += 1
        self.mass += atom.mass
        self.size = self.molsize()
        self.graph = []

    #############################################
    ### aligns 2 molecules based on atoms 1,2 ###
    #############################################
    def alignmol(self,atom1,atom2):
        # INPUT
        #   - atom1: atom3D in first molecule
        #   - atom2: atom3D in second molecule
        # OUTPUT
        #   - self: aligned molecule
        # get vector of distance between atoms 1,2
        dv = atom2.distancev(atom1)
        # align molecule
        self.translate(dv)

    ###########################################
    ### performs bond centric manipulation  ###
    ###########################################
    def BCM(self,idx1,idx2,d):
        # INPUT
        #   - idx1: index of atom to be moved
        #   - idx2: index of anchor atom
        #   - d: new bond length (A)
        bondv = self.getAtom(idx1).distancev(self.getAtom(idx2)) # 1 - 2
        # compute current bond length
        u = 0.0
        for u0 in bondv:
            u += (u0*u0)
        u = sqrt(u)
        dl = d - u # dl > 0: stretch, dl < 0: shrink
        dR = [i*(d/u - 1) for i in bondv]
        for i in self.getBondedAtoms(idx1):
            if i != idx2:
                self.getAtom(i).translate(dR)
        self.getAtom(idx1).translate(dR)

    #############################################
    ### calculates center of mass of molecule ###
    #############################################
    def centermass(self):
        # OUTPUT
        #   - pcm: vector representing center of mass
        # initialize center of mass and mol mass
        pmc = [0, 0, 0]
        mmass = 0
        # loop over atoms in molecule
        if self.natoms > 0:
            for atom in self.atoms:
            # calculate center of mass (relative weight according to atomic mass)
                xyz = atom.coords()
                pmc[0] +=  xyz[0]*atom.mass
                pmc[1] +=  xyz[1]*atom.mass
                pmc[2] +=  xyz[2]*atom.mass
                mmass += atom.mass
            # normalize
            pmc[0] /= mmass
            pmc[1] /= mmass
            pmc[2] /= mmass
        else:
            pmc = False
            print 'ERROR: Center of mass calculation failed. Structure will be inaccurate.\n'
        return pmc

    ############################################
    ### calculates simple center of symmetry ###
    ############################################
    def centersym(self):
        # OUTPUT
        #   - pcm: vector representing center of mass
        # initialize center of mass and mol mass
        pmc = [0, 0, 0]
        # loop over atoms in molecule
        for atom in self.atoms:
        # calculate center of mass (relative weight according to atomic mass)
            xyz = atom.coords()
            pmc[0] +=  xyz[0]
            pmc[1] +=  xyz[1]
            pmc[2] +=  xyz[2]
        # normalize
        pmc[0] /= self.natoms
        pmc[1] /= self.natoms
        pmc[2] /= self.natoms
        return pmc

    ############################################################
    ### converts OBMol to mol3D and adds to current molecule ###
    ############################################################
    def convert2mol3D(self):
        # initialize again
        self.initialize()
        # get elements dictionary
        elem = globalvars().elementsbynum()
        # loop over atoms
        for atom in openbabel.OBMolAtomIter(self.OBMol):
            # get coordinates
            pos = [atom.GetX(),atom.GetY(),atom.GetZ()]
            # get atomic symbol
            sym = elem[atom.GetAtomicNum() -1]
            # add atom to molecule
            self.addAtom(atom3D(sym,[pos[0],pos[1],pos[2]]))
            
    ############################################################
    ### converts mol3D to OBmol and adds to current molecule ###
    ############################################################
    def convert2OBMol(self):
        # write temp xyz
        self.writexyz('tempr.xyz')
        
        obConversion = openbabel.OBConversion()
        obConversion.SetInFormat("xyz")

        OBMol = openbabel.OBMol()
        obConversion.ReadFile(OBMol,'tempr.xyz')        
        
        self.OBMol = OBMol
        os.remove('tempr.xyz')
        
    ###################################
    ### combines 2 molecules in one ###
    ###################################
    def combine(self,mol):
        # INPUT
        #   - mol: second molecule to be "adsorbed"
        # OUTPUT
        #   - cmol: combined mol3D
        cmol = self
        '''combines 2 molecules in self'''
        for atom in mol.atoms:
            cmol.addAtom(atom)
        cmol.graph = []
        return cmol

    ############################################################
    ### returns the coordinates of all atoms in the molecule ###
    ############################################################
    def coords(self):
        # OUTPUT
        #   - atom: string with xyz-style coordinates
        ss = '' # initialize returning string
        ss += "%d \n\n" % self.natoms
        for atom in self.atoms:
            xyz = atom.coords()
            ss += "%s \t%f\t%f\t%f\n" % (atom.sym,xyz[0],xyz[1],xyz[2])
        return ss

    ############################################################
    ### returns the coordinates of all atoms in the molecule ###
    ############################################################
    def coordsvect(self):
        # OUTPUT
        #   - atom: vector with xyz-style coordinates
        ss = [] # initialize returning vector
        for atom in self.atoms:
            xyz = atom.coords()
            ss.append(xyz)
        return ss

    #####################################
    ### copy molecule to new molecule ###
    #####################################
    def copymol3D(self,mol0):
        # INPUT
        #   - mol0: molecule (mol3D) to be copied
        # copy atoms
        for i,atom0 in enumerate(mol0.atoms):
            self.addAtom(atom3D(atom0.sym,atom0.coords()))
            if atom0.frozen:
                self.getAtom(i).frozen = True
        # copy other attributes
        self.cat = mol0.cat
        self.charge = mol0.charge
        self.denticity = mol0.denticity
        self.ident = mol0.ident
        self.ffopt = mol0.ffopt
    #####################################
    ##### create molecular graph #######
    #####################################
    def createMolecularGraph(self):
        ## create connectivity matrix from mol3D information
        index_set = range(0,self.natoms)
        A  = np.matrix(np.zeros((self.natoms,self.natoms)))
        for i in index_set:
            this_bonded_atoms = self.getBondedAtomsOct(i,debug=False)
            for j in index_set:
                if j in this_bonded_atoms:
                    A[i,j] = 1
        self.graph = A
    ###########################################
    ### deletes specific atom from molecule ###
    ###########################################
    def deleteatom(self,atomIdx):
        # INPUT
        #   - atomIdx: index of atom to be deleted
        self.mass -= self.getAtom(atomIdx).mass
        self.natoms -= 1
        self.graph = []
        del(self.atoms[atomIdx])

    ###########################################
    ### freezes specific atom from molecule ###
    ###########################################
    def freezeatom(self,atomIdx):
        # INPUT
        #   - atomIdx: index of atom to be frozen
        self.atoms[atomIdx].frozen = True
    ##########################################
    ### deletes listed atoms from molecule ###
    ##########################################
    def deleteatoms(self,Alist):
        # INPUT
        #   - Alist: list of atoms to be deleted
        for h in sorted(Alist,reverse=True):
            self.deleteatom(h)
    ##########################################
    ### freezes listed atoms from molecule ###
    ##########################################
    def freezeatoms(self,Alist):
        # INPUT
        #   - Alist: list of atoms to be frozen
        for h in sorted(Alist,reverse=True):
            self.freezeatom(h)
    #######################################
    ### deletes hydrogens from molecule ###
    #######################################
    def deleteHs(self):
        hlist = []
        for i in range(self.natoms):
            if self.getAtom(i).sym=='H':
                hlist.append(i)
        self.deleteatoms(hlist)

    ###########################################################
    ### gets distance between 2 molecules (centers of mass) ###
    ###########################################################
    def distance(self,mol):
        # INPUT
        #   - mol: second molecule
        # OUTPUT
        #   - pcm: distance between centers of mass
        cm0 = self.centermass()
        cm1 = mol.centermass()
        pmc = distance(cm0,cm1)
        return pmc

    ###########################################################
    ### draws svg of molecule in popup window (uses pyqt5)  ###
    ###########################################################
    def draw_svg(self,filename):
        obConversion = openbabel.OBConversion()
        obConversion.SetOutFormat("svg")
        obConversion.AddOption("i", obConversion.OUTOPTIONS, "") 
        ### return the svg with atom labels as a string
        svgstr = obConversion.WriteString(self.OBMol)
        
        namespace = "http://www.w3.org/2000/svg"
        ET.register_namespace("", namespace)
        tree = ET.fromstring(svgstr)
        svg = tree.find("{{{ns}}}g/{{{ns}}}svg".format(ns=namespace))
        newsvg = ET.tostring(svg).decode("utf-8")
        ### write unpacked svg file   
        fname = filename+'.svg'
        with open(fname, "w") as svg_file:
            svg_file.write(newsvg)
        if qtflag:
            ### Initialize fake gui   
            fakegui = miniGUI(sys.argv)
            ### Add the svg to the window
            fakegui.addsvg(fname)
            ### Show window
            fakegui.show()
        else:
            print('No PyQt5 found. SVG file written to directory.')

    #######################################
    ### finds closest metal in molecule ###
    #######################################
    def findcloseMetal(self,atom0):
        # OUTPUT
        #   - mm: indices of all metals in the molecule
        mm = False
        mindist = 1000
        for i,atom in enumerate(self.atoms):
            if atom.ismetal():
                if distance(atom.coords(),atom0.coords()) < mindist:
                    mindist = distance(atom.coords(),atom0.coords())
                    mm = i
        # if no metal, find heaviest atom
        if not mm:
            maxaw = 0
            for i,atom in enumerate(self.atoms):
                if atom.atno > maxaw:
                    mm = i
        return mm

    ###############################
    ### finds metal in molecule ###
    ###############################
    def findMetal(self):
        # OUTPUT
        #   - mm: indices of all metals in the molecule
        mm = []
        for i,atom in enumerate(self.atoms):
            if atom.ismetal():
                mm.append(i)
        return mm

    #########################################
    ### finds atoms by symbol in molecule ###
    #########################################
    def findAtomsbySymbol(self,sym):
        # INPUT
        #  - sym: symbol of atom
        # OUTPUT
        #  - mm: indices of all atoms with symbol sym in the molecule
        mm = []
        for i,atom in enumerate(self.atoms):
            if atom.sym==sym:
                mm.append(i)
        return mm

    ##############################################
    ### finds submolecule based on connections ###
    ##############################################
    def findsubMol(self,atom0,atomN):
        # INPUT
        #   - atom0: index of start of submolecule
        #   - atomN: index of atom used to separate molecule
        # OUTPUT
        #   - subm: indices of all atoms in submolecule
        subm = []
        conatoms = [atom0]
        conatoms += self.getBondedAtoms(atom0) # connected atoms to atom0
        if atomN in conatoms:
            conatoms.remove(atomN)  # check for atomN and remove
        subm += conatoms # add to submolecule
        while len(conatoms) > 0: # while list of atoms to check loop
            for atidx in subm: # loop over initial connected atoms
                if atidx != atomN: # check for separation atom
                    newcon = self.getBondedAtoms(atidx)
                    if atomN in newcon:
                        newcon.remove(atomN)
                    for newat in newcon:
                        if newat not in conatoms and newat not in subm:
                            conatoms.append(newat)
                            subm.append(newat)
                if atidx in conatoms:
                    conatoms.remove(atidx) # remove from list to check
        subm.sort()
        return subm

    ########################################
    ########################################
    ### returns a specific atom by index ###
    ########################################
    def getAtom(self,idx):
        # INPUT
        #   - idx: index of atom in molecule
        # OUTPUT
        #   atom3D of atom with index idx
        return self.atoms[idx]

    #################################
    ### returns atoms in molecule ###
    #################################
    def getAtoms(self):
        # OUTPUT
        #   number of atoms in molecule
        return self.atoms
    #################################
    ### returns # of atom types   ###
    #################################
    def getAtomTypes(self):
        # OUTPUT
        #   list of types atoms in molecule
        unique_atoms_list = list()
        for atoms in self.getAtoms():
            if atoms.symbol() not in unique_atoms_list:
                        unique_atoms_list.append(atoms.symbol())

        return unique_atoms_list
    ############################################
    ### returns coordinates of atom by index ###
    ############################################
    def getAtomCoords(self,idx):
        # INPUT
        #   - idx: index of atom in molecule
        # OUTPUT
        #   coordinates
        return self.atoms[idx].coords()

    #######################################################
    ### returns list of bonded atoms to a specific atom ###
    #######################################################
    def getBondedAtoms(self,ind):
        # INPUT
        #   - ind: index of reference atom
        # OUTPUT
        #   - nats: list of indices of connected atoms
        ratom = self.getAtom(ind)
        # calculates adjacent number of atoms
        nats = []
        for i,atom in enumerate(self.atoms):
            d = distance(ratom.coords(),atom.coords())
            distance_max = 1.30*(atom.rad+ratom.rad) 
            if atom.symbol() == "C" and not ratom.symbol() == "H":
                distance_max = min(2.8,distance_max)
            if ratom.symbol() == "C" and not atom.symbol() == "H":
                distance_max = min(2.8,distance_max)
            if ratom.symbol() == "H" and atom.ismetal:
                ## tight cutoff for metal-H bonds
                distance_max = 1.1*(atom.rad+ratom.rad) 
            if atom.symbol() == "H" and ratom.ismetal:
                ## tight cutoff for metal-H bonds
                distance_max = 1.1*(atom.rad+ratom.rad) 
            if (d < distance_max and i!=ind):
                    nats.append(i)
                    
        return nats
    #######################################################
    ### returns list of bonded atoms to a specific atom ###
    ### specialized for use in octagedral single metal  ###
    #################### complexes  #######################
    #######################################################
    def getBondedAtomsOct(self,ind,debug=False):
        # INPUT
        #   - ind: index of reference atom
        # OUTPUT
        #   - nats: list of indices of connected atoms
        ratom = self.getAtom(ind)
        #print('called slow function...')
        # calculates adjacent number of atoms
        nats = []
        for i,atom in enumerate(self.atoms):
            valid = True # flag 
            d = distance(ratom.coords(),atom.coords())
            ## default interatomic radius
            ## for non-metalics
            distance_max = 1.15*(atom.rad+ratom.rad)
            if atom.ismetal() or ratom.ismetal(): 
                if debug:
                    print('metal in  cat ' + str(atom.symbol()) + ' and rat ' +str(ratom.symbol()) )
                ## one the atoms is a metal!
                ## use a longer max for metals
                distance_max = 1.30*(atom.rad+ratom.rad) 
                if d < distance_max and i!=ind:
                    ### trim Hydrogens
                    if atom.symbol() == 'H' or ratom.symbol() == 'H':
                        if debug:
                            print('invalid due to hydrogens: ')
                            print(atom.symbol())
                            print(ratom.symbol())
                        valid = False
                    if d < distance_max and i!=ind and valid:
                        if atom.symbol() == "C":           
                            ## in this case, atom might be intruder C!
                            possible_inds = self.getBondedAtomsnotH(ind) ## bonded to metal
                            if len(possible_inds)>6:
                                metal_prox = sorted(possible_inds,key=lambda x: self.getDistToMetal(x,ind))
                               
                                allowed_inds = metal_prox[0:6]
                                if debug:
                                    print('ind: '+str(ind))
                                    print('metal prox:' + str(metal_prox))
                                    print('trimmed to '+str(allowed_inds))
                                    print(allowed_inds)
                                if not i in allowed_inds:
                                    valid = False
                                    if debug:
                                        print('bond rejected based on atom: ' + str(i) + ' not in ' +str(allowed_inds))
                                else:
                                    if debug:
                                        print('Ok based on atom')
                        if ratom.symbol() == "C":  
                            ## in this case, ratom might be intruder C!
                            possible_inds = self.getBondedAtomsnotH(i) ## bonded to metal
                            metal_prox = sorted(possible_inds,key=lambda x: self.getDistToMetal(x,i))
                            if len(possible_inds)>6:
                                allowed_inds = metal_prox[0:6]
                                if debug:
                                    print('ind: '+str(ind))
                                    print('metal prox:' + str(metal_prox))
                                    print('trimmed to '+str(allowed_inds))
                                    print(allowed_inds)
                                if not ind in allowed_inds:
                                    valid = False
                                    if debug:
                                        print('bond rejected based on ratom ' + str(ind) + ' with symbol ' + ratom.symbol())
                                else:
                                    if debug:
                                        print('ok based on ratom...')
                else:
                    if debug:
                        print('distance too great')
            if (d < distance_max and i!=ind):
                if valid:
                    if debug:
                        print('Valid atom  ind ' + str(i) + ' (' + atom.symbol() + ') and ' + str(ind) + ' (' + ratom.symbol() + ')')
                        print(' at distance ' + str(d) + ' (which is less than ' + str(distance_max) + ')')
                    nats.append(i)
                else:
                    if debug:
                        print('atom  ind ' + str(i) + ' (' + atom.symbol() + ')')
                        print('has been disallowed from bond with ' + str(ind) + ' (' + ratom.symbol() + ')')
                        print(' at distance ' + str(d) + ' (which would normally be less than ' + str(distance_max) + ')')
                    if d<2 and not atom.symbol() == 'H' and not ratom.symbol() == 'H':
                        print('Error, mol3D could not understand conenctivity in mol' )
                        os.exit()
        return nats
    #######################################################
    ### returns list of bonded atoms to a specific atom ###
    ###### using the molecular graph, or creates it  ######
    #######################################################
    ## OCTHEDRAL COMPLEXES ONLY
    def getBondedAtomsSmart(self,ind):
        if not len(self.graph):
            self.createMolecularGraph()
        return list(np.nonzero(np.ravel(self.graph[ind]))[0])

        
        
    
    #######################################################
    ### returns list of bonded atoms to a specific atom ###
    #######################################################
    def getBondedAtomsnotH(self,ind):
        # INPUT
        #   - ind: index of reference atom
        # OUTPUT
        #   - nats: list of indices of connected atoms
        ratom = self.getAtom(ind)
        # calculates adjacent number of atoms
        nats = []
        for i,atom in enumerate(self.atoms):
            d = distance(ratom.coords(),atom.coords())
            distance_max = 1.15*(atom.rad+ratom.rad)
            if atom.ismetal() or ratom.ismetal(): 
                distance_max = 1.30*(atom.rad+ratom.rad) 
            else:
                distance_max = 1.15*(atom.rad+ratom.rad)
            if (d < distance_max and i!=ind and atom.sym!='H'):
                nats.append(i)
        return nats

    ###########################################################
    ### returns distance of atom that's the furthest away #####
    ########### from COM in a specific direction ##############
    ###########################################################
    def getfarAtomdir(self,uP):
        # INPUT
        #   - u: direction to search
        # OUTPUT
        #   - d: distance of atom
        dd = 1000.0
        atomc = [0.0,0.0,0.0]
        for atom in self.atoms:
            d0 = distance(atom.coords(),uP)
            if d0 < dd:
                dd = d0
                atomc = atom.coords()
        return distance(self.centermass(),atomc)

    ####################################
    ### gets hydrogens from molecule ###
    ####################################
    def getHs(self):
        hlist = []
        for i in range(self.natoms):
            if self.getAtom(i).sym=='H':
                hlist.append(i)
        return hlist

    ###########################################################
    ### returns list of hydrogens bonded to a specific atom ###
    ###########################################################
    def getHsbyAtom(self,ratom):
        # INPUT
        #   - ratom: reference atom3D
        # OUTPUT
        #   - nHs: list of indices of connected Hydrogens
        nHs = []
        for i,atom in enumerate(self.atoms):
            if atom.sym == 'H':
                d = distance(ratom.coords(),atom.coords())
                if (d < 1.2*(atom.rad+ratom.rad) and d > 0.01):
                    nHs.append(i)
        return nHs

    ###########################################################
    ### returns list of hydrogens bonded to a specific atom ###
    ###########################################################
    def getHsbyIndex(self,idx):
        # INPUT
        #   - idx: index of reference atom
        # OUTPUT
        #   - nHs: list of indices of connected Hydrogens
        # calculates adjacent number of hydrogens
        nHs = []
        for i,atom in enumerate(self.atoms):
            if atom.sym == 'H':
                d = distance(atom.coords(),self.getAtom(idx).coords())
                if (d < 1.2*(atom.rad+self.getAtom(idx).rad) and d > 0.01):
                    nHs.append(i)
        return nHs

    #######################################################
    ### gets closest atom from molecule to another atom ###
    #######################################################
    def getClosestAtom(self,atom0):
        # INPUT
        #   - atom0: reference atom3D
        # OUTPUT
        #   - idx: index of closest atom to atom0 from molecule
        idx = 0
        cdist = 1000
        for iat,atom in enumerate(self.atoms):
            ds = atom.distance(atom0)
            if (ds < cdist):
                idx = iat
                cdist = ds
        return idx

    ###########################################
    ### gets point that corresponds to mask ###
    ###########################################
    def getMask(self,mask):
        # INPUT
        #   - mask: identifier for atoms
        # OUTPUT
        #   - P: 3D point corresponding to masked atoms
        globs = globalvars()
        elements = globs.elementsbynum()
        # check center of mass
        ats = []
        # loop over entries in mask
        for entry in mask:
            # check for center of mass
            if ('com' in entry.lower()) or ('cm' in entry.lower()):
                return self.centermass()
            # check for range
            elif '-' in entry:
                at0 = entry.split('-')[0]
                at1 = entry.split('-')[-1]
                for i in range(int(at0),int(at1)+1):
                    ats.append(i-1) # python indexing
            elif entry in elements:
                ats += self.findAtomsbySymbol(entry)
            else:
                # try to convert to integer
                try:
                    t = int(entry)
                    ats.append(t-1)
                except:
                    return self.centermass()
        maux = mol3D()
        for at in ats:
            maux.addAtom(self.getAtom(at))
        if maux.natoms==0:
            return self.centermass()
        else:
            return maux.centermass()

    #######################################################
    ### gets closest atom from molecule to another atom ###
    #######################################################
    def getClosestAtomnoHs(self,atom0):
        # INPUT
        #   - atom0: reference atom3D
        # OUTPUT
        #   - idx: index of closest atom to atom0 from molecule
        idx = 0
        cdist = 1000
        for iat,atom in enumerate(self.atoms):
            ds = atom.distance(atom0)
            if (ds < cdist) and atom.sym!='H':
                idx = iat
                cdist = ds
        return idx
    #######################################################
    ### gets closest atom from molecule to a metal atom ###
    #######################################################
    def getDistToMetal(self,idx,metalx):
        # INPUT
        #   - idx: index of an atom in mol3D
        # OUTPUT
        #   - d: distance to metal
        d = self.getAtom(idx).distance(self.getAtom(metalx))
        #print('distance to ' + self.getAtom(idx).symbol() + ' is ' + str(d))
        return d

    #######################################################################
    ### gets closest atom from molecule to an atom in the same molecule ###
    #######################################################################
    def getClosestAtomnoHs2(self,atidx):
        # INPUT
        #   - atom0: reference atom3D
        # OUTPUT
        #   - idx: index of closest atom to atom0 from molecule
        idx = 0
        cdist = 1000
        for iat,atom in enumerate(self.atoms):
            ds = atom.distance(self.getAtom(atidx))
            if (ds < cdist) and atom.sym!='H' and iat!=atidx:
                idx = iat
                cdist = ds
        return idx

    ###############################################################
    ### assigns openbabel molecule from smiles or xyz/mol files ###
    ###############################################################
    def getOBMol(self,fst,convtype):
        # INPUT
        #   - fst: filename
        #   - convtype: type of input file
        # OUTPUT
        #   - OBMol: openbabel molecule loaded from file
        obConversion = openbabel.OBConversion()
        OBMol = openbabel.OBMol()
        if convtype == 'smistring':
            obConversion.SetInFormat('smi')
            obConversion.ReadString(OBMol,fst)
        else:
            obConversion.SetInFormat(convtype[:-1])
            obConversion.ReadFile(OBMol,fst)
        if 'smi' in convtype:
            OBMol.AddHydrogens()    
            b = openbabel.OBBuilder()
            b.Build(OBMol)
        #forcefield = openbabel.OBForceField.FindForceField('mmff94')
        #forcefield.Setup(OBMol)
        #forcefield.ConjugateGradients(200)
        #forcefield.GetCoordinates(OBMol)
        self.OBMol = OBMol
        return OBMol

    ############################################
    ### initialize for conversion from OBMol ###
    ############################################
    def initialize(self):
        """ Remove attributes """
        self.atoms = []
        self.natoms = 0
        self.mass = 0
        self.size = 0
        self.graph = []

    ################################################################
    ### calculates maximum distance between atoms in 2 molecules ###
    ################################################################
    def maxdist(self,mol):
        # INPUT
        #   - mol: second molecule
        # OUTPUT
        #   - maxd: maximum distance between atoms of the 2 molecules
        maxd = 0
        for atom1 in mol.atoms:
            for atom0 in self.atoms:
                if (distance(atom1.coords(),atom0.coords()) > maxd):
                    maxd = distance(atom1.coords(),atom0.coords())
        return maxd

    ################################################################
    ### calculates minimum distance between atoms in 2 molecules ###
    ################################################################
    def mindist(self,mol):
        # INPUT
        #   - mol: second molecule
        # OUTPUT
        #   - mind: minimum distance between atoms of the 2 molecules
        mind = 1000
        for atom1 in mol.atoms:
            for atom0 in self.atoms:
                if (distance(atom1.coords(),atom0.coords()) < mind):
                    mind = distance(atom1.coords(),atom0.coords())
        return mind

    #################################################################
    ### calculates minimum distance between atoms in the molecule ###
    #################################################################
    def mindistmol(self):
        # INPUT
        #   - mol: second molecule
        # OUTPUT
        #   - mind: minimum distance between atoms of the 2 molecules
        mind = 1000
        for ii,atom1 in enumerate(self.atoms):
            for jj,atom0 in enumerate(self.atoms):
                d = distance(atom1.coords(),atom0.coords())
                if  (d < mind) and ii!=jj:
                    mind = distance(atom1.coords(),atom0.coords())
        return mind

    #############################################################################
    ### calculates minimum distance between atoms in the molecule and a point ###
    #############################################################################
    def mindisttopoint(self,point):
        # INPUT
        #   - mol: second molecule
        #   - point: [x,y,z] of point
        # OUTPUT
        #   - mind: minimum distance between atoms and point
        mind = 1000
        for atom1 in self.atoms:
            d = distance(atom1.coords(),point)
            if (d < mind):
                mind = d
        return mind

    #############################################################################
    ### calculates minimum distance between non-hydrogen atoms in 2 molecules ###
    #############################################################################
    def mindistnonH(self,mol):
        # INPUT
        #   - mol: second molecule
        # OUTPUT
        #   - mind: minimum distance between atoms of the 2 molecules
        mind = 1000
        for atom1 in mol.atoms:
            for atom0 in self.atoms:
                if (distance(atom1.coords(),atom0.coords()) < mind):
                    if (atom1.sym!='H' and atom0.sym!='H'):
                        mind = distance(atom1.coords(),atom0.coords())
        return mind

    ###########################################
    ### calculates the size of the molecule ###
    ###########################################
    def molsize(self):
        # OUTPUT
        #   - maxd: maximum distance between atom and center of mass
        maxd = 0
        cm = self.centermass()
        for atom in self.atoms:
            if distance(cm,atom.coords()) > maxd:
                maxd = distance(cm,atom.coords())
        return maxd

    ################################################
    ### checks for overlap with another molecule ###
    ################################################
    def overlapcheck(self,mol,silence):
        # INPUT
        #   - mol: second molecule
        #   - silence: flag for printing warning
        # OUTPUT
        #   - overlap: flag for overlap (True if there is overlap)
        overlap = False
        for atom1 in mol.atoms:
            for atom0 in self.atoms:
                if (distance(atom1.coords(),atom0.coords()) < 0.85*(atom1.rad + atom0.rad)):
                    overlap = True
                    if not (silence):
                        print "#############################################################"
                        print "!!!Molecules might be overlapping. Increase distance!!!"
                        print "#############################################################"
                    break
        return overlap

    ################################################
    ### checks for overlap with another molecule ###
    ###         with increased tolerance         ###
    ################################################
    def overlapcheckh(self,mol):
        # INPUT
        #   - mol: second molecule
        #   - silence: flag for printing warning
        # OUTPUT
        #   - overlap: flag for overlap (True if there is overlap)
        overlap = False
        for atom1 in mol.atoms:
            for atom0 in self.atoms:
                if (distance(atom1.coords(),atom0.coords()) < 1.0):
                    overlap = True
                    break
        return overlap
    #############################
    ### print xyz coordinates ###
    #############################
    def printxyz(self):
        # OUTPUT
        #   - atom: string with xyz-style coordinates
        ''' prints xyz coordinates for molecule'''
        for atom in self.atoms:
            xyz = atom.coords()
            ss = "%s \t%f\t%f\t%f\n" % (atom.sym,xyz[0],xyz[1],xyz[2])
            print ss

    ###################################
    ### read molecule from xyz file ###
    ###################################
    def readfromxyz(self,filename):
        # INPUT
        #   - filename: xyz file
        ''' reads molecule from xyz file'''
        self.graph = [] 
        fname = filename.split('.xyz')[0]
        f = open(fname+'.xyz','r')
        s = f.read().splitlines()
        f.close()
        for line in s[2:]:
            l = filter(None,line.split(None))
            if len(l) > 3:
                atom = atom3D(l[0],[float(l[1]),float(l[2]),float(l[3])])
                self.addAtom(atom)

    ################################################
    ### calculate the RMSD between two molecules ###
    ################################################
    def rmsd(self,mol2):
        # INPUT
        #   - mol2: second molecule
        # OUTPUT
        #   rmsd between molecules
        """ Calculate Root-mean-square deviation from two sets of vectors V and W.
        """
        Nat0 = self.natoms
        Nat1 = mol2.natoms
        if (Nat0 != Nat1):
            print "ERROR: RMSD can be calculated only for molecules with the same number of atoms.."
            return NaN
        else:
            rmsd = 0
            for atom0,atom1 in zip(self.getAtoms(),mol2.getAtoms()):
                rmsd += (atom0.distance(atom1))**2
            rmsd /= Nat0
            return sqrt(rmsd)

    ##############################################
    ### Checks for overlap within the molecule ###
    ##############################################
    def sanitycheck(self,silence):
        # INPUT
        #   - mol: second molecule
        #   - silence: flag for printing warning
        # OUTPUT
        #   - overlap: flag for overlap (True if there is overlap)
        overlap = False
        mind = 1000
        for ii,atom1 in enumerate(self.atoms):
            for jj,atom0 in enumerate(self.atoms):
                if ii!=jj and (distance(atom1.coords(),atom0.coords()) < 0.7*(atom1.rad + atom0.rad)):
                    overlap = True
                    if distance(atom1.coords(),atom0.coords()) < mind:
                        mind = distance(atom1.coords(),atom0.coords())
                    if not (silence):
                        print "#############################################################"
                        print "!!!Molecules might be overlapping. Increase distance!!!"
                        print "#############################################################"
                    break
        return overlap,mind

    ##########################################
    ### translates molecule by vector dxyz ###
    ##########################################
    def translate(self,dxyz):
        # INPUT
        #   - dxyz: translation vector
        for atom in self.atoms:
            atom.translate(dxyz)

    #################################
    ### write xyz file for gamess ###
    #################################
    def writegxyz(self,filename):
        # INPUT
        #   - filename: name for xyz file
        ''' writes gamess format file for self molecule'''
        ss = '' # initialize returning string
        ss += "Date:"+time.strftime('%m/%d/%Y %H:%M')+", XYZ structure generated by mol3D Class, "+self.globs.PROGRAM+"\nC1\n"
        for atom in self.atoms:
            xyz = atom.coords()
            ss += "%s \t%.1f\t%f\t%f\t%f\n" % (atom.sym,float(atom.atno),xyz[0],xyz[1],xyz[2])
        fname = filename.split('.gxyz')[0]
        f=open(fname+'.gxyz','w')
        f.write(ss)
        f.close()

    ######################
    ### write xyz file ###
    ######################
    def writexyz(self,filename):
        # INPUT
        #   - filename: name for xyz file
        ''' writes xyz file for self molecule'''
        ss = '' # initialize returning string
        ss += str(self.natoms)+"\n"+time.strftime('%m/%d/%Y %H:%M')+", XYZ structure generated by mol3D Class, "+self.globs.PROGRAM+"\n"
        for atom in self.atoms:
            xyz = atom.coords()
            ss += "%s \t%f\t%f\t%f\n" % (atom.sym,xyz[0],xyz[1],xyz[2])
        fname = filename.split('.xyz')[0]
        f=open(fname+'.xyz','w')
        f.write(ss)
        f.close()

    ###############################################
    ### write xyz file for 2 molecules combined ###
    ###############################################
    def writemxyz(self,mol,filename):
        # INPUT
        #   - mol: second molecule
        #   - filename: name for xyz file
        ''' writes xyz file for 2 molecules'''
        ss = '' # initialize returning string
        ss += str(self.natoms+mol.natoms)+"\n"+time.strftime('%m/%d/%Y %H:%M')+", XYZ structure generated by mol3D Class, "+self.globs.PROGRAM+"\n"
        for atom in self.atoms:
            xyz = atom.coords()
            ss += "%s \t%f\t%f\t%f\n" % (atom.sym,xyz[0],xyz[1],xyz[2])
        for atom in mol.atoms:
            xyz = atom.coords()
            ss += "%s \t%f\t%f\t%f\n" % (atom.sym,xyz[0],xyz[1],xyz[2])
        fname = filename.split('.xyz')[0]
        f=open(fname+'.xyz','w')
        f.write(ss)
        f.close()

    ################################################
    ### write xyz file for 2 molecules separated ###
    ################################################
    def writesepxyz(self,mol,filename):
        # INPUT
        #   - mol: second molecule
        #   - filename: name for xyz file
        ''' writes xyz file for 2 molecules'''
        ss = '' # initialize returning string
        ss += str(self.natoms)+"\n"+time.strftime('%m/%d/%Y %H:%M')+", XYZ structure generated by mol3D Class, "+self.globs.PROGRAM+"\n"
        for atom in self.atoms:
            xyz = atom.coords()
            ss += "%s \t%f\t%f\t%f\n" % (atom.sym,xyz[0],xyz[1],xyz[2])
        ss += "--\n"+str(mol.natoms)+"\n\n"
        for atom in mol.atoms:
            xyz = atom.coords()
            ss += "%s \t%f\t%f\t%f\n" % (atom.sym,xyz[0],xyz[1],xyz[2])
        fname = filename.split('.xyz')[0]
        f=open(fname+'.xyz','w')
        f.write(ss)
        f.close()

    #####################################
    ### print methods for mol3D class ###
    #####################################
    def __repr__(self):
        # OUTPUT
        #   - ss: string with all methods
        # overloaded function
        """ when calls mol3D object without attribute e.g. t """
        ss = "\nClass mol3D has the following methods:\n"
        for method in dir(self):
            if callable(getattr(self, method)):
                ss += method +'\n'
        return ss
