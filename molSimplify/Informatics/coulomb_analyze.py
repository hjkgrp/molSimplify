import glob
import string
import sys
import os
import numpy as np
import math
import random
import string
import numpy
from geometry import *
from atom3D import *
from globalvars import globalvars
from mol3D import*
#name    metal   ox  axlig_charge    eqlig charge    axlig_dent  eqlig_dent  axlig_connect   eqlig_connect   axlig_natoms    eqlig_natoms    axlig_mdelen    eqlig_mdelen


########### UNIT CONVERSION
HF_to_Kcal_mol = 627.5095
def create_columb_matrix(mol):
    ## create Coulomb matrix from mol3D information
    index_set = range(0,mol.natoms)
    ## fetch the database of nuclear charges
    globs =globalvars() 
    amassdict = globs.amass()
    A  = numpy.matrix(numpy.zeros((mol.natoms,mol.natoms)))
    ## main build 
    for i in index_set:
        for j in index_set:
            if i==j:
                A[i,j] = 0.5*numpy.power(float(amassdict[mol.getAtom(i).symbol()][1]),(2.4))
            else:
                this_dist = distance(mol.getAtom(i).coords(),mol.getAtom(j).coords())
                if this_dist != 0.0:
                    A[i,j] = float(amassdict[mol.getAtom(i).symbol()][1])*float(amassdict[mol.getAtom(j).symbol()][1])/float(this_dist)
                else:
                   A[i,j] = 0
    ## sort by columns in increasing order
    weights = []
    for col in A.T:
        weights.append(numpy.linalg.norm(col))
    sort_weights = (numpy.argsort(weights))[::-1]
    A = A[:,sort_weights]
    ## sort by rows in increasing order
    weights  = []
    for row in A:
        weights.append(numpy.linalg.norm(row))
    sort_weights = (numpy.argsort(weights))[::-1]
    A = A[sort_weights,:]
    return A
def pad_mol(mol,target_atoms):
    ## adds placeholder atoms 
    ## with zero nuclear charge
    ## located at the origin
    ## in order to get consistent size
    ## coulomb matrix
    this_natoms = mol.natoms
    blank_atom = atom3D(Sym='X') # placeholder type
    safe_stop = False
    counter = 0;
    while this_natoms < target_atoms and not safe_stop:
        mol.addAtom(blank_atom)
        this_natoms = mol.natoms
        counter +=1
        if counter > target_atoms:
            safe_stop = True
            print('error padding mol')
    return mol



