import networkx as nx
from molSimplify.Classes.globalvars import globalvars
from molSimplify.Classes.mol3DX import mol3D
from molSimplify.Classes.atom3D import atom3D
import re

def readfrommol2(mol, filename, readstring=False, trunc_sym="X"):
    # assign global variables
    globs = globalvars()
    amassdict = globs.amass()
    # read from file or from string
    if readstring:
        s = filename.splitlines()
    else:
        with open(filename, 'r') as f:
            s = f.read().splitlines()
    # set up booleans to determine block of mol2 file
    read_atoms = False
    read_bonds = False
    # reset mol3D charge
    mol.charge = 0
    # iterate through lines in mol2 string
    for line in s:
        if '<TRIPOS>ATOM' in line:
            read_atoms = True
            read_bonds = False
        elif '<TRIPOS>BOND' in line:
            read_atoms = False
            read_bonds = True
        elif '<TRIPOS>SUBSTRUCTURE' in line:
            read_bonds = False
            read_atoms = False
        # Get Atoms
        elif read_atoms:
            s_line = line.split()
            # Check redundancy in Chemical Symbols and clean atom symbol
            atom_symbol1 = re.sub('[0-9]+[A-Z]+', '', line.split()[1])
            atom_symbol1 = re.sub('[0-9]+', '', atom_symbol1)
            # do it again for the other atom symbol
            atom_symbol2 = line.split()[5]
            # get the atom type
            if len(atom_symbol2.split('.')) > 1:
                atype = atom_symbol2.split('.')[1]
            else:
                atype = None
            atom_symbol2 = atom_symbol2.split('.')[0]
            # check if either atom symbol is in the dict set that as the atom to add, 
            # get xyz and add it too
            if atom_symbol1 in list(amassdict.keys()):
                atom = atom3D(atom_symbol1, [float(s_line[2]), float(
                    s_line[3]), float(s_line[4])], name=atype,partialcharge=float(s_line[8]))
            elif atom_symbol2 in list(amassdict.keys()):
                atom = atom3D(atom_symbol2, [float(s_line[2]), float(
                    s_line[3]), float(s_line[4])], name=atype,partialcharge=float(s_line[8]))
            else:
                print('Cannot find atom symbol in amassdict')
                sys.exit()
            # add mol2 charge
            mol.charge += float(s_line[8])
            # add atom
            mol.addAtom(atom)
        # get Bonds
        elif read_bonds:  
            s_line = line.split()
            atom1=int(s_line[1])-1
            atom2=int(s_line[2])-1
            bond_type=s_line[3]
            mol.addBond(atom1,atom2,bond_type)