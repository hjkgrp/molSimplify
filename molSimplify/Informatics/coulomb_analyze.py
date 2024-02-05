import numpy as np

from molSimplify.Classes.atom3D import atom3D
from molSimplify.Classes.globalvars import globalvars
from molSimplify.Scripts.geometry import distance
from molSimplify.utils.decorators import deprecated


@deprecated('Use the correctly spelled version create_coulomb_matrix instead.')
def create_columb_matrix(mol):
    return create_coulomb_matrix(mol)


def create_coulomb_matrix(mol):
    # create Coulomb matrix from mol3D information
    index_set = list(range(0, mol.natoms))
    # fetch the database of nuclear charges
    globs = globalvars()
    amassdict = globs.amass()
    A = np.zeros((mol.natoms, mol.natoms))
    # main build
    for i in index_set:
        Zi = float(amassdict[mol.getAtom(i).symbol()][1])
        for j in index_set:
            if i == j:
                A[i, j] = 0.5*np.power(Zi, (2.4))
            else:
                Zj = float(amassdict[mol.getAtom(j).symbol()][1])
                this_dist = distance(mol.getAtom(i).coords(), mol.getAtom(j).coords())
                if this_dist != 0.0:
                    A[i, j] = Zi*Zj/float(this_dist)
                else:
                    A[i, j] = 0
    # sort by columns in increasing order
    weights = []
    for col in A.T:
        weights.append(np.linalg.norm(col))
    sort_weights = (np.argsort(weights))[::-1]
    A = A[:, sort_weights]
    # sort by rows in increasing order
    weights = []
    for row in A:
        weights.append(np.linalg.norm(row))
    sort_weights = (np.argsort(weights))[::-1]
    A = A[sort_weights, :]
    return A


def pad_mol(mol, target_atoms):
    # adds placeholder atoms
    # with zero nuclear charge
    # located at the origin
    # in order to get consistent size
    # coulomb matrix
    this_natoms = mol.natoms
    blank_atom = atom3D(Sym='X')  # placeholder type
    blank_atom.frozen = False
    safe_stop = False
    counter = 0
    while this_natoms < target_atoms and not safe_stop:
        mol.addAtom(blank_atom)
        this_natoms = mol.natoms
        counter += 1
        if counter > target_atoms:
            safe_stop = True
            print('error padding mol')
    return mol
