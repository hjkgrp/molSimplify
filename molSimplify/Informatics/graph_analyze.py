# Written by JP Janet for HJK Group
# Dpt of Chemical Engineering, MIT

# #########################################################
# ###### This script is a collection of helper ############
# #######  routines that convert a molecule    #############
# #######   in mol3D form to a truncated graph #############
# #########################################################
# import modules
from molSimplify.Classes.globalvars import globalvars
from molSimplify.Classes.mol3D import mol3D
import numpy as np


def obtain_truncation(mol, con_atoms, hops):
    # # this function truncates a ligand to a certain number of
    # # hops from the core
    # Inputs:
    #       mol - mol3D class to truncate
    #       con_atoms - index of atoms that connect to metal
    #       hops - int, number of hops to truncate
    trunc_mol = mol3D(use_atom_specific_cutoffs=mol.use_atom_specific_cutoffs)
    # metal_ind = mol.findMetal()[0]
    # trunc_mol.addAtom(mol.getAtom(metal_ind))
    added_list = list()
    # added_list.append(metal_ind)
    for connections in con_atoms:
        hopped = 0
        active_set = [connections]
        while hopped < hops:
            hopped += 1
            new_active_set = list()
            for this_atom in active_set:
                # add all connection atoms
                if this_atom not in added_list:
                    trunc_mol.addAtom(mol.getAtom(this_atom))
                    added_list.append(this_atom)
                # prepare all atoms attached to this connection
                this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom)
                for bound_atoms in this_atoms_neighbors:
                    if (bound_atoms not in added_list):
                        trunc_mol.addAtom(mol.getAtom(bound_atoms))
                        added_list.append(bound_atoms)
                    [new_active_set.append(element) for element in this_atoms_neighbors]
            active_set = new_active_set
    if len(mol.graph):
        delete_inds = [x for x in range(mol.natoms) if x not in added_list]
        sort_inds = np.argsort(added_list)
        trunc_mol.atoms = [trunc_mol.atoms[x] for x in sort_inds]
        trunc_mol.graph = np.delete(np.delete(mol.graph, delete_inds, 0), delete_inds, 1)
    if mol.bo_dict:
        save_bo_dict = mol.get_bo_dict_from_inds(added_list)
        trunc_mol.bo_dict = save_bo_dict
    return trunc_mol


def obtain_truncation_metal(mol, hops):
    # # this function truncates a ligand to a certain number of
    # # hops from the core
    # Inputs:
    #       mol - mol3D class to truncate
    #       con_atoms - index of atoms that connect to metal
    #       hops - int, number of hops to truncate
    trunc_mol = mol3D(use_atom_specific_cutoffs=mol.use_atom_specific_cutoffs)
    metal_ind = mol.findMetal()[0]
    trunc_mol.addAtom(mol.getAtom(metal_ind))
    added_list = list()
    added_list.append(metal_ind)
    hopped = 0
    active_set = [metal_ind]
    while hopped < hops:
        hopped += 1
        new_active_set = list()
        for this_atom in active_set:
            # add all connection atoms
            if this_atom not in added_list:
                trunc_mol.addAtom(mol.getAtom(this_atom))
                added_list.append(this_atom)
            # prepare all atoms attached to this connection
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in added_list):
                    trunc_mol.addAtom(mol.getAtom(bound_atoms))
                    added_list.append(bound_atoms)
                [new_active_set.append(element) for element in this_atoms_neighbors]
        active_set = new_active_set
    sort_inds = np.argsort(added_list)
    mapping_sub2mol = {}
    for ii, ind in enumerate(sort_inds):
        mapping_sub2mol.update({ind: added_list[ii]})
    trunc_mol.mapping_sub2mol = mapping_sub2mol
    if len(mol.graph):
        delete_inds = [x for x in range(mol.natoms) if x not in added_list]
        trunc_mol.atoms = [trunc_mol.atoms[x] for x in sort_inds]
        trunc_mol.graph = np.delete(np.delete(mol.graph, delete_inds, 0), delete_inds, 1)
    if mol.bo_dict:
        save_bo_dict = mol.get_bo_dict_from_inds(added_list)
        trunc_mol.bo_dict = save_bo_dict
    return trunc_mol


def create_graph(mol):
    if not len(mol.graph):
        # create connectivity matrix from mol3D information
        index_set = range(0, mol.natoms)
        A = np.zeros((mol.natoms, mol.natoms))
        for i in index_set:
            this_bonded_atoms = mol.getBondedAtomsSmart(i)
            for j in index_set:
                if j in this_bonded_atoms:
                    A[i, j] = 1
    else:
        A = mol.graph
    return A


def get_lig_EN(mol, connection_atoms):
    # calculate the maximum abs electronegativity
    # difference between connection atom an all
    # neighbors
    max_EN = 0
    globs = globalvars()
    for atoms in connection_atoms:
        this_atoms_neighbors = mol.getBondedAtomsSmart(atoms)
        for bound_atoms in this_atoms_neighbors:
            this_EN = float(globs.endict()[mol.getAtom(atoms).symbol()]) - float(
                globs.endict()[mol.getAtom(bound_atoms).symbol()])
            if (abs(this_EN) >= max_EN):
                max_EN = this_EN
    return max_EN


def remove_diagonals(matrix):
    n = matrix.shape[0]
    for i in range(0, n):
        matrix[i, i] = 0
    return matrix


def kier(mol):
    copy_mol = mol3D()
    copy_mol.copymol3D(mol)
    copy_mol.deleteHs()
    A = create_graph(copy_mol)
    n = A.shape[0]
    twopath = A * A
    remove_diagonals(twopath)
    p2 = twopath.sum() / 2

    if (p2 != 0):
        two_kappa = ((np.power(n, 3) - 5 * np.power(n, 2) + 8 * n - 4)
                     / (np.power(p2, 2)))
    else:
        two_kappa = 0
    return (two_kappa)


def get_truncated_kier(ligand, connection_atoms):
    # three hop truncation
    trunc_mol = obtain_truncation(ligand, connection_atoms, 3)
    # trunc_mol.writexyz('trunc.xyz')
    this_kier = kier(trunc_mol)
    return this_kier
