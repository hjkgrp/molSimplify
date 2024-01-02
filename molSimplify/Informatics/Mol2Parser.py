import numpy as np
import scipy as sp

class Mol2Parser:
    ##### This parser takes in a mol2 file and extracts info.
    ##### Currently contains methods for connectivity.

    def __init__(self, mol2path):
        self.mol2path = mol2path
        with open(self.mol2path) as f:
            data = f.readlines()
            self.data = data

    def build_BO_matrix(self):
        list_of_atomic_info = self.get_atoms()
        number_of_atoms = len(list_of_atomic_info)
        list_of_bonding_info = self.get_connectivity()
        BO_matrix = np.zeros((number_of_atoms,number_of_atoms))
        for i, row in enumerate(list_of_bonding_info):
            print(row)
            BO_matrix[int(row[1])-1, int(row[2])-1] = int(row[3])
        return BO_matrix

    def get_atoms(self):
        atom_counter = False
        list_of_atomic_info = []
        for i, row in enumerate(self.data):
            if 'BOND' in row:
                atom_counter = False
            if atom_counter:
                list_of_atomic_info.append(row.split())
            if 'ATOM' in row:
                atom_counter = True
        return list_of_atomic_info

    def get_connectivity(self):
        connectivity_info = False
        list_of_bonding_info = []
        for i, row in enumerate(self.data):
            if 'SUBSTRUCTURE' in row:
                connectivity_info = False
            if connectivity_info:
                list_of_bonding_info.append(row.split())
            if 'BOND' in row:
                connectivity_info = True
        return list_of_bonding_info
