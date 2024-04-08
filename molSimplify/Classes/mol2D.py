import networkx as nx
import numpy as np
import os
from packaging import version
from molSimplify.Classes.globalvars import globalvars

try:
    from openbabel import openbabel  # version 3 style import
except ImportError:
    import openbabel  # fallback to version 2


class Mol2D(nx.Graph):

    def __repr__(self):
        atom_counts = {}
        for _, sym in self.nodes.data(data="symbol", default="X"):
            if sym not in atom_counts:
                atom_counts[sym] = 1
            else:
                atom_counts[sym] += 1

        symbols = globalvars().elementsbynum()
        formula = ""
        for sym in sorted(atom_counts.keys(),
                          key=lambda x: symbols.index(x), reverse=True):
            formula += f"{sym}{atom_counts[sym]}"
        return f"Mol2D({formula})"

    @classmethod
    def from_smiles(cls, smiles):
        mol = cls()

        # Load using openbabel OBMol
        obConversion = openbabel.OBConversion()
        OBMol = openbabel.OBMol()
        obConversion.SetInFormat('smi')
        obConversion.ReadString(OBMol, smiles)
        OBMol.AddHydrogens()

        symbols = globalvars().elementsbynum()
        # Add atoms
        for i, atom in enumerate(openbabel.OBMolAtomIter(OBMol)):
            sym = symbols[atom.GetAtomicNum() - 1]
            mol.add_node(i, symbol=sym)

        # Add bonds
        for bond in openbabel.OBMolBondIter(OBMol):
            # Subtract 1 because of zero indexing vs. one indexing
            mol.add_edge(bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1)

        return mol

    @classmethod
    def from_mol2_file(cls, filename):
        mol = cls()
        if os.path.exists(str(filename)):
            with open(filename, "r") as fin:
                lines = fin.readlines()
        else:
            lines = filename.readlines()
        # Read counts line:
        sp = lines[2].split()
        n_atoms = int(sp[0])
        n_bonds = int(sp[1])

        atom_start = lines.index("@<TRIPOS>ATOM\n") + 1
        for i, line in enumerate(lines[atom_start:atom_start + n_atoms]):
            sp = line.split()
            sym = sp[5].split(".")[0]
            mol.add_node(i, symbol=sym)

        bond_start = lines.index("@<TRIPOS>BOND\n") + 1
        for line in lines[bond_start:bond_start + n_bonds]:
            sp = line.split()
            # Subtract 1 because of zero indexing vs. one indexing
            mol.add_edge(int(sp[1]) - 1, int(sp[2]) - 1)

        return mol

    @classmethod
    def from_mol_file(cls, filename):
        mol = cls()

        with open(filename, "r") as fin:
            lines = fin.readlines()

        # Read counts line:
        sp = lines[3].split()
        n_atoms = int(sp[0])
        n_bonds = int(sp[1])

        # Add atoms (offset of 4 for the header lines):
        for i, line in enumerate(lines[4:4 + n_atoms]):
            sp = line.split()
            mol.add_node(i, symbol=sp[3])

        # Add bonds:
        for line in lines[4 + n_atoms:4 + n_atoms + n_bonds]:
            sp = line.split()
            # Subtract 1 because of zero indexing vs. one indexing
            mol.add_edge(int(sp[0]) - 1, int(sp[1]) - 1)

        return mol

    def graph_hash(self):
        # This is necessary because networkx < 2.7 had a bug in the implementation
        # of weisfeiler_lehman_graph_hash
        # https://github.com/networkx/networkx/pull/4946#issuecomment-914623654
        assert version.parse(nx.__version__) >= version.parse("2.7")
        return nx.weisfeiler_lehman_graph_hash(self, node_attr="symbol")

    def graph_hash_edge_attr(self):
        # This is necessary because networkx < 2.7 had a bug in the implementation
        # of weisfeiler_lehman_graph_hash
        # https://github.com/networkx/networkx/pull/4946#issuecomment-914623654
        assert version.parse(nx.__version__) >= version.parse("2.7")
        # Copy orginal graph before adding edge attributes
        G = self.copy()

        for i, j in G.edges:
            G.edges[i, j]["label"] = "".join(sorted([G.nodes[i]["symbol"],
                                                     G.nodes[j]["symbol"]]))

        return nx.weisfeiler_lehman_graph_hash(G, edge_attr="label")

    def graph_determinant(self, return_string=True):
        atomic_masses = globalvars().amass()
        weights = np.array(
            [
                atomic_masses[atom[1]][0]
                for atom in self.nodes(data="symbol", default="X")
            ]
        )
        adjacency = nx.to_numpy_array(self)
        mat = np.outer(weights, weights) * adjacency
        np.fill_diagonal(mat, weights)
        with np.errstate(over='raise'):
            try:
                det = np.linalg.det(mat)
            except (np.linalg.LinAlgError, FloatingPointError):
                (sign, det) = np.linalg.slogdet(mat)
                if sign != 0:
                    det = sign*det
        if return_string:
            det = str(det)
            if "e+" in det:
                sp = det.split("e+")
                det = sp[0][0:12] + "e+" + sp[1]
            else:
                det = det[0:12]
        return det
