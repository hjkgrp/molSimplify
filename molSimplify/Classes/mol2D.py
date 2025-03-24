import networkx as nx
import numpy as np
from typing import List, Union
from packaging import version
from molSimplify.Classes.globalvars import globalvars
from molSimplify.Classes.mol3D import mol3D as Mol3D
import itertools

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
    def from_smiles(cls, smiles: str):
        """Create a Mol2D object from a SMILES string.

        Parameters
        ----------
        smiles : str
            SMILES representation of the molecule.

        Returns
        -------
        Mol2D
            Mol2D object of the molecule

        Examples
        --------
        Create a furan molecule from SMILES:

        >>> mol = Mol2D.from_smiles("o1cccc1")
        >>> mol
        Mol2D(O1C4H4)
        """
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

        with open(filename, "r") as fin:
            lines = fin.readlines()

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

    @classmethod
    def from_mol3d(cls, mol3d: Mol3D):
        if len(mol3d.graph) == 0:
            raise ValueError("Mol3D object does not have molecular graph attached.")

        mol = cls()

        for i, atom in enumerate(mol3d.atoms):
            mol.add_node(i, symbol=atom.sym)

        bonds = ((int(e[0]), int(e[1])) for e in zip(*mol3d.graph.nonzero()))
        mol.add_edges_from(bonds)
        return mol

    def graph_hash(self) -> str:
        """Calculates the node attributed graph hash of the molecule.

        Returns
        -------
        str
            node attributed graph hash

        Examples
        --------
        Create a furan molecule from SMILES:

        >>> mol = Mol2D.from_smiles("o1cccc1")

        and calculate the node attributed graph hash:

        >>> mol.graph_hash()
        'f6090276d7369c0c0a535113ec1d97cf'
        """
        # This is necessary because networkx < 2.7 had a bug in the implementation
        # of weisfeiler_lehman_graph_hash
        # https://github.com/networkx/networkx/pull/4946#issuecomment-914623654
        assert version.parse(nx.__version__) >= version.parse("2.7")
        return nx.weisfeiler_lehman_graph_hash(self, node_attr="symbol", iterations = 5)

    def graph_hash_edge_attr(self) -> str:
        """Calculates the edge attributed graph hash of the molecule.

        Returns
        -------
        str
            edge attributed graph hash

        Examples
        --------
        Create a furan molecule from SMILES:

        >>> mol = Mol2D.from_smiles("o1cccc1")

        and calculate the edge attributed graph hash:

        >>> mol.graph_hash_edge_attr()
        'a9b6fbc7b5f53613546d5e91a7544ed6'
        """
        # This is necessary because networkx < 2.7 had a bug in the implementation
        # of weisfeiler_lehman_graph_hash
        # https://github.com/networkx/networkx/pull/4946#issuecomment-914623654
        assert version.parse(nx.__version__) >= version.parse("2.7")
        # Copy orginal graph before adding edge attributes
        G = self.copy()

        for i, j in G.edges:
            G.edges[i, j]["label"] = "".join(sorted([G.nodes[i]["symbol"],
                                                     G.nodes[j]["symbol"]]))

        return nx.weisfeiler_lehman_graph_hash(G, edge_attr="label", iterations = 5)

    def graph_determinant(self, return_string: bool = True) -> Union[str, float]:
        """Calculates the molecular graph determinant.

        Parameters
        ----------
        return_string : bool, optional
            Flag to return the determinant as a string. Default is True.

        Returns
        -------
        str | float
            graph determinant

        Examples
        --------
        Create a furan molecule from SMILES:

        >>> mol = Mol2D.from_smiles("o1cccc1")

        and calculate the graph determinant:

        >>> mol.graph_determinant()
        '-19404698740'
        """
        atomic_masses = globalvars().amass()
        weights = np.array(
            [
                atomic_masses[sym][0]
                for _, sym in self.nodes(data="symbol", default="X")
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

    def find_metal(self, transition_metals_only: bool = True) -> List[int]:
        """
        Find indices of metal(s) in a Mol2D class.

        Parameters
        ----------
            transition_metals_only : bool, optional
                Only find transition metals. Default is true.

        Returns
        -------
            metal_list : list
                List of indices of metal atoms in Mol2D.

        Examples
        --------
        Build Vanadyl acetylacetonate from SMILES:

        >>> mol = Mol2D.from_smiles("CC(=[O+]1)C=C(C)O[V-3]12(#[O+])OC(C)=CC(C)=[O+]2")
        >>> mol.find_metal()
        [7]
        """
        globs = globalvars()

        metal_list = []
        for i, sym in self.nodes.data(data="symbol", default="X"):
            if sym in globs.metalslist(transition_metals_only=transition_metals_only):
                metal_list.append(i)
        return metal_list

    def find_simple_paths(self, source, sink, cutoff=None, constraints=None):
        """
        Find simple (i.e., no repeated nodes) path(s) between source and sink nodes in Mol2D class.

        Parameters
        ----------
            source: int
                Index of source node
            sink: int
                Index of sink node
            cutoff: int
                Depth at which to stop path search
                Default=None
            constraints: list
                Nodes which may not be crossed during path
                Default=None

        Returns
        -------
            simple_paths : list
                List of lists of simple paths
        """
        simple_paths = [path for path in nx.all_simple_paths(self, source=source, target=sink, cutoff=cutoff)]
        simple_paths = [path for path in simple_paths if not np.isin(path, constraints).any()] if constraints else simple_paths
        return simple_paths

    def denticity_hapticity(self, catoms):
        """
        Get denticity and hapticity from molecular graph and known coordinating atoms
        Number of coordinating atoms = denticity * hapticity

        Parameters
        ----------
            catoms: list
                List of coordinating atom indices

        Returns
        -------
            denticity : int
                Number of independent coordination paths in graph
            hapticity : list
                Length of each separate coordination path in graph
        """
        non_catoms = [atom for atom in range(len(self.nodes)) if atom not in catoms]
        # get all pairwise combinations of coordinating atoms
        catom_pairs = list(itertools.combinations(catoms, 2))

        # get all paths between pairs of coordinating atoms which consist only of other fellow coordinating atoms
        coordination_paths = []
        for pair in catom_pairs:
            coordination_paths.extend(self.find_simple_paths(source=pair[0], sink=pair[1], constraints=non_catoms))
        # remove all paths which are subsets of other paths
        coordination_paths = [path for path in coordination_paths if not any(
            set(path).issubset(set(other_path)) and path != other_path for other_path in coordination_paths)]
        # add back in isolated atoms which are not in any paths
        coordination_paths.extend(
            [[atom] for atom in set(catoms) - set(atom for path in coordination_paths for atom in path)])

        denticity = len(coordination_paths)
        hapticity = [len(path) for path in coordination_paths]

        return denticity, hapticity
