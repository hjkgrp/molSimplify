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
        '8366132e88f24330fedbbf24367877f7'
        """
        # This is necessary because networkx < 2.7 had a bug in the implementation
        # of weisfeiler_lehman_graph_hash
        # https://github.com/networkx/networkx/pull/4946#issuecomment-914623654
        assert version.parse(nx.__version__) >= version.parse("2.7")
        return nx.weisfeiler_lehman_graph_hash(self, node_attr="symbol", iterations=5)

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
        'b9aa3fc505879a7a2a9a1789aee922f5'
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

        return nx.weisfeiler_lehman_graph_hash(G, edge_attr="label", iterations=5)

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
            coordination_paths : list
                Atom indices corresponding to each coordination path
        """
        # ensure provided coordinating atoms are all within range of atoms in molecule
        if max(catoms) >= self.number_of_nodes():
            raise KeyError('coordinating atom indices must be within range of number of atoms')
        non_catoms = [atom for atom in range(len(self.nodes)) if atom not in catoms]
        # get all pairwise combinations of coordinating atoms
        catom_pairs = list(itertools.combinations(catoms, 2))

        # get all paths between pairs of coordinating atoms which consist only of other fellow coordinating atoms
        coordination_paths = []
        for pair in catom_pairs:
            coordination_paths.extend(self.find_simple_paths(source=pair[0], sink=pair[1], constraints=non_catoms))
        # remove all paths which are subsets of other paths, but are not the maximum path (i.e., allow for rings)
        max_length = max(len(path) for path in coordination_paths) if len(coordination_paths) > 0 else 0
        coordination_paths = list({frozenset(path) for path in coordination_paths
                                   if not any(set(path).issubset(set(other_path))
                                              and path != other_path
                                              and len(path) < max_length
                                              for other_path in coordination_paths)})
        coordination_paths = [list(path) for path in coordination_paths]
        # add back in isolated atoms which are not in any paths
        coordination_paths.extend(
            [[atom] for atom in set(catoms) - set(atom for path in coordination_paths for atom in path)])

        denticity = len(coordination_paths)
        hapticity = [len(path) for path in coordination_paths]

        return denticity, hapticity, coordination_paths

from __future__ import annotations

import networkx as nx
from typing import Dict, Iterable, List, Optional, Tuple, Union


def substructure_search(
    list_of_mol2Ds: List["Mol2D"],
    r: int,
    atom_filter: Optional[Union[Iterable[str], set]] = None,
    wl_iterations: int = 5,
    include_occurrences: bool = True,
) -> Dict[str, dict]:
    """
    Radius-based motif search over Mol2D graphs using Weisfeiler–Lehman hashing.

    For each molecule and for each allowed starting atom, this builds the induced
    subgraph containing all atoms within graph distance <= r (number of bonds),
    computes a node-attributed WL hash, and groups identical hashes as "motifs".

    Parameters
    ----------
    list_of_mol2Ds : list[Mol2D]
        Molecules to search.
    r : int
        Radius in number of bonds (graph distance) around each starting atom.
        r=0 -> single atom motifs; r=1 -> nearest neighbors; r=2 -> next-nearest; etc.
    atom_filter : iterable[str] | set[str] | None
        If provided, only atoms whose `symbol` is in this set are used as starting atoms.
        Example: {"Fe", "Co", "Ni"} or {"C", "N", "O"}.
    wl_iterations : int
        Number of WL refinement iterations for hashing (default 5).
    include_occurrences : bool
        If True, store where each motif occurs (mol index, center atom, nodes in subgraph).

    Returns
    -------
    dict
        Mapping: motif_hash -> {
            "count": int,
            "unique_molecules": int,
            "occurrences": list[tuple]  # optional
        }

        Each occurrence tuple is (mol_idx, center_atom_idx, tuple(sorted(nodes_in_subgraph))).

    Notes
    -----
    - Motifs are defined purely by graph topology + node symbols within radius r.
    - If your Mol2D nodes have "symbol" attributes (as in your class), this works out of the box.
    """
    if r < 0:
        raise ValueError("r must be >= 0")

    allowed: Optional[set] = set(atom_filter) if atom_filter is not None else None

    motifs: Dict[str, dict] = {}

    for mol_idx, mol in enumerate(list_of_mol2Ds):
        # Defensive: ensure it's a graph with nodes
        if mol is None or mol.number_of_nodes() == 0:
            continue

        for center in mol.nodes:
            sym = mol.nodes[center].get("symbol", "X")
            if allowed is not None and sym not in allowed:
                continue

            # Nodes within radius r (graph distance <= r)
            lengths = nx.single_source_shortest_path_length(mol, center, cutoff=r)
            nodes_in_ball = list(lengths.keys())

            # Induced subgraph (copy so WL hash doesn't see view-only)
            subG = mol.subgraph(nodes_in_ball).copy()

            # WL hash (node-attributed)
            h = nx.weisfeiler_lehman_graph_hash(
                subG, node_attr="symbol", iterations=wl_iterations
            )

            if h not in motifs:
                motifs[h] = {
                    "count": 0,
                    "molecule_set": set(),  # internal bookkeeping
                }
                if include_occurrences:
                    motifs[h]["occurrences"] = []

            motifs[h]["count"] += 1
            motifs[h]["molecule_set"].add(mol_idx)
            if include_occurrences:
                motifs[h]["occurrences"].append(
                    (mol_idx, int(center), tuple(sorted(nodes_in_ball)))
                )

    # finalize: replace molecule_set with unique_molecules
    for h, d in motifs.items():
        d["unique_molecules"] = len(d["molecule_set"])
        del d["molecule_set"]

    return motifs


def motif_counts_only(
    list_of_mol2Ds: List["Mol2D"],
    r: int,
    atom_filter: Optional[Union[Iterable[str], set]] = None,
    wl_iterations: int = 5,
) -> Dict[str, int]:
    """
    Convenience wrapper: returns only hash -> count.
    """
    motifs = substructure_search(
        list_of_mol2Ds,
        r,
        atom_filter=atom_filter,
        wl_iterations=wl_iterations,
        include_occurrences=False,
    )
    return {h: d["count"] for h, d in motifs.items()}
