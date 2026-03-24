import os
import re
import sys
import time
import tempfile
import numpy as np
import networkx as nx
from scipy.sparse import csgraph
try:
    from openbabel import openbabel  # version 3 style import
except ImportError:
    import openbabel  # fallback to version 2

from molSimplify.Classes.atom3D import atom3D
from molSimplify.Classes.globalvars import globalvars
from molSimplify.utils.decorators import deprecated

class Mol3DIO:
    """
    Mixin class for mol3D containing I/O routines.
    """
    @classmethod
    def from_smiles(cls, smiles, gen3d: bool = True):
        """
        Generate a mol3D object from a SMILES string.
        """

        mol = cls()
        mol.getOBMol(smiles, "smistring", gen3d=gen3d)

        elem = globalvars().elementsbynum()
        # Add atoms
        for atom in openbabel.OBMolAtomIter(mol.OBMol):
            # Get coordinates
            pos = [atom.GetX(), atom.GetY(), atom.GetZ()]
            # Get atomic symbol
            sym = elem[atom.GetAtomicNum() - 1]
            # Add atom to molecule
            # atom3D_list.append(atom3D(sym, pos))
            mol.addAtom(atom3D(sym, pos))

        # Add bonds
        mol.graph = np.zeros([mol.natoms, mol.natoms])
        mol.bo_mat = np.zeros([mol.natoms, mol.natoms])
        for bond in openbabel.OBMolBondIter(mol.OBMol):
            i = bond.GetBeginAtomIdx() - 1
            j = bond.GetEndAtomIdx() - 1
            bond_order = bond.GetBondOrder()
            if bond.IsAromatic():
                bond_order = 1.5
            mol.graph[i, j] = mol.graph[j, i] = 1
            mol.bo_mat[i, j] = mol.bo_mat[j, i] = bond_order
            mol.bo_dict[tuple(sorted([i, j]))] = bond_order
        return mol

    def get_smiles(self, canonicalize=False, use_mol2=False) -> str:
        """
        Returns the SMILES string representing the mol3D object.

        Parameters
        ----------
            canonicalize : bool, optional
                Openbabel canonicalization of smiles. Default is False.
            use_mol2 : bool, optional
                Use graph in mol2 instead of interpreting it. Default is False.

        Returns
        -------
            smiles : str
                SMILES from a mol3D object. Watch out for charges.
        """

        # Used to get the SMILES string of a given mol3D object.
        conv = openbabel.OBConversion()
        conv.SetOutFormat('smi')
        if canonicalize:
            conv.SetOutFormat('can')
        if not self.OBMol:
            if use_mol2:
                # Produces a smiles with the enforced BO matrix,
                # which is needed for correct behavior for fingerprints.
                self.convert2OBMol2(ignoreX=True)
            else:
                self.convert2OBMol()
        smi = conv.WriteString(self.OBMol).split()[0]
        return smi

    def get_smilesOBmol_charge(self):
        """
        Get the charge of a mol3D object through adjusted OBmol hydrogen/smiles conversion.
        Note that currently this function should only be applied to ligands (organic molecules).
        """

        # Use this as dummy mol3D class. Shouldn't interfere with other functionality.
        nh = len([x for x in self.symvect() if x == 'H'])  # Get initial hydrogens count.
        smi = self.get_smiles(use_mol2=True, canonicalize=True)
        self.my_mol_trunc = self.from_smiles(smi, gen3d=False)
        charge = self.my_mol_trunc.OBMol.GetTotalCharge()
        formula = self.my_mol_trunc.OBMol.GetFormula()
        if 'H' in formula:
            hs_tmp = formula.split('H')[1]
            nh_obmol = ''
            if len(hs_tmp) > 0:
                if hs_tmp[0].isnumeric():
                    for x in hs_tmp:
                        if x.isnumeric():
                            nh_obmol += x
                        else:
                            break
                else:
                    nh_obmol += '1'
            else:
                nh_obmol += '1'
        else:
            nh_obmol = '0'
        nh_obmol = int(nh_obmol)
        charge = charge - nh_obmol + nh
        return charge

    def read_bo_from_mol(self, molfile):
        with open(molfile, 'r') as fo:
            for line in fo:
                ll = line.split()
                if len(ll) == 7 and all([x.isdigit() for x in ll]):
                    self.bo_mat[int(ll[0])-1, int(ll[1])-1] = int(ll[2])
                    self.bo_mat[int(ll[1])-1, int(ll[0])-1] = int(ll[2])

    def read_bond_order(self, bofile):
        """
        Get bond order information from file.

        Parameters
        ----------
            bofile : str
                Path to a bond order file.
        """

        bonds_organic = {'H': 1, 'C': 4, 'N': 3,
                         'O': 2, 'F': 1, 'P': 3, 'S': 2}
        self.bv_dict = {}
        self.ve_dict = {}
        self.bvd_dict = {}
        self.bodstd_dict = {}
        self.bodavrg_dict = {}
        self.bo_mat = np.zeros(shape=(self.natoms, self.natoms))
        if os.path.isfile(bofile):
            with open(bofile, "r") as fo:
                for line in fo:
                    ll = line.split()
                    if len(ll) == 5 and ll[0].isdigit() and ll[1].isdigit():
                        self.bo_mat[int(ll[0]), int(ll[1])] = float(ll[2])
                        self.bo_mat[int(ll[1]), int(ll[0])] = float(ll[2])
                        if int(ll[0]) == int(ll[1]):
                            self.bv_dict.update({int(ll[0]): float(ll[2])})
        else:
            print(("bofile does not exist.", bofile))
        for ii in range(self.natoms):
            self.ve_dict.update({ii: bonds_organic[self.atoms[ii].symbol()]})
            self.bvd_dict.update({ii: self.bv_dict[ii] - self.ve_dict[ii]})
            vec = self.bo_mat[ii, :][self.bo_mat[ii, :] > 0.1]
            if vec.shape[0] == 0:
                self.bodstd_dict.update({ii: 0})
                self.bodavrg_dict.update({ii: 0})
            else:
                devi = [abs(v - max(round(v), 1)) for v in vec]
                self.bodstd_dict.update({ii: np.std(devi)})
                self.bodavrg_dict.update({ii: np.mean(devi)})

    def read_charge(self, chargefile):
        """
        Get charge information from file.

        Parameters
        ----------
            chargefile : str
                Path to a charge file.
        """

        self.charge_dict = {}
        if os.path.isfile(chargefile):
            with open(chargefile, "r") as fo:
                for line in fo:
                    ll = line.split()
                    if len(ll) == 3 and ll[0].isdigit():
                        self.charge_dict.update({int(ll[0]) - 1: float(ll[2])})
        else:
            print(("chargefile does not exist.", chargefile))

    @deprecated('read_smiles is deprecated and will be removed in a future release. '
                'Use mol3D.from_smiles() instead.')
    def read_smiles(self, smiles, ff="mmff94", steps=2500):
        """
        Read a smiles string and convert it to a mol3D class instance.

        .. deprecated::
            This method is deprecated and will be removed in a future release.
            Use :meth:`from_smiles` instead.

        Parameters
        ----------
            smiles : str
                SMILES string to be interpreted by openbabel.
            ff : str, optional
                Forcefield to be used by openbabel. Default is mmff94.
            steps : int, optional
                Steps to be taken by forcefield. Default is 2500.
        """

        # Used to convert from one format (ex, SMILES) to another (ex, mol3D).
        obConversion = openbabel.OBConversion()

        # The input format "SMILES"; Reads the SMILES - all stacked as 2-D - one on top of the other.
        obConversion.SetInFormat("SMILES")
        OBMol = openbabel.OBMol()
        obConversion.ReadString(OBMol, smiles)

        # Adds hydrogens
        OBMol.AddHydrogens()

        # Get a 3-D structure with H's.
        builder = openbabel.OBBuilder()
        builder.Build(OBMol)

        # Force field optimization is done in the specified number of "steps" using the specified "ff" force field.
        if ff:
            forcefield = openbabel.OBForceField.FindForceField(ff)
            s = forcefield.Setup(OBMol)
            if not s:
                print('FF setup failed')
            forcefield.ConjugateGradients(steps)
            forcefield.GetCoordinates(OBMol)

        # mol3D structure
        self.OBMol = OBMol
        self.convert2mol3D()

    def readfrommol(self, filename):
        """
        Read mol into a mol3D class instance. Stores the bond orders and atom types.

        Parameters
        -------
            filename : string
                String of path to MOL file. Path may be local or global.
        """

        with open(filename, 'r') as f:
            contents = f.readlines()

        counts_block_line_idx, num_atoms, num_bonds = None, None, None

        # Searching for counts block.
        for idx, line in enumerate(contents):
            split_line = line.split()

            # Counts block
            if len(split_line) == 11 or len(split_line) == 12:
                counts_block_line_idx = idx
                num_atoms = int(split_line[0])
                num_bonds = int(split_line[1])
                break

        if counts_block_line_idx is None:
            print('Failed to read the .mol file.')
            return

        # Atoms block
        for idx, line in enumerate(contents[counts_block_line_idx+1:counts_block_line_idx+num_atoms+1]):
            split_line = line.split()
            x_coord = float(split_line[0])
            y_coord = float(split_line[1])
            z_coord = float(split_line[2])
            sym = split_line[3]

            my_atom = atom3D(Sym=sym, xyz=[x_coord,y_coord,z_coord])
            self.addAtom(my_atom)

        self.graph = np.zeros((num_atoms, num_atoms))
        self.bo_mat = np.zeros((num_atoms, num_atoms))
        self.bo_dict = {}

        # Bonds block
        for idx, line in enumerate(contents[counts_block_line_idx+num_atoms+1:counts_block_line_idx+num_atoms+num_bonds+1]):
            split_line = line.split()

            atom1_idx = int(split_line[0])-1
            atom2_idx = int(split_line[1])-1
            bond_type = split_line[2]

            self.graph[atom1_idx, atom2_idx] = 1
            self.graph[atom2_idx, atom1_idx] = 1
            self.bo_mat[atom1_idx, atom2_idx] = bond_type
            self.bo_mat[atom2_idx, atom1_idx] = bond_type

            self.bo_dict[tuple(sorted([atom1_idx, atom2_idx]))] = bond_type

    def readfrommol2(self, filename, readstring=False):
        """
        Read mol2 into a mol3D class instance. Stores the bond orders and atom types (SYBYL).

        Parameters
        -------
            filename : string
                String of path to MOL2 file. Path may be local or global. May be read in as a string.
            readstring : bool
                Flag for deciding whether a string of mol2 file is being passed as the filename.
        """

        globs = globalvars()
        amassdict = globs.amass()
        graph = False
        bo_graph = False
        bo_dict = False
        if readstring:
            s = filename.splitlines()
        else:
            with open(filename, 'r') as f:
                s = f.read().splitlines()
        read_atoms = False
        read_bonds = False
        self.charge = 0
        for line in s:
            # Get atoms first.
            if '<TRIPOS>BOND' in line:
                read_atoms = False
            if '<TRIPOS>SUBSTRUCTURE' in line:
                read_bonds = False
            if '<TRIPOS>UNITY_ATOM_ATTR' in line:
                read_atoms = False
            if read_atoms:
                s_line = line.split()
                # Check redundancy in chemical symbols.
                atom_symbol1 = re.sub('[0-9]+[A-Z]+', '', line.split()[1])
                atom_symbol1 = re.sub('[0-9]+', '', atom_symbol1)
                atom_symbol2 = line.split()[5]
                if len(atom_symbol2.split('.')) > 1:
                    atype = atom_symbol2.split('.')[1]
                else:
                    atype = False
                atom_symbol2 = atom_symbol2.split('.')[0]
                if atom_symbol1 in list(amassdict.keys()):
                    atom = atom3D(atom_symbol1, [float(s_line[2]), float(
                        s_line[3]), float(s_line[4])], name=atype)
                elif atom_symbol2 in list(amassdict.keys()):
                    atom = atom3D(atom_symbol2, [float(s_line[2]), float(
                        s_line[3]), float(s_line[4])], name=atype)
                else:
                    print('Cannot find atom symbol in amassdict')
                    sys.exit()
                self.charge += float(s_line[8])
                self.partialcharges.append(float(s_line[8]))
                self.addAtom(atom)
            if '<TRIPOS>ATOM' in line:
                read_atoms = True
            if read_bonds:  # Read in bonds to molecular graph.
                s_line = line.split()
                graph[int(s_line[1]) - 1, int(s_line[2]) - 1] = 1
                graph[int(s_line[2]) - 1, int(s_line[1]) - 1] = 1
                if s_line[3] in ["ar"]:
                    bo_graph[int(s_line[1]) - 1, int(s_line[2]) - 1] = 1.5
                    bo_graph[int(s_line[2]) - 1, int(s_line[1]) - 1] = 1.5
                elif s_line[3] in ["am"]:
                    bo_graph[int(s_line[1]) - 1, int(s_line[2]) - 1] = 1
                    bo_graph[int(s_line[2]) - 1, int(s_line[1]) - 1] = 1
                elif s_line[3] in ["un"]:
                    bo_graph[int(s_line[1]) - 1, int(s_line[2]) - 1] = np.nan
                    bo_graph[int(s_line[2]) - 1, int(s_line[1]) - 1] = np.nan
                else:
                    bo_graph[int(s_line[1]) - 1, int(s_line[2]) - 1] = s_line[3]
                    bo_graph[int(s_line[2]) - 1, int(s_line[1]) - 1] = s_line[3]
                bo_dict[tuple(
                    sorted([int(s_line[1]) - 1, int(s_line[2]) - 1]))] = s_line[3]
            if '<TRIPOS>BOND' in line:
                read_bonds = True
                # Initialize molecular graph.
                graph = np.zeros((self.natoms, self.natoms))
                bo_graph = np.zeros((self.natoms, self.natoms))
                bo_dict = dict()
        if isinstance(graph, np.ndarray):  # Enforce mol2 molecular graph if it exists.
            self.graph = graph
            self.bo_mat = bo_graph
            self.bo_dict = bo_dict
        else:
            self.graph = np.array([])
            self.bo_mat = np.array([])
            self.bo_dict = {}

    @deprecated('Duplicate function will be removed in a future release.'
                'Use readfromxyz(readstring=True) instead.')
    def readfromstring(self, xyzstring):
        """
        Read XYZ from string.

        Parameters
        -------
            xyzstring : string
                String of XYZ file.
        """

        globs = globalvars()
        amassdict = globs.amass()
        self.graph = np.array([])
        s = xyzstring.split('\n')
        try:
            s.remove('')
        except ValueError:
            pass
        s = [f'{val}\n' for val in s]
        for line in s[0:]:
            line_split = line.split()
            if len(line_split) == 4 and line_split[0]:
                # This looks for unique atom IDs in files.
                lm = re.search(r'\d+$', line_split[0])
                # If the string ends in digits m will be a Match object, or None otherwise.
                if lm is not None:
                    symb = re.sub(r'\d+', '', line_split[0])
                    atom = atom3D(symb, [float(line_split[1]), float(line_split[2]), float(line_split[3])],
                                  name=line_split[0])
                elif line_split[0] in list(amassdict.keys()):
                    atom = atom3D(line_split[0], [float(line_split[1]), float(
                        line_split[2]), float(line_split[3])])
                else:
                    print('Cannot find atom type.')
                    sys.exit()
                self.addAtom(atom)

    def readfromtxt(self, txt):
        """
        Read XYZ from textfile.

        Parameters
        -------
            txt : list
                List of lists that comes as a result of readlines.
        """

        globs = globalvars()
        en_dict = globs.endict()
        self.graph = np.array([])
        for line in txt:
            line_split = line.split()
            if len(line_split) == 4 and line_split[0]:
                # This looks for unique atom IDs in files.
                lm = re.search(r'\d+$', line_split[0])
                # If the string ends in digits m will be a Match object, or None otherwise.
                if lm is not None:
                    symb = re.sub(r'\d+', '', line_split[0])
                    atom = atom3D(symb, [float(line_split[1]), float(line_split[2]), float(line_split[3])],
                                  name=line_split[0])
                elif line_split[0] in list(en_dict.keys()):
                    atom = atom3D(line_split[0], [float(line_split[1]), float(
                        line_split[2]), float(line_split[3])])
                else:
                    print('Cannot find atom type.')
                    sys.exit()
                self.addAtom(atom)

    def readfromxyz(self, filename: str, ligand_unique_id=False, read_final_optim_step=False, readstring=False):
        """
        Read XYZ into a mol3D class instance.

        Parameters
        -------
            filename : string
                String of path to XYZ file. Path may be local or global.
            ligand_unique_id : string
                Unique identifier for a ligand. In MR diagnostics, we abstract the atom based graph to a ligand based graph.
                For ligands, they don't have a natural name, so they are named with a UUID. Hard to attribute MR character to
                just atoms, so it is attributed ligands instead.
            read_final_optim_step : boolean
                if there are multiple geometries in the xyz file
                (after an optimization run) use only the last one.
            readstring : boolean
                Flag for deciding whether a string or xyz file is being passed as the filename.
        """

        globs = globalvars()
        amassdict = globs.amass()
        self.graph = np.array([])
        self.xyzfile = filename

        if readstring:
            s = filename.split('\n')
            try:
                s.remove('')
            except ValueError:
                pass
            s = [f'{val}\n' for val in s]
            for line in s[0:]:
                line_split = line.split()
                if len(line_split) == 4 and line_split[0]:
                    # This looks for unique atom IDs in files.
                    lm = re.search(r'\d+$', line_split[0])
                    # If the string ends in digits m will be a Match object, or None otherwise.
                    if lm is not None:
                        symb = re.sub(r'\d+', '', line_split[0])
                        atom = atom3D(symb, [float(line_split[1]), float(line_split[2]), float(line_split[3])],
                                      name=line_split[0])
                    elif line_split[0] in list(amassdict.keys()):
                        atom = atom3D(line_split[0], [float(line_split[1]), float(
                            line_split[2]), float(line_split[3])])
                    else:
                        print('Cannot find atom type.')
                        sys.exit()
                    self.addAtom(atom)
        else:
            with open(filename, 'r') as f:
                s = f.read().splitlines()
            try:
                atom_count = int(s[0])
            except ValueError:
                atom_count = 0
            start = 2
            if read_final_optim_step:
                start = len(s) - int(s[0])
            for line in s[start:start+atom_count]:
                line_split = line.split()
                # If the split line has more than 4 elements, only elements 0 through 3 will be used.
                # This means that it should work with any XYZ file that also stores something like Mulliken charge.
                # Next, this looks for unique atom IDs in files.
                if len(line_split):
                    lm = re.search(r'\d+$', line_split[0])
                    # If the string ends in digits m will be a Match object, or None otherwise.
                    if line_split[0] in list(amassdict.keys()) or ligand_unique_id:
                        atom = atom3D(line_split[0], [float(line_split[1]), float(
                            line_split[2]), float(line_split[3])])
                    elif lm is not None:
                        symb = re.sub(r'\d+', '', line_split[0])
                        atom = atom3D(symb, [float(line_split[1]), float(line_split[2]), float(line_split[3])],
                                      name=line_split[0])
                    else:
                        print('Cannot find atom type.')
                        sys.exit()
                    self.addAtom(atom)

    def writegxyz(self, filename):
        """
        Write GAMESS XYZ file.

        Parameters
        ----------
            filename : str
                Path to XYZ file.
        """

        ss = ''  # initialize returning string
        ss += "Date:" + time.strftime(
            '%m/%d/%Y %H:%M') + f", XYZ structure generated by mol3D Class, {self.globs.PROGRAM}\nC1\n"
        for atom in self.atoms:
            xyz = atom.coords()
            ss += f"{atom.sym} \t{float(atom.atno):.1f}\t{xyz[0]:.6f}\t{xyz[1]:.6f}\t{xyz[2]:.6f}\n"
        fname = filename.split('.gxyz')[0]
        with open(fname + '.gxyz', 'w') as f:
            f.write(ss)

    def writemol(self, filename):
        """
        Write mol file from mol3D object.
        Not advised if molecule has > 99 atoms.
        If there is no bond order information available,
        all bonds will be set as single bonds.

        Parameters
        ----------
            filename : str
                Path to mol file.
        """

        if not len(self.graph):
            # Set self.graph.
            self.createMolecularGraph()
        num_atoms = self.natoms
        num_bonds = int(np.count_nonzero(self.graph) / 2)

        mol_contents = [
        '',
        'Generated with molSimplify',
        '',
        f' {num_atoms} {num_bonds}  0  0  0  0  0  0  0  0999 V2000'
        ]

        # Atom section
        coords, syms = self.get_coordinate_array(), self.get_element_list()
        for coord, sym in zip(coords, syms):
            s = f' {coord[0]:9.4f} {coord[1]:9.4f} {coord[2]:9.4f} {sym.ljust(2)}  0  0  0  0  0  0  0  0  0  0  0  0'
            mol_contents.append(s)

        # Bond section
        # .mol files use 1 indexing.
        # Use bond order information if available
        # (self.bo_dict or self.bo_mat).
        if len(self.bo_dict) != 0:
            # If self.bo_dict is set, use that.
            bo_dict_keys = list(self.bo_dict.keys())
            bo_dict_keys.sort()
            for k in bo_dict_keys:
                v = self.bo_dict[k]
                s = f' {k[0]+1:2.0f} {k[1]+1:2.0f}  {v}  0  0  0  0'
                mol_contents.append(s)
        elif self.bo_mat.size != 0:
            # Only self.bo_mat is set, not self.bo_dict.
            rows, cols = np.nonzero(np.triu(self.bo_mat))
            for i, j in zip(rows, cols):
                s = f' {i+1:2.0f} {j+1:2.0f}  {int(self.bo_mat[i][j])}  0  0  0  0'
                mol_contents.append(s)
        else:
            # Make all bond orders be one.
            # Use triu since we only care about bonding pairs
            # where i < j.
            rows, cols = np.nonzero(np.triu(self.graph))
            for i, j in zip(rows, cols):
                s = f' {i+1:2.0f} {j+1:2.0f}  1  0  0  0  0'
                mol_contents.append(s)

        mol_contents.extend(['M  END', ''])
        mol_contents = '\n'.join(mol_contents)

        with open(filename, 'w') as f:
            f.write(mol_contents)

    def writemol2(self, filename, writestring=False, ignoreX=False, force=False):
        """
        Write mol2 file from mol3D object. Partial charges are appended if given.
        Else, total charge of the complex (given or interpreted by OBMol) is assigned
        to the metal.

        Parameters
        ----------
            filename : str
                Path to mol2 file.
            writestring : bool, optional
                Flag to write to a string if True or file if False. Default is False.
            ignoreX : bool, optional
                Flag to delete atom X. Default is False.
            force : bool, optional
                Flag to dictate if bond orders are written (obmol/assigned) or =1.
        """

        if ignoreX:
            for i, atom in enumerate(self.atoms):
                if atom.sym == 'X':
                    self.deleteatom(i)
        if not len(self.graph):
            self.createMolecularGraph()
        if not self.bo_dict and not force:
            self.convert2OBMol2()
        csg = csgraph.csgraph_from_dense(self.graph)
        disjoint_components = csgraph.connected_components(csg)
        if disjoint_components[0] > 1:
            atom_group_names = [f'RES{x+1}' for x in disjoint_components[1]]
            atom_groups = [str(x+1) for x in disjoint_components[1]]
        else:
            atom_group_names = ['RES1']*self.natoms
            atom_groups = [str(1)]*self.natoms
        atom_types = list(set(self.symvect()))
        atom_type_numbers = np.ones(len(atom_types))
        atom_types_mol2 = []
        try:
            metal_ind = self.findMetal()[0]
        except IndexError:
            metal_ind = 0
        if len(self.partialcharges):
            charges = self.partialcharges
            charge_string = 'PartialCharges'
        elif self.charge:  # Assign total charge to metal.
            charges = np.zeros(self.natoms)
            charges[metal_ind] = self.charge
            charge_string = 'UserTotalCharge'
        else:  # Calculate total charge with OBMol, assign to metal.
            if self.OBMol:
                charges = np.zeros(self.natoms)
                charges[metal_ind] = self.OBMol.GetTotalCharge()
                charge_string = 'OBmolTotalCharge'
            else:
                charges = np.zeros(self.natoms)
                charge_string = 'ZeroCharges'
        ss = f'@<TRIPOS>MOLECULE\n{filename}\n'
        ss += f'{self.natoms}\t{int(csg.nnz/2)}\t{disjoint_components[0]}\n'
        ss += 'SMALL\n'
        ss += charge_string + '\n' + '****\n' + 'Generated from molSimplify\n\n'
        ss += '@<TRIPOS>ATOM\n'
        atom_default_dict = {'C': '3', 'N': '3', 'O': '2', 'S': '3', 'P': '3'}
        for i, atom in enumerate(self.atoms):
            if atom.name != atom.sym:
                atom_types_mol2 = '.'+atom.name
            elif atom.sym in list(atom_default_dict.keys()):
                atom_types_mol2 = '.' + atom_default_dict[atom.sym]
            else:
                atom_types_mol2 = ''
            type_ind = atom_types.index(atom.sym)
            atom_coords = atom.coords()
            ss += f'{i+1} {atom.sym}{int(atom_type_numbers[type_ind])}\t' + \
                f'{atom_coords[0]}  {atom_coords[1]}  {atom_coords[2]} ' + \
                f'{atom.sym}{atom_types_mol2}\t{atom_groups[i]}' + \
                f' {atom_group_names[i]} {charges[i]}\n'
            atom_type_numbers[type_ind] += 1
        ss += '@<TRIPOS>BOND\n'
        bonds = csg.nonzero()
        bond_count = 1
        if self.bo_dict:
            bondorders = True
        else:
            bondorders = False
        for i, b1 in enumerate(bonds[0]):
            b2 = bonds[1][i]
            if b2 > b1 and not bondorders:
                ss += f'{bond_count} {b1+1} {b2+1} 1\n'
                bond_count += 1
            elif b2 > b1 and bondorders:
                ss += f'{bond_count} {b1+1} {b2+1} {self.bo_dict[(int(b1), int(b2))]}\n'
        ss += '@<TRIPOS>SUBSTRUCTURE\n'
        unique_group_names = np.unique(atom_group_names)
        for i, name in enumerate(unique_group_names):
            ss += f'{i+1} {name} {atom_group_names.count(name)}\n'
        ss += '\n'
        if writestring:
            return ss
        else:
            if '.mol2' not in filename:
                if '.' not in filename:
                    filename += '.mol2'
                else:
                    filename = filename.split('.')[0]+'.mol2'
            with open(filename, 'w') as file1:
                file1.write(ss)

    def writemol2_bodict(
            self,
            ignore_dummy_atoms=True,
            write_bond_orders=True,
            return_string=True,
            output_file=None
        ):
        """
        Generate a MOL2-format string or file from atomic coordinates and bonding data.

        Parameters
        ----------
            ignore_dummy_atoms : bool, optional (default=True)
                If True, atoms with element symbol 'X' will be ignored in both atoms and bonds.
            write_bond_orders : bool, optional (default=True)
                If True, writes the actual bond orders from `bond_order_dict`.
                If False, all bonds are assigned order '1'.
            return_string : bool, optional (default=True)
                If True, returns the MOL2 content as a string.
                If False, writes to `output_file`.
            output_file : str or None, optional
                If `return_string` is False, this must be the path to the file to write.

        Returns
        -------
            str or None
                Returns the MOL2-format string if `return_string` is True, otherwise writes to file
                and returns None.

        Notes
        -----
            - Atoms are renumbered starting from 1.
            - Element-based labels (e.g., C1, C2) are assigned using counts per element.
            - Substructures are inferred using connected components in the bond graph.
            - Only bonds where both atoms are not dummy atoms are retained if `ignore_dummy_atoms` is True.
        """

        # Filter out dummy atoms
        filtered_atoms = []
        index_map = {}
        counter_by_element = {}
        new_index = 1

        # get the atoms
        atom_coords = []
        atom_elements = []
        for atom in self.atoms:
            atom_coords.append(atom.coords())
            atom_elements.append(atom.sym)

        # get bond_order dictionary
        bond_order_dict = self.bo_dict

        for i, (coord, elem) in enumerate(zip(atom_coords, atom_elements)):
            if ignore_dummy_atoms and elem.upper() == 'X':
                continue
            elem_clean = elem.capitalize()
            counter_by_element.setdefault(elem_clean, 0)
            counter_by_element[elem_clean] += 1
            atom_label = f"{elem_clean}{counter_by_element[elem_clean]}"
            filtered_atoms.append((new_index, atom_label, coord, elem_clean))
            index_map[i] = new_index
            new_index += 1

        # Rebuild bond list using new indices
        bonds = []
        for (i, j), order in bond_order_dict.items():
            if i in index_map and j in index_map:
                idx1 = index_map[i]
                idx2 = index_map[j]
                bond_type = order if write_bond_orders else '1'
                bonds.append((idx1, idx2, bond_type))

        # Determine substructures using NetworkX
        G = nx.Graph()
        G.add_nodes_from([idx for idx, _, _, _ in filtered_atoms])
        G.add_edges_from([(i, j) for i, j, _ in bonds])
        components = list(nx.connected_components(G))
        substructure_lookup = {}
        for idx, comp in enumerate(components, start=1):
            for atom_idx in comp:
                substructure_lookup[atom_idx] = idx

        # Compose mol2 content
        mol2_lines = []
        mol2_lines.append("@<TRIPOS>MOLECULE")
        mol2_lines.append("GeneratedMol")
        mol2_lines.append(f"{len(filtered_atoms)} {len(bonds)} 1")
        mol2_lines.append("SMALL")
        mol2_lines.append("NO_CHARGES\n")

        mol2_lines.append("@<TRIPOS>ATOM")
        for idx, label, coord, elem in filtered_atoms:
            mol2_lines.append(f"{idx:<5d} {label:<6s} {coord[0]:>10.4f} {coord[1]:>10.4f} {coord[2]:>10.4f} {elem:<4s} {substructure_lookup[idx]:>2d} RES{substructure_lookup[idx]} 0.0000")

        mol2_lines.append("@<TRIPOS>BOND")
        for i, (idx1, idx2, bond_type) in enumerate(bonds, start=1):
            mol2_lines.append(f"{i:<5d} {idx1:<4d} {idx2:<4d} {bond_type}")

        mol2_lines.append("@<TRIPOS>SUBSTRUCTURE")
        for sub_id in sorted(set(substructure_lookup.values())):
            mol2_lines.append(f"{sub_id:<3d} RES{sub_id}       1 TEMP              0 ****  ****    0 ROOT")

        mol2_str = '\n'.join(mol2_lines)

        if return_string:
            return mol2_str
        elif output_file:
            with open(output_file, 'w') as f:
                f.write(mol2_str)
        else:
            raise ValueError("Must specify either return_string=True or output_file='filename.mol2'")

    def writemxyz(self, mol, filename, no_tabs=False):
        """
        Write standard XYZ file with two molecules.

        Parameters
        ----------
            mol : mol3D
                mol3D instance of second molecule.
            filename : str
                Path to XYZ file.
            no_tabs : bool, optional
                Whether or not to use tabs in coordinate columns.
        """

        ss = ''  # initialize returning string
        ss += f'{self.natoms + mol.natoms}\n' + time.strftime(
            '%m/%d/%Y %H:%M') + f', XYZ structure generated by mol3D Class, {self.globs.PROGRAM}\n'
        for atom in self.atoms:
            xyz = atom.coords()
            ss += f"{atom.sym} \t{xyz[0]:.6f}\t{xyz[1]:.6f}\t{xyz[2]:.6f}\n"
        for atom in mol.atoms:
            xyz = atom.coords()
            ss += f"{atom.sym} \t{xyz[0]:.6f}\t{xyz[1]:.6f}\t{xyz[2]:.6f}\n"
        if no_tabs:
            ss = ss.replace('\t', ' ' * 8)
        fname = filename.split('.xyz')[0]
        with open(fname + '.xyz', 'w') as f:
            f.write(ss)

    def writenumberedxyz(self, filename):
        """
        Write standard XYZ file with numbers instead of symbols.

        Parameters
        ----------
            filename : str
                Path to XYZ file.
        """

        ss = ''  # Initialize returning string.
        ss += f'{self.natoms}\n' + time.strftime(
            '%m/%d/%Y %H:%M') + f', XYZ structure generated by mol3D Class, {self.globs.PROGRAM}\n'
        unique_types = dict()

        for atom in self.atoms:
            this_sym = atom.symbol()
            if this_sym not in list(unique_types.keys()):
                unique_types.update({this_sym: 1})
            else:
                unique_types.update({this_sym: unique_types[this_sym] + 1})
            atom_name = f'{atom.symbol()}{unique_types[this_sym]}'
            xyz = atom.coords()
            ss += f"{atom_name} \t{xyz[0]:.6f}\t{xyz[1]:.6f}\t{xyz[2]:.6f}\n"
        fname = filename.split('.xyz')[0]
        with open(fname + '.xyz', 'w') as f:
            f.write(ss)

    def writesepxyz(self, mol, filename):
        """
        Write standard XYZ file with two molecules separated.

        Parameters
        ----------
            mol : mol3D
                mol3D instance of second molecule.
            filename : str
                Path to XYZ file.
        """

        ss = ''  # Initialize returning string.
        ss += f'{self.natoms}\n' + time.strftime(
            '%m/%d/%Y %H:%M') + f', XYZ structure generated by mol3D Class, {self.globs.PROGRAM}\n'
        for atom in self.atoms:
            xyz = atom.coords()
            ss += f"{atom.sym} \t{xyz[0]:.6f}\t{xyz[1]:.6f}\t{xyz[2]:.6f}\n"
        ss += f'--\n{mol.natoms}\n\n'
        for atom in mol.atoms:
            xyz = atom.coords()
            ss += f"{atom.sym} \t{xyz[0]:.6f}\t{xyz[1]:.6f}\t{xyz[2]:.6f}\n"
        fname = filename.split('.xyz')[0]
        with open(fname + '.xyz', 'w') as f:
            f.write(ss)

    def writexyz(self, filename, symbsonly=True, ignoreX=False,
                  ordering=False, writestring=False, withgraph=False,
                  specialheader=False, no_tabs=False):
        """
        Write standard XYZ file.

        Parameters
        ----------
            filename : str
                Path to XYZ file.
            symbsonly : bool, optional
                Only write symbols to file. Default is True.
            ignoreX : bool, optional
                Ignore X element when writing. Default is False.
            ordering : bool, optional
                If handed a list, will order atoms in a specific order. Default is False.
            writestring : bool, optional
                Flag to write to a string if True or file if False. Default is False.
            withgraph : bool, optional
                Flag to write with graph (after XYZ) if True. Default is False.
                If True, sparse graph written. All bonds indicated as single.
            specialheader : str, optional
                String to write information into header. Default is False. If True, a special string is written.
            no_tabs : bool, optional
                Whether or not to use tabs in coordinate columns.

        Returns
        -------
            ss : str
                XYZ contents, if writestring is True.
        """

        ss = ''  # Initialize returning string.
        natoms = self.natoms
        if not ordering:
            ordering = list(range(natoms))
        if ignoreX:
            natoms -= sum([1 for i in self.atoms if i.sym == "X"])

        if specialheader:
            ss += f'{natoms}\n'
            ss += f'{specialheader}\n'
        else:
            ss += f'{natoms}\n' + time.strftime(
                '%m/%d/%Y %H:%M') + f', XYZ structure generated by mol3D Class, {self.globs.PROGRAM}\n'
        for ii in ordering:
            atom = self.getAtom(ii)
            if not (ignoreX and atom.sym == 'X'):
                xyz = atom.coords()
                if symbsonly:
                    ss += f"{atom.sym} \t{xyz[0]:.6f}\t{xyz[1]:.6f}\t{xyz[2]:.6f}\n"
                else:
                    ss += f"{atom.name} \t{xyz[0]:.6f}\t{xyz[1]:.6f}\t{xyz[2]:.6f}\n"
        if withgraph:
            if not len(self.graph):
                self.createMolecularGraph()

            csg = csgraph.csgraph_from_dense(self.graph)
            x, y = csg.nonzero()
            tempstr = ''
            for row1, row2 in zip(x, y):
                tempstr += str(row1).rjust(4)
                tempstr += str(row2).rjust(5)
                tempstr += ' S\n' # Indicate all bonds as single.
            ss += tempstr

        if no_tabs:
            ss = ss.replace('\t', ' ' * 8)

        if writestring:
            return ss
        else:
            fname = filename.split('.xyz')[0]
            with open(fname + '.xyz', 'w') as f:
                f.write(ss)
