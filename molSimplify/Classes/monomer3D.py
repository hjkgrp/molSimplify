# @file monomer3D.py
#  Defines monomer3D class and contains useful manipulation/retrieval routines.
#
#  Written by HJK Group
#
#  Dpt of Chemical Engineering, MIT


class monomer3D:
    """Holds information about a monomer, used to do manipulations.
    Reads information from structure file (pdb) or is directly built from molsimplify.

    """

    def __init__(self, three_lc='GLY', chain='undef', id=-1, occup=1.00, loc=''):
        # List of atom3D objects
        self.atoms = []
        # Number of atoms
        self.natoms = 0
        # 3-letter code of amino acid (in all caps); alternatively, 1-letter code of nucleotide
        self.three_lc = three_lc  # if no name is specified, defaults to glycine
        # Mass of molecule
        self.mass = 0
        # Size of molecule
        self.size = 0
        # Chain of monomer
        self.chain = chain
        # ID of monomer (position in chain)
        self.id = id
        # Occupancy of monomer in chain
        self.occup = occup
        # Peptide bond atoms
        self.c = []
        self.n = []
        # Bonds
        self.bonds = {}
        # Next monomer in the chain
        self.next = None
        # Previous monomer in the chain
        self.prev = None
        # Location (if more than one conformation)
        self.loc = loc
        # Temporary list for storing conformations
        self.temp_list = []

    def __repr__(self):
        return f"monomer3D({self.three_lc}, id={self.id})"

    def identify(self):
        """ States whether the amino acid is (positively/negatively) charged, polar, or hydrophobic.

        Returns
        -------
        aa_type : string
            Positively charged, Negatively charged, Polar, Hydrophobic

        """
        if self.three_lc == "ARG" or self.three_lc == "LYS":
            return "Positively charged"
        elif self.three_lc == "ASP" or self.three_lc == "GLU":
            return "Negatively charged"
        elif self.three_lc in {"GLN", "ASN", "HIS", "SER", "THR", "TYR", "CYS"}:
            return "Polar"
        else:
            return "Hydrophobic"

    def getGreek(self, greek):
        """ Finds the Greek lettered carbon(s) or other atom(s) of the user's choice.

        Parameters
        ----------
        greek : string
            The Greek lettered atom (e.g. alpha carbon) we want.  Inputs should be form 'CA' or similar.

        Returns
        -------
        greek_atoms : list of atom3Ds
            A list of atom3D class objects that contains the Greek lettered atom(s) we want.

        """
        greek_atoms = []
        for (ii, a) in self.atoms:
            if greek in a.greek:
                greek_atoms.append(a)
        return greek_atoms

    def coords(self):
        """Method to obtain string of coordinates in monomer.

        Returns
        -------
            coord_string : string
                String of molecular coordinates with atom identities in XYZ format.
        """
        coord_string = ''  # initialize returning string
        coord_string += "%d \n\n" % self.natoms
        for (ii, atom) in self.atoms:
            xyz = atom.coords()
            coord_string += "%s \t%f\t%f\t%f\n" % (atom.sym, xyz[0], xyz[1], xyz[2])
        return coord_string

    def centermass(self):
        """Computes coordinates of center of mass of monomer.
        Returns
        -------
            center_of_mass : list
                Coordinates of center of mass. List of length 3: (X, Y, Z).
        """

        center_of_mass = [0, 0, 0]
        mmass = 0
        # loop over atoms in molecule
        if self.natoms > 0:
            for (ii, atom) in self.atoms:
                # calculate center of mass (relative weight according to atomic mass)
                xyz = atom.coords()
                center_of_mass[0] += xyz[0] * atom.mass
                center_of_mass[1] += xyz[1] * atom.mass
                center_of_mass[2] += xyz[2] * atom.mass
                mmass += atom.mass
            # normalize
            center_of_mass[0] /= mmass
            center_of_mass[1] /= mmass
            center_of_mass[2] /= mmass
        else:
            center_of_mass = False
            print(
                'ERROR: Center of mass calculation failed. Structure will be inaccurate.\n')
        return center_of_mass

    def centroid(self):
        """Computes coordinates of centroid of monomer.
        Returns
        -------
            centroid : list
                Coordinates of centroid. List of length 3: (X, Y, Z).
        """

        centroid = [0, 0, 0]
        # loop over atoms in molecule
        if self.natoms > 0:
            for (ii, atom) in self.atoms:
                # calculate center of mass (relative weight according to atomic mass)
                xyz = atom.coords()
                centroid[0] += xyz[0]
                centroid[1] += xyz[1]
                centroid[2] += xyz[2]
        else:
            centroid = False
            print(
                'ERROR: Centroid calculation failed. Structure will be inaccurate.\n')
        return centroid

    def setBonds(self):
        """ Sets the bonds between atoms within the monomer.

        """
        # bonds :  dictionary (represents an undirected graph)
        #        Keyed by atom3D atoms in the monomer
        #        Valued by a set consisting of bonded atoms
        for (ii, key) in self.atoms:
            if key not in self.bonds.keys():
                self.bonds[key] = set()
                if key.greek == 'C':
                    for (a_id, a) in self.atoms:
                        if a.greek == 'O' or a.greek == "CA":
                            self.bonds[key].add(a)
                            if a not in self.bonds.keys():
                                self.bonds[a] = set()
                            self.bonds[a].add(key)
                if key.greek == 'CA':
                    for (a_id, a) in self.atoms:
                        if a.greek == 'CB' or a.greek == "N":
                            self.bonds[key].add(a)
                            if a not in self.bonds.keys():
                                self.bonds[a] = set()
                            self.bonds[a].add(key)
                if "G" in key.greek:
                    for (a_id, a) in self.atoms:
                        if 'B' in a.greek or "D" in a.greek:
                            self.bonds[key].add(a)
                            if a not in self.bonds.keys():
                                self.bonds[a] = set()
                            self.bonds[a].add(key)
                if "E" in key.greek:
                    for (a_id, a) in self.atoms:
                        if 'Z' in a.greek or "D" in a.greek:
                            self.bonds[key].add(a)
                            if a not in self.bonds.keys():
                                self.bonds[a] = set()
                            self.bonds[a].add(key)

    def getPeptideAtoms(self):
        """ Makes the atoms involved in peptide bonding attributes.

        """
        for (ii, a) in self.atoms:
            if a.greek == "C":
                self.c = a
            elif a.greek == "N":
                self.n = a

    def setNext(self, next_aa):
        """ Sets the next monomer in the chain of a protein (if applicable).

        Parameters
        ----------
            next_aa : monomer3D
                the monomer that is next in the protein chain
        """
        self.next = next_aa

    def setPrev(self, prev_aa):
        """ Sets the previous monomer in the chain of a protein (if applicable).

        Parameters
        ----------
            prev_aa : monomer3D
                the monomer that comes before self in the protein chain
        """
        self.prev = prev_aa

    def addAtom(self, atom, index=None):
        """Adds an atom to the atoms attribute, which contains a list of
        atom3D class instances.

        Parameters
        ----------
            atom : atom3D
                atom3D class instance of added atom.
            index : int, optional
                Index of added atom. Default is None.
            auto_populate_BO_dict : bool, optional
                Populate bond order dictionary with newly added atom. Default is True.
        """

        if index is None:
            index = len(self.atoms)
        # self.atoms.append(atom)
        self.atoms.append((index, atom))
        self.natoms += 1
        self.mass += atom.mass
        self.metal = False

    def setLoc(self, loc):
        """ Sets the conformation of a monomer in the chain of a protein.

        Parameters
        ----------
            loc : str
                a one-character string representing the conformation
        """
        self.loc = loc
