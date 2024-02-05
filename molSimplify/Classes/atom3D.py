# @file atom3D.py
#  Defines atom3D class and contains useful manipulation/retrieval routines.
#
#  Written by Kulik Group
#
#  Department of Chemical Engineering, MIT

from math import sqrt
from molSimplify.Classes.globalvars import globalvars


class atom3D:
    """ atom3D class. Base class in molSimplify for representing an element.

        Parameters
        ----------
            Sym : str, optional
                Symbol for atom3D instantiation. Element symbol. Default is 'C'.
            xyz : list, optional
                List of coordinates for new atom. Default is [0.0, 0.0, 0.0].
            name : str, optional
                Unique identifier for atom 3D instantiation. Default is False.
            partialcharge : int, optional
                Charge assigned to atom when added to mol. Default is None.
    """
    def __init__(self, Sym='C', xyz=None, name=False, partialcharge=None, Tfactor=0, greek='', occup=1.00, loc='', line=""):

        # Element symbol
        self.sym = Sym
        self.partialcharge = None
        globs = globalvars()
        amass = globs.amass()
        if Sym not in amass:  # assign default values if not in dictionary
            print(("We didn't find the atomic mass of %s in the dictionary. Assigning default value of 12!\n" % (Sym)))
            # Atomic mass
            self.mass = 12  # default atomic mass
            # Atomic number
            self.atno = 6  # default atomic number
            # Covalent radius
            self.rad = 0.75  # default atomic radius
        else:
            self.mass = amass[Sym][0]
            self.atno = amass[Sym][1]
            self.rad = amass[Sym][2]
        # Flag for freezing in optimization
        self.frozen = False
        # Flag for atom name
        if name:
            self.name = name
        else:
            self.name = Sym

        # Coordinates
        if xyz is None:
            xyz = [0.0, 0.0, 0.0]
        self.__xyz = xyz

        # Temperature factor (only useful for proteins)
        self.Tfactor = Tfactor

        # Greek letter (e.g. alpha carbon - only useful for proteins)
        if greek == '':
            self.greek = Sym
        else:
            self.greek = greek

        # Occupancy (only useful for proteins)
        self.occup = occup

        # EDIA score (only useful for proteins)
        self.EDIA = 0

        # Conformation (only useful for proteins)
        self.loc = ""

        # PDB line (only useful for proteins)
        self.line = line

    def __repr__(self):
        """Returns all bound methods of the mol3D class..

        Returns
        -------
            method_string : string
                String of methods available in mol3D class.
        """

        method_string = "\nClass atom3D has the following methods:\n"
        for method in dir(self):
            if callable(getattr(self, method)):
                method_string += method + '\n'
        return method_string

    def coords(self):
        """ Get coordinates of a given atom.

        Returns
        -------
            coords : list
                List of coordinates in X, Y, Z format.
        """
        x, y, z = self.__xyz
        return [x, y, z]

    def distance(self, atom2):
        """ Get distance from one atom3D class to another.

        Parameters
        ----------
            atom2 : atom3D
                atom3D class of the atom to measure distance from.


        Returns
        -------
            dist : float
                Distance in angstroms.
        """
        xyz = self.coords()
        point = atom2.coords()
        dx = xyz[0]-point[0]
        dy = xyz[1]-point[1]
        dz = xyz[2]-point[2]
        return sqrt(dx*dx+dy*dy+dz*dz)

    def distancev(self, atom2):
        """ Get distance vector from one atom3D class to another.

        Parameters
        ----------
            atom2 : atom3D
                atom3D class of the atom to measure distance from.


        Returns
        -------
            dist_list : list
                List of distances in vector form: [dx, dy, dz] with units of Angstroms.
        """
        xyz = self.coords()
        point = atom2.coords()
        dx = xyz[0]-point[0]
        dy = xyz[1]-point[1]
        dz = xyz[2]-point[2]
        return [dx, dy, dz]

    def ismetal(self, transition_metals_only=True):
        """ Identify whether an atom is a metal.

        Parameters
        ----------
            transition_metals_only : bool, optional
                Identify only transition metals. Default is true.

        Returns
        -------
            metal : bool
                Bool for whether or not an atom is a metal.
        """
        return self.sym in globalvars().metalslist(transition_metals_only=transition_metals_only)

    def setcoords(self, xyz):
        """ Set coordinates of an atom3D class to a new location.

        Parameters
        ----------
            xyz : list
                List of coordinates, has length 3: [X, Y, Z]
        """
        self.__xyz[0] = xyz[0]
        self.__xyz[1] = xyz[1]
        self.__xyz[2] = xyz[2]

    def symbol(self):
        """ Return symbol of atom3D.

        Returns
        -------
            symbol : str
                Element symbol for atom3D class.
        """
        return self.sym

    def mutate(self, newType='C'):
        """ Mutate an element to another element in the atom3D.

        Parameters
        ----------
            newType : str, optional
                Element name for new element. Default is 'C'.

        """
        globs = globalvars()
        amass = globs.amass()
        if newType not in list(amass.keys()):
            print(('Error, unknown atom atom type transformation to ' + str(newType)))
            print('no changes made')
        else:
            self.mass = amass[newType][0]
            self.atno = amass[newType][1]
            self.rad = amass[newType][2]
            self.name = newType
            self.sym = newType

    def translate(self, dxyz):
        """ Move the atom3D by a displacement vector.

        Parameters
        ----------
            dxyz : list
                Displacement vector of length 3: [dx, dy, dz].

        """
        x, y, z = self.__xyz
        self.__xyz[0] = x + dxyz[0]
        self.__xyz[1] = y + dxyz[1]
        self.__xyz[2] = z + dxyz[2]

    def setEDIA(self, score):
        """ Sets the EDIA score of an individual atom3D.

        Parameters
        ----------
            score : float
                Desired EDIA score of atom
        """
        self.EDIA = score
