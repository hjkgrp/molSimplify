import numpy as np

class Mol3DGeometry:
    """
    Mixin class for mol3D containing geometric manipulation routines.
    """
    def get_coords_matrix(self) -> np.ndarray:
        """
        Returns an (N, 3) numpy array of all atom coordinates.

        Returns
        -------
            coords : np.ndarray
                (N, 3) array of coordinates.
        """
        if not self.atoms:
            return np.empty((0, 3))
        return np.array([atom.coords() for atom in self.atoms])

    def get_mass_vector(self) -> np.ndarray:
        """
        Returns an (N,) numpy array of all atom masses.

        Returns
        -------
            masses : np.ndarray
                (N,) array of atomic masses.
        """
        if not self.atoms:
            return np.empty((0,))
        return np.array([atom.mass for atom in self.atoms])

    def centermass(self):
        """
        Computes coordinates of center of mass of molecule.
        Vectorized implementation.

        Returns
        -------
            center_of_mass : list
                Coordinates of center of mass. List of length 3: (X, Y, Z).
        """
        if self.natoms > 0:
            coords = self.get_coords_matrix()
            masses = self.get_mass_vector()
            total_mass = np.sum(masses)
            if total_mass > 0:
                # Weighted average of coordinates
                cm = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
                return list(cm)

        print('ERROR: Center of mass calculation failed. Structure will be inaccurate.\n')
        return False

    def coordsvect(self):
        """
        Method to obtain array of coordinates in molecule.
        Vectorized implementation.

        Returns
        -------
            list_of_coordinates : np.array
                Two dimensional numpy array of molecular coordinates.
                (N by 3) dimension, N is number of atoms.
        """
        return self.get_coords_matrix()

    def centersym(self):
        """
        Computes coordinates of center of symmetry of molecule.
        Vectorized implementation.

        Returns
        -------
            center_of_symmetry : list
                Coordinates of center of symmetry. List of length 3: (X, Y, Z).
        """
        if self.natoms > 0:
            coords = self.get_coords_matrix()
            cs = np.mean(coords, axis=0)
            return list(cs)

        print('ERROR: Center of symmetry calculation failed. Structure will be inaccurate.\n')
        return False
