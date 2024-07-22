import numpy as np

def moments_of_inertia(coords, masses):
    """
    Returns the moment of inertia tensor, given the atomic positions and masses.
    Should be centered around the center of mass before this function is used.

    Parameters
    ----------
        coords : np.array
            Natoms x 3 array giving the atomic coordinates
        masses : np.array
            Natoms x 1 array giving the atomic masses

    Returns
    -------
        I : np.array
            3 x 3 array giving the moments of inertia tensor
    """
    I = np.zeros((3, 3))
    for i in range(len(masses)):
        x, y, z = coords[i]
        mass = masses[i]
        I[0, 0] += mass * (y**2 + z**2) #xx
        I[1, 1] += mass * (x**2 + z**2) #yy
        I[2, 2] += mass * (x**2 + y**2) #zz
        I[0, 1] -= mass * (x * y) #xy
        I[0, 2] -= mass * (z * x) #xz
        I[1, 2] -= mass * (y * z) #yz

    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]
    return I

def principal_axes(coords, masses):
    """
    Returns the eigenvectors defining the principal axes.

    Parameters
    ----------
        coords : np.array
            Natoms x 3 array giving the atomic coordinates
        masses : np.array
            Natoms x 1 array giving the atomic masses
    Returns
    -------
        eigvecs : np.array
            3x3 array where each row is an eigenvector,
            sorted from lowest to highest eigenvalue.
    """
    I = moments_of_inertia(coords, masses)
    #diagonalize the moments of inertia
    eigvals, eigvecs = np.linalg.eigh(I)

    #sort the eigenvectors from lowest to highest, store so that eigvecs[i] is the ith eigenvector
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order].T

    return eigvecs

def rot_mat_from_angle_axis(angle, axis):
    """
    Returns a Cartesian rotation matrix from an angle and rotation axis.

    Parameters
    ----------
        angle : float
            Angle (in radians) that one wants to rotate by
        axis : np.array
            length-3 vector defining the axis to rotate about

    Returns
    -------
        R : np.array
            3x3 rotation matrix that performs the rotation upon left-multiplication.
    """
    norm_axis = axis / np.linalg.norm(axis)
    C = np.cos(angle)
    S = np.sin(angle)
    U = 1 - np.cos(angle)
    ax, ay, az = norm_axis
    R = np.array([
        [U*ax**2 + C, U*ax*ay - S*az, U*ax*az + S*ay],
        [U*ax*ay + S*az, U*ay**2 + C, U*ay*az - S*ax],
        [U*ax*az - S*ay, U*ay*az + S*ax, U*az**2 + C]
    ])
    return R

def rotate_onto_principal_axes(mol):
    """
    Given a mol3D object, returns natoms x 3 array that is
    aligned to have its largest moment of inertia on the positive z
    axis, next largest on the positive y axis.

    Does so through rotations, which preserve chirality.

    Parameters
    ----------
        mol : mol3D
            mol3D instance.
    Returns
    -------
        coords : np.array
            n x 3 array of the coordinates in the mol.
    """
    #Center on the center of mass
    coords = np.array(mol.coordsvect()) - np.array(mol.centermass())

    #Get masses
    masses = np.array([atom.mass for atom in mol.getAtoms()])

    possible_coords = []
    #outer loop to catch possible alignments
    for iter in range(4):
        #align the z, y axes
        for idx in [-1, -2]:
            #define the positive axis you want to align
            axis_vec = np.zeros(3)
            axis_vec[idx] = 1

            #get the normalized eigenvector of interest
            eigvecs = principal_axes(coords, masses)
            eigvec = eigvecs[idx] / np.linalg.norm(eigvecs[idx])

            #get the angle between the coordinate and principal axis
            angle = np.arccos(np.dot(axis_vec, eigvec))
            if iter == 1 and idx == -1:
                angle = -angle
            elif iter == 2 and idx == -2:
                angle = -angle
            elif iter == 3:
                angle = -angle

            #get the axis to rotate about
            rot_axis = np.cross(axis_vec, eigvec)
            if np.isclose(np.linalg.norm(rot_axis), 0):
                #if the two axes are already parallel, rotate about x
                rot_axis = np.array([1, 0, 0])
            else:
                rot_axis = rot_axis / np.linalg.norm(rot_axis)

            #build the rotation matrix
            rot_mat = rot_mat_from_angle_axis(angle, rot_axis)

            #redefine the coordinates
            coords = np.vstack([rot_mat @ coords[i, :] for i in range(len(coords))])

        possible_coords.append(coords)

    proper_coords = None
    min_error = np.inf

    for coords in possible_coords:
        for idx, atom in enumerate(mol.atoms):
            atom.setcoords(coords[idx, :])
        I = mol.moments_of_inertia()
        off_diag = I - np.diag(np.diag(I))
        error = np.sum(np.abs(off_diag))
        if error < min_error:
            proper_coords = coords

    return proper_coords


def project_onto_principal_axes(mol):
    """
    Given a mol3D object, returns natoms x 3 array that is
    aligned to have its largest moment of inertia on the positive z
    axis, next largest on the positive y axis.

    Accomplishes this by projecting onto the eigenvectors of the moment of inertia
    tensor, which may result in reflections / other undesired transformations.

    Parameters
    ----------
        mol : mol3D
            mol3D instance.
    Returns
    -------
        coords : np.array
            n x 3 array of the coordinates in the mol.
    """
    #Center on the center of mass
    coords = np.array(mol.coordsvect()) - np.array(mol.centermass())
    #Get the transformation matrix (the inverse of the eigenvectors)
    eigvals, eigvecs = mol.principal_moments_of_inertia(return_eigvecs=True)
    P = np.linalg.inv(eigvecs)
    #Get the coordinates where the principal axes have been rotated onto the coordinate axes
    coords = np.vstack([P @ coords[i, :] for i in range(len(coords))])
    #Reorder the columns so that the largest axis is in the z spot
    coords = coords[:, np.argsort(eigvals)]

    return coords
