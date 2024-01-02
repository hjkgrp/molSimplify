# @file geometry.py
#  Contains many useful 3D Euclidean geometric manipulation routines.
#
#  Unless otherwise stated, all "points" refer to 3-element lists.
#
#  Written by Kulik Group
#
#  Department of Chemical Engineering, MIT

import numpy as np


def norm(u):
    """Get euclidean norm of vector.

        Parameters
        ----------
            u : list
                Vector of interest.

        Returns
        -------
            norm : float
                Norm of u.

    """
    d = 0.0
    for u0 in u:
        d += (u0 * u0)
    d = np.sqrt(d)
    return d


def normalize(u):
    """Normalize a vector.

        Parameters
        ----------
            u : list
                Vector of interest.

        Returns
        -------
            norm_vect : list
                Normalized vector.

    """
    d = norm(u)
    un = []
    if d > 1.0e-13:
        [un.append(ui / d) for ui in u]
    return un


def distance(r1, r2):
    """Euclidean distance between points.

        Parameters
        ----------
            r1 : list
                Coordinates of point 1.
            r2 : list
                Coordinates of point 2.

        Returns
        -------
            dist : float
                Euclidean distance between points 1 and 2.

    """
    dx = r1[0] - r2[0]
    dy = r1[1] - r2[1]
    dz = r1[2] - r2[2]
    d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return d


def vecdiff(r1, r2):
    """Element-wise vector difference

        Parameters
        ----------
            r1 : list
                Vector 1.
            r2 : list
                Vector 2.

        Returns
        -------
            diff : list
                Vector difference between points 1 and 2.

    """
    dr = [a - b for a, b in zip(r1, r2)]
    return dr


def midpt(r1, r2):
    """Vector midpoint.

        Parameters
        ----------
            r1 : list
                Vector 1.
            r2 : list
                Vector 2.

        Returns
        -------
            mid : list
                Midpoint between vector 1 and 2.

    """
    m = [0.5 * (a + b) for a, b in zip(r1, r2)]
    return m


def checkcolinear(r1, r2, r3):
    """Checks if three points are collinear.

        Parameters
        ----------
            r1 : list
                Coordinates of point 1.
            r2 : list
                Coordinates of point 2.
            r3 : list
                Coordinates of point 3.

        Returns
        -------
            collinear_flag : bool
                Flag for collinearity. True if collinear.

    """
    dr1 = vecdiff(r2, r1)
    dr2 = vecdiff(r1, r3)
    dd = np.cross(np.array(dr1), np.array(dr2))
    if norm(dd) < 1.e-01:
        return True
    else:
        return False


def checkplanar(r1, r2, r3, r4):
    """Checks if four points are coplanar.

        Parameters
        ----------
            r1 : list
                Coordinates of point 1.
            r2 : list
                Coordinates of point 2.
            r3 : list
                Coordinates of point 3.
            r4 : list
                Coordinates of point 4.

        Returns
        -------
            coplanar_flag : bool
                Flag for coplanarity. True if coplanarity.

    """
    r31 = vecdiff(r3, r1)
    r21 = vecdiff(r2, r1)
    r43 = vecdiff(r4, r3)
    cr0 = np.cross(np.array(r21), np.array(r43))
    dd = np.dot(r31, cr0)
    if abs(dd) < 1.e-1:
        return True
    else:
        return False


def vecangle(r1, r2):
    """Computes angle between two vectors.

        Parameters
        ----------
            r1 : list
                Vector 1.
            r2 : list
                Vector 2.

        Returns
        -------
            theta : float
                Angle between two vectors in degrees.

    """
    if (norm(r2) * norm(r1) > 1e-16):
        inner_prod = np.round(np.dot(r2, r1) / (norm(r2) * norm(r1)), 10)
        theta = 180 * np.arccos(inner_prod) / np.pi
    else:
        theta = 0.0
    return theta


def getPointu(Rr, dist, u):
    """Gets point given reference point, direction vector and distance.

        Parameters
        ----------
            Rr : list
                Reference point.
            dist : float
                Distance in angstroms.
            u : list
                Direction vector.

        Returns
        -------
            P : list
                Final point.

    """
    # get float bond length
    bl = float(dist)
    # get unit vector through line r = r0 + t*u
    t = bl / norm(u)  # get t as t=bl/norm(r1-r0)
    # get point
    P = [0, 0, 0]
    P[0] = t * u[0] + Rr[0]
    P[1] = t * u[1] + Rr[1]
    P[2] = t * u[2] + Rr[2]
    return P


def rotation_params(r0, r1, r2):
    """Gets angle between three points (r10 and r21) and the normal vector to the plane containing three points.

        Parameters
        ----------
            r0 : list
                Coordinates for point 1.
            r1 : list
                Coordinates for point 2.
            r2 : list
                Coordinates for point 3.

        Returns
        -------
            theta : float
                Angle in units of degrees.
            u : list
                Normal vector.

    """
    r10 = [a - b for a, b in zip(r1, r0)]
    r21 = [a - b for a, b in zip(r2, r1)]
    # print('r10 is ' +str(r10) )
    # print('r21 is ' +str(r21) )
    # angle between r10 and r21
    # print('arg to arcos  is ' +str(np.dot(r21,r10)/(norm(r21)*norm(r10))) )
    arg = np.dot(r21, r10) / (norm(r21) * norm(r10))
    if (norm(r21) * norm(r10) > 1e-16):
        if arg < 0:
            theta = 180 * np.arccos(max(-1, arg)) / np.pi
        else:
            theta = 180 * np.arccos(min(1, arg)) / np.pi
    else:
        theta = 0.0
    # get normal vector to plane r0 r1 r2
    u = np.cross(r21, r10)
    # check for collinear case
    if norm(u) < 1e-16:
        # pick random perpendicular vector
        if (abs(r21[0]) > 1e-16):
            u = [(-r21[1] - r21[2]) / r21[0], 1, 1]
        elif (abs(r21[1]) > 1e-16):
            u = [1, (-r21[0] - r21[2]) / r21[1], 1]
        elif (abs(r21[2]) > 1e-16):
            u = [1, 1, (-r21[0] - r21[1]) / r21[2]]
    return theta, u


def dihedral(mol, idx1, idx2, idx3, idx4):
    """Computes dihedral angle for a set of four atom indices.

        Parameters
        ----------
            mol0 : mol3D
                mol3D class instance of molecule for which we compute dihedral angle.
            idx1 : int
                Index of atom 1.
            idx2 : int
                Index of atom 2.
            idx3 : int
                Index of atom 3.
            idx4 : int
                Index of atom 4.
    """

    r1 = mol.getAtom(idx1).coords()
    r2 = mol.getAtom(idx2).coords()
    r3 = mol.getAtom(idx3).coords()
    r4 = mol.getAtom(idx4).coords()

    v1 = np.array(r2)-np.array(r1)  # vector formed between atoms 1 and 2
    v2 = np.array(r3)-np.array(r2)  # vector formed between atoms 2 and 3
    v3 = np.array(r4)-np.array(r3)  # vector formed between atoms 3 and 4

    v1_x_v2 = np.cross(v1, v2)  # cross product of v1 and v2
    v2_x_v3 = np.cross(v2, v3)  # cross product of v2 and v3

    normal_1 = v1_x_v2/(np.linalg.norm(v1_x_v2))  # normal to the plane formed by 1,2,3
    normal_2 = v2_x_v3/(np.linalg.norm(v2_x_v3))  # normal to the plane formed by 2,3,4

    unit_1 = v2/(np.linalg.norm(v2))
    unit_2 = np.cross(unit_1, normal_2)

    cos_angle = np.dot(normal_1, normal_2)
    sine_angle = np.dot(normal_1, unit_2)

    dihedral_angle = round(np.degrees(-np.arctan2(sine_angle, cos_angle)), 3)
    return dihedral_angle


def kabsch(mol0, mol1):
    """Aligns (translates and rotates) two molecules to minimize RMSD using the Kabsch algorithm.

        Parameters
        ----------
            mol0 : mol3D
                mol3D class instance of molecule to be aligned. Will be translated and rotated.
            mol1 : mol3D
                mol3D class instance of reference molecule. Will be translated.

        Returns
        -------
            mol0 : mol3D
                mol3D class instance of aligned molecule.
            U : list
                Rotation matrix as list of lists.
            d0 : list
                Translation vector for mol0.
            d1 : list
                Translation vector for mol1.

    """
    if (mol0.getNumAtoms() != mol1.getNumAtoms()):
        print(f'issue: {mol0.getNumAtoms()} != {mol1.getNumAtoms()}')
        raise ValueError('The two molecules should have the same number of atoms.')

    # translate to align centroids with origin
    mol0, d0 = setPdistance(mol0, mol0.centersym(), [0, 0, 0], 0)
    mol1, d1 = setPdistance(mol1, mol1.centersym(), [0, 0, 0], 0)
    # get coordinates and matrices P,Q
    P, Q = [], []
    for atom0, atom1 in zip(mol0.getAtoms(), mol1.getAtoms()):
        P.append(atom0.coords())
        Q.append(atom1.coords())
    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)
    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    # Create Rotation matrix U
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    # Rotate P
    P = np.dot(P, U)
    # write back coordinates
    for i, atom in enumerate(mol0.getAtoms()):
        atom.setcoords(P[i])
    return mol0, U.tolist(), d0, d1


def ReflectPlane(u, r, Rp):
    """Reflects point about plane defined by its normal vector and a point on the plane

        Parameters
        ----------
            u : list
                Normal vector to plane.
            r : list
                Point to be reflected.
            Rp : list
                Reference point on plane.

        Returns
        -------
            rn : list
                Reflected point.

    """
    un = norm(u)
    if (un > 1e-16):
        u[0] = u[0] / un
        u[1] = u[1] / un
        u[2] = u[2] / un
    # construct augmented vector rr = [r;1]
    d = -u[0] * Rp[0] - u[1] * Rp[1] - u[2] * Rp[2]
    # reflection matrix
    R = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    rn = [0, 0, 0]
    R[0][0] = 1 - 2 * u[0] * u[0]
    R[0][1] = -2 * u[0] * u[1]
    R[0][2] = -2 * u[0] * u[2]
    R[0][3] = -2 * u[0] * d
    R[1][0] = -2 * u[1] * u[0]
    R[1][1] = 1 - 2 * u[1] * u[1]
    R[1][2] = -2 * u[1] * u[2]
    R[1][3] = -2 * u[1] * d
    R[2][0] = -2 * u[2] * u[0]
    R[2][1] = -2 * u[1] * u[2]
    R[2][2] = 1 - 2 * u[2] * u[2]
    R[2][3] = -2 * u[2] * d
    R[3][3] = 1
    # get new point
    rn[0] = R[0][0] * r[0] + R[0][1] * r[1] + R[0][2] * r[2] + R[0][3]
    rn[1] = R[1][0] * r[0] + R[1][1] * r[1] + R[1][2] * r[2] + R[1][3]
    rn[2] = R[2][0] * r[0] + R[2][1] * r[1] + R[2][2] * r[2] + R[2][3]
    return rn


def PointRotateAxis(u, rp, r, theta):
    """Rotates point about axis defined by direction vector and point on axis. Theta units in radians.

    Parameters
    ----------
        u : list
            Direction vector of axis.
        rp : list
            Reference point along axis
        r : list
            Point to be rotated
        theta : float
            Angle of rotation in RADIANS.

    Returns
    -------
        rotated : list
            Rotated point.

    """
    # construct augmented vector rr = [r;1]
    rr = r
    rr.append(1)
    # rotation matrix about arbitrary line through rp
    R = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    rn = [0, 0, 0]
    R[0][0] = np.cos(theta) + u[0] ** 2 * (1 - np.cos(theta))
    R[0][1] = u[0] * u[1] * (1 - np.cos(theta)) - u[2] * np.sin(theta)
    R[0][2] = u[0] * u[2] * (1 - np.cos(theta)) + u[1] * np.sin(theta)
    R[0][3] = (rp[0] * (u[1] ** 2 + u[2] ** 2) - u[0] *
               (rp[1] * u[1] + rp[2] * u[2])) * (1 - np.cos(theta))
    R[0][3] += (rp[1] * u[2] - rp[2] * u[1]) * np.sin(theta)
    R[1][0] = u[1] * u[0] * (1 - np.cos(theta)) + u[2] * np.sin(theta)
    R[1][1] = np.cos(theta) + u[1] ** 2 * (1 - np.cos(theta))
    R[1][2] = u[1] * u[2] * (1 - np.cos(theta)) - u[0] * np.sin(theta)
    R[1][3] = (rp[1] * (u[0] ** 2 + u[2] ** 2) - u[1] *
               (rp[0] * u[0] + rp[2] * u[2])) * (1 - np.cos(theta))
    R[1][3] += (rp[2] * u[0] - rp[0] * u[2]) * np.sin(theta)
    R[2][0] = u[2] * u[0] * (1 - np.cos(theta)) - u[1] * np.sin(theta)
    R[2][1] = u[2] * u[1] * (1 - np.cos(theta)) + u[0] * np.sin(theta)
    R[2][2] = np.cos(theta) + u[2] ** 2 * (1 - np.cos(theta))
    R[2][3] = (rp[2] * (u[0] ** 2 + u[1] ** 2) - u[2] *
               (rp[0] * u[0] + rp[1] * u[1])) * (1 - np.cos(theta))
    R[2][3] += (rp[0] * u[1] - rp[1] * u[0]) * np.sin(theta)
    R[3][3] = 1
    # get new point
    rn[0] = R[0][0] * r[0] + R[0][1] * r[1] + R[0][2] * r[2] + R[0][3]
    rn[1] = R[1][0] * r[0] + R[1][1] * r[1] + R[1][2] * r[2] + R[1][3]
    rn[2] = R[2][0] * r[0] + R[2][1] * r[1] + R[2][2] * r[2] + R[2][3]
    return rn


def PointRotateMat(r, R):
    """Rotates point using arbitrary 3x3 rotation matrix

    Parameters
    ----------
        r : list
            Point to be rotated
        R : list
            List of lists for 3 by 3 rotation matrix.

    Returns
    -------
        rotated : list
            Rotated point.

    """
    rn = [0, 0, 0]
    rn[0] = R[0][0] * r[0] + R[1][0] * r[1] + R[2][0] * r[2]
    rn[1] = R[0][1] * r[0] + R[1][1] * r[1] + R[2][1] * r[2]
    rn[2] = R[0][2] * r[0] + R[1][2] * r[1] + R[2][2] * r[2]
    return rn


def PointTranslateSph(Rp, p0, D):
    """Translates point in spherical coordinates.

    Parameters
    ----------
        Rp : list
            Origin of sphere
        p0 : list
            Point to be translated
        D : list
            [final radial distance, change in polar phi, change in azimuthal theta]. Angles in RADIANS.

    Returns
    -------
        p : list
            Translated point.

    """
    # translate to origin
    ps = [0, 0, 0]
    ps[0] = p0[0] - Rp[0]
    ps[1] = p0[1] - Rp[1]
    ps[2] = p0[2] - Rp[2]
    # get initial spherical coords
    r0 = norm(ps)
    if (r0 < 1e-16):
        phi0 = 0.5 * np.pi
        theta0 = 0
    else:
        phi0 = np.arccos(ps[2] / r0)  # z/r
        theta0 = np.arctan2(ps[1], ps[0])  # y/x
    # get new point
    p = [0, 0, 0]
    p[0] = (D[0]) * np.sin(phi0 + D[2]) * np.cos(theta0 + D[1]) + Rp[0]
    p[1] = (D[0]) * np.sin(phi0 + D[2]) * np.sin(theta0 + D[1]) + Rp[1]
    p[2] = (D[0]) * np.cos(phi0 + D[2]) + Rp[2]
    return p


def PointTranslateSphgivenphi(Rp, p0, D):
    """Translates point in spherical coordinates. Redundant with PointTranslateSph. Will be deprecated.

    Parameters
    ----------
        Rp : list
            Origin of sphere
        p0 : list
            Point to be translated
        D : list
            [final radial distance, change in polar phi, change in azimuthal theta]. Angles in RADIANS.

    Returns
    -------
        p : list
            Translated point.

    """
    # translate to origin
    ps = [0, 0, 0]
    ps[0] = p0[0] - Rp[0]
    ps[1] = p0[1] - Rp[1]
    ps[2] = p0[2] - Rp[2]
    # get initial spherical coords
    r0 = norm(ps)
    if (r0 < 1e-16):
        phi0 = 0.5 * np.pi
        theta0 = 0
    else:
        phi0 = np.arccos(ps[2] / r0)  # z/r
        theta0 = np.arctan2(ps[1], ps[0])  # y/x
    # get new point
    p = [0, 0, 0]
    p[0] = (D[0]) * np.sin(phi0 + D[1]) * np.cos(theta0) + Rp[0]
    p[1] = (D[0]) * np.sin(phi0 + D[1]) * np.sin(theta0) + Rp[1]
    p[2] = (D[0]) * np.cos(phi0 + D[1]) + Rp[2]
    return p


def PointTranslateSphgivenr(Rp, p0, D, pref, r):
    """Translates point in spherical coordinates given R.

    Parameters
    ----------
        Rp : list
            Origin of sphere
        p0 : list
            Point to be translated
        D : list
            [final radial distance, change in polar phi, change in azimuthal theta]. Angles in RADIANS.
        pref : list
            Coordinates of reference point.
        r : float
            Given radius.

    Returns
    -------
        p : list
            Translated point.

    """
    # translate to origin
    ps = [0, 0, 0]
    ps[0] = p0[0] - Rp[0]
    ps[1] = p0[1] - Rp[1]
    ps[2] = p0[2] - Rp[2]
    # get initial spherical coords
    r0 = norm(ps)
    if (r0 < 1e-16):
        phi0 = 0.5 * np.pi
        theta0 = 0
    else:
        phi0 = np.arccos(ps[2] / r0)  # z/r
        theta0 = np.arctan2(ps[1], ps[0])  # y/x
    # get new point
    p = [0, 0, 0]
    r0 = 0
    theta0 = 0
    while abs(1 - r0 / r) > 0.01 and theta0 < 2 * np.pi:
        p[0] = (D[0]) * np.sin(phi0 + D[1]) * np.cos(theta0) + Rp[0]
        p[1] = (D[0]) * np.sin(phi0 + D[1]) * np.sin(theta0) + Rp[1]
        p[2] = (D[0]) * np.cos(phi0 + D[1]) + Rp[2]
        r0 = distance(p, pref)
        theta0 += 0.01
    return p


def PointTranslatetoPSph(Rp, p0, D):
    """Converts spherical translation vector into Cartesian translation vector

    Parameters
    ----------
        Rp : list
            Origin of sphere
        p0 : list
            Point to be translated
        D : list
            [final radial distance, change in polar phi, change in azimuthal theta]. Angles in RADIANS.

    Returns
    -------
        p : list
            Translation vector

    """
    # translate to origin
    ps = [0, 0, 0]
    ps[0] = p0[0] - Rp[0]
    ps[1] = p0[1] - Rp[1]
    ps[2] = p0[2] - Rp[2]
    # get current spherical coords
    r0 = norm(ps)
    if (r0 < 1e-16):
        phi0 = 0.5 * np.pi
        theta0 = 0
    else:
        phi0 = np.arccos(ps[2] / r0)  # z/r
        theta0 = np.arctan2(ps[1], ps[0])  # y/x
    # get translation vector
    p = [0, 0, 0]
    p[0] = D[0] * np.sin(phi0 + D[2]) * np.cos(theta0 + D[1])
    p[1] = D[0] * np.sin(phi0 + D[2]) * np.sin(theta0 + D[1])
    p[2] = D[0] * np.cos(phi0 + D[2])
    return p


def PointRotateSph(Rp, p0, D):
    """Rotates point about Cartesian axes defined relative to given origin.

    Parameters
    ----------
        Rp : list
            Cartesian origin
        p0 : list
            Point to be rotated
        D : list
            [theta-x, theta-y, theta-z] in RADIANS

    Returns
    -------
        p : list
            Rotated point

    """
    # translate to origin (reference)
    ps = [0, 0, 0]
    ps[0] = p0[0] - Rp[0]
    ps[1] = p0[1] - Rp[1]
    ps[2] = p0[2] - Rp[2]
    # build 3D rotation matrices about x,y,z axes
    Mx = [[1, 0, 0], [0, np.cos(D[0]), -np.sin(D[0])], [0, np.sin(D[0]), np.cos(D[0])]]
    My = [[np.cos(D[1]), 0, np.sin(D[1])], [0, 1, 0], [-np.sin(D[1]), 0, np.cos(D[1])]]
    Mz = [[np.cos(D[2]), -np.sin(D[2]), 0], [np.sin(D[2]), np.cos(D[2]), 0], [0, 0, 1]]
    # get full rotation matrix
    M = np.array(np.mat(Mx) * np.mat(My) * np.mat(Mz))
    p = [0.0, 0.0, 0.0]
    # rotate atom and translate it back from origin
    p[0] = M[0][0] * ps[0] + M[0][1] * ps[1] + M[0][2] * ps[2] + Rp[0]
    p[1] = M[1][0] * ps[0] + M[1][1] * ps[1] + M[1][2] * ps[2] + Rp[1]
    p[2] = M[2][0] * ps[0] + M[2][1] * ps[1] + M[2][2] * ps[2] + Rp[2]
    return p


def reflect_through_plane(mol, u, Rp):
    """Reflects molecule about plane defined by its normal vector and a point on the plane.
    Loops over ReflectPlane().

    Parameters
    ----------
        mol : mol3D
            mol3D class instance of molecule to be reflected.
        u : list
            Normal vector to plane.
        Rp : list
            Reference point on plane.

    Returns
    -------
        mol : mol3D
            mol3D class instance of reflected molecule.

    """
    un = norm(u)
    if (un > 1e-16):
        u[0] = u[0] / un
        u[1] = u[1] / un
        u[2] = u[2] / un
    for atom in mol.atoms:
        # Get new point after rotation
        Rt = ReflectPlane(u, atom.coords(), Rp)
        atom.setcoords(Rt)
    return mol


def rotate_around_axis(mol, Rp, u, theta):
    """Rotates molecule about axis defined by direction vector and point on axis.
    Loops over PointRotateAxis().

    Parameters
    ----------
        mol : mol3D
            mol3D class instance of molecule to be rotated.
        Rp : list
            Reference point along axis.
        u : list
            Direction vector of axis.
        theta : float
            Angle of rotation in DEGREES.

    Returns
    -------
        mol : mol3D
            mol3D class instance of rotated molecule.

    """
    un = norm(u)
    theta = (theta / 180.0) * np.pi
    if (un > 1e-16):
        u[0] = u[0] / un
        u[1] = u[1] / un
        u[2] = u[2] / un
    for atom in mol.atoms:
        # Get new point after rotation
        Rt = PointRotateAxis(u, Rp, atom.coords(), theta)
        atom.setcoords(Rt)
    return mol


def rotate_mat(mol, R):
    """Rotates molecule using arbitrary rotation matrix.
    Loops over PointRotateMat().

    Parameters
    ----------
        mol : mol3D
            mol3D class instance of molecule to be rotated.
        R : list
            List of lists containing rotation matrix.

    Returns
    -------
        mol : mol3D
            mol3D class instance of rotated molecule.

    """
    for atom in mol.atoms:
        # Get new point after rotation
        Rt = PointRotateMat(atom.coords(), R)
        atom.setcoords(Rt)
    return mol


def setPdistance(mol, Rr, Rp, bond):
    """Translates molecule such that a given point in the molecule is at a given distance from a reference point.
    The molecule is moved along the axis given by the two points.

    Parameters
    ----------
        mol : mol3D
            mol3D class instance of molecule to be translated.
        Rr : list
            Point in molecule to be aligned.
        Rp : list
            Reference alignment point.
        bond : float
            Final distance of aligned point to alignment point

    Returns
    -------
        mol : mol3D
            mol3D class instance of translated molecule.
        dxyz : np.array
            The translation vector.

    """
    # get float bond length
    bl = float(bond)
    # get center of mass
    # get unit vector through line r = r0 + t*u
    dxyz = [0, 0, 0]
    try:
        u = [a - b for a, b in zip(Rr, Rp)]
        t = bl / norm(u)  # get t as t=bl/norm(r1-r0)
        # get shift for centermass
        dxyz[0] = Rp[0] + t * u[0] - Rr[0]
        dxyz[1] = Rp[1] + t * u[1] - Rr[1]
        dxyz[2] = Rp[2] + t * u[2] - Rr[2]
    except ZeroDivisionError:
        pass
        # translate molecule
    mol.translate(dxyz)
    return mol, dxyz


def setPdistanceu(mol, Rr, Rp, bond, u):
    """Translates molecule such that a given point in the molecule is at a given distance from a reference point.
    The molecule is moved along an arbitrary axis.

    Parameters
    ----------
        mol : mol3D
            mol3D class instance of molecule to be translated.
        Rr : list
            Point in molecule to be aligned.
        Rp : list
            Reference alignment point.
        bond : float
            Final distance of aligned point to alignment point
        u : list
            Direction vector of axis

    Returns
    -------
        mol : mol3D
            mol3D class instance of translated molecule.

    """
    # get float bond length
    bl = float(bond)
    # get unit vector through line r = r0 + t*u
    t = bl / norm(u)  # get t as t=bl/norm(r1-r0)
    # get shift for centermass
    dxyz = [0, 0, 0]
    dxyz[0] = Rp[0] + t * u[0] - Rr[0]
    dxyz[1] = Rp[1] + t * u[1] - Rr[1]
    dxyz[2] = Rp[2] + t * u[2] - Rr[2]
    # translate molecule
    mol.translate(dxyz)
    return mol


def setcmdistance(mol, Rp, bond):
    """Translates molecule such that its center of mass is at a given distance from a reference point.
    The molecule is moved along the axis given by the two points.

    Parameters
    ----------
        mol : mol3D
            mol3D class instance of molecule to be translated.
        Rp : list
            Reference alignment point.
        bond : float
            Final distance of aligned point to alignment point

    Returns
    -------
        mol : mol3D
            mol3D class instance of translated molecule.

    """
    # get float bond length
    bl = float(bond)
    # get center of mass
    cm = mol.centermass()
    # get unit vector through line r = r0 + t*u
    u = [a - b for a, b in zip(cm, Rp)]
    t = bl / norm(u)  # get t as t=bl/norm(r1-r0)
    # get shift for centermass
    dxyz = [0, 0, 0]
    dxyz[0] = Rp[0] + t * u[0] - cm[0]
    dxyz[1] = Rp[1] + t * u[1] - cm[1]
    dxyz[2] = Rp[2] + t * u[2] - cm[2]
    # translate molecule
    mol.translate(dxyz)
    return mol


def protate(mol, Rr, D):
    """Translates molecule in spherical coordinates based on center of mass reference.
    Loops over PointTranslateSph().

    Parameters
    ----------
        mol : mol3D
            mol3D class instance of molecule to be translated.
        Rr : list
            Origin of sphere.
        D : list
            [final radial distance, change in polar phi, change in azimuthal theta] in RADIANS

    Returns
    -------
        mol : mol3D
            mol3D class instance of translated molecule.

    """
    # convert to rad
    D[0] = float(D[0])
    D[1] = (float(D[1]) / 180.0) * np.pi
    D[2] = (float(D[2]) / 180.0) * np.pi
    # rotate/translate about reference point
    # get center of mass
    pmc = mol.centermass()
    # get translation vector that corresponds to new coords
    Rt = PointTranslateSph(Rr, pmc, D)
    # translate molecule
    mol.translate(Rt)
    return mol


def protateref(mol, Rr, Rref, D):
    """Translates molecule in spherical coordinates based on arbitrary reference.
    Loops over PointTranslateSph().

    Parameters
    ----------
        mol : mol3D
            mol3D class instance of molecule to be translated.
        Rr : list
            Origin of sphere.
        Rref : list
            Reference point in molecule
        D : list
            [final radial distance, change in polar phi, change in azimuthal theta] in RADIANS

    Returns
    -------
        mol : mol3D
            mol3D class instance of translated molecule.

    """
    # rotate/translate about reference point
    # convert to rad
    D[0] = float(D[0])
    D[1] = (float(D[1]) / 180.0) * np.pi
    D[2] = (float(D[2]) / 180.0) * np.pi
    # rotate/translate about reference point
    # get translation vector that corresponds to new coords
    Rt = PointTranslateSph(Rr, Rref, D)
    # translate molecule
    mol.translate(Rt)
    return mol


def cmrotate(mol, D):
    """Rotates molecule about its center of mass
    Loops over PointRotateSph().

    Parameters
    ----------
        mol : mol3D
            mol3D class instance of molecule to be rotated.
        D : list
            [theta-x, theta-y, theta-z] in RADIANS

    Returns
    -------
        mol : mol3D
            mol3D class instance of rotated molecule.

    """
    # convert to rad
    D[0] = (float(D[0]) / 180.0) * np.pi
    D[1] = (float(D[1]) / 180.0) * np.pi
    D[2] = (float(D[2]) / 180.0) * np.pi
    # perform rotation
    pmc = mol.centermass()
    for atom in mol.atoms:
        # Get new point after rotation
        Rt = PointRotateSph(pmc, atom.coords(), D)
        atom.setcoords(Rt)
    return mol


def rotateRef(mol, Ref, D):
    """Rotates molecule about an arbitrary point
    Loops over PointRotateSph().

    Parameters
    ----------
        mol : mol3D
            mol3D class instance of molecule to be rotated.
        Ref : list
            Reference point
        D : list
            [theta-x, theta-y, theta-z] in RADIANS

    Returns
    -------
        mol : mol3D
            mol3D class instance of rotated molecule.

    """
    # convert to rad
    D[0] = (float(D[0]) / 180.0) * np.pi
    D[1] = (float(D[1]) / 180.0) * np.pi
    D[2] = (float(D[2]) / 180.0) * np.pi
    # perform rotation
    for atom in mol.atoms:
        # Get new point after rotation
        Rt = PointRotateSph(Ref, atom.coords(), D)
        atom.setcoords(Rt)
    return mol


def aligntoaxis(mol, Rr, Rp, u):
    """Translates molecule to align point to axis at constant distance.

    Parameters
    ----------
        mol : mol3D
            mol3D class instance of molecule to be translated.
        Rr : list
            Point to be aligned
        Rp : list
            Reference point on axis
        u : list
            Target axis for alignment

    Returns
    -------
        mol : mol3D
            mol3D class instance of aligned molecule.

    """
    # get current distance
    d0 = distance(Rp, Rr)
    # normalize u
    t = d0 / norm(u)  # get t as t=bl/norm(r1-r0)
    # get shift for point
    dxyz = [0, 0, 0]
    dxyz[0] = Rp[0] + t * u[0] - Rr[0]
    dxyz[1] = Rp[1] + t * u[1] - Rr[1]
    dxyz[2] = Rp[2] + t * u[2] - Rr[2]
    # translate molecule
    mol.translate(dxyz)
    return mol


def aligntoaxis2(mol, Rr, Rp, u, d):
    """Translates molecule to align point to axis at arbitrary distance

    Parameters
    ----------
        mol : mol3D
            mol3D class instance of molecule to be translated.
        Rr : list
            Point to be aligned
        Rp : list
            Reference point on axis
        u : list
            Target axis for alignment
        d : float
            Final distance from aligned point to axis

    Returns
    -------
        mol : mol3D
            mol3D class instance of translated molecule.

    """
    # normalize u
    t = d / norm(u)  # get t as t=bl/norm(r1-r0)
    # get shift for point
    dxyz = [0, 0, 0]
    dxyz[0] = Rp[0] + t * u[0] - Rr[0]
    dxyz[1] = Rp[1] + t * u[1] - Rr[1]
    dxyz[2] = Rp[2] + t * u[2] - Rr[2]
    # translate molecule
    mol.translate(dxyz)
    return mol


def alignPtoaxis(Rr, Rp, u, d):
    """Translates point and aligns to axis

    Parameters
    ----------
        Rr : list
            Point to be aligned
        Rp : list
            Reference point on axis
        u : list
            Target axis for alignment. Direction vector.
        d : float
            Final distance from aligned point to axis

    Returns
    -------
        dxyz : list
            Translation vector

    """
    # normalize u
    t = d / norm(u)  # get t as t=bl/norm(r1-r0)
    # get shift for point
    dxyz = [0, 0, 0]
    dxyz[0] = Rp[0] + t * u[0]
    dxyz[1] = Rp[1] + t * u[1]
    dxyz[2] = Rp[2] + t * u[2]
    return dxyz


def pmrotate(mol, Rp, D):
    """Rotates molecule about Cartesian axes defined relative to given origin.
    Loops over PointRotateSph().

    Parameters
    ----------
        mol : mol3D
            mol3D class instance of molecule to be rotated.
        Rp : list
            Cartesian origin.
        D : list
            [theta-x, theta-y, theta-z] in DEGREES

    Returns
    -------
        mol : mol3D
            mol3D class instance of rotated molecule.

    """
    # convert to rad
    D[0] = (float(D[0]) / 180.0) * np.pi
    D[1] = (float(D[1]) / 180.0) * np.pi
    D[2] = (float(D[2]) / 180.0) * np.pi
    # perform rotation
    for atom in mol.atoms:
        # Get new point after rotation
        Rt = PointRotateSph(Rp, atom.coords(), D)
        atom.setcoords(Rt)
    return mol


def connectivity_match(inds1, inds2, mol1, mol2):
    """Check whether the connectivity of two fragments of mols match.
    Note: This will mark atom transfers between different ligands as False, which may not be correct mathmetically
    as the graph after atoms transfer can still be the same. We disallow these cases from chemical concerns and avoid
    the NP-hard porblem of comparing two adjecent matrix.

    Parameters
    ----------
        inds1 : list
            List of atom inds in molecule 1.
        inds2 : list
            List of atom inds in molecule 2.
        mol1 : mol3D
            mol3D class instance for molecule 1.
        mol2 : mol3D
            mol3D class instance for molecule 2.

    Returns
    -------
        match_flag : bool
            Flag for if connectivity matches. True if so.

    """
    match = False
    if len(inds1) == len(inds2):
        inds1.sort()
        inds2.sort()
        _mol1 = mol1.create_mol_with_inds(inds1)
        _mol2 = mol2.create_mol_with_inds(inds2)
        _mol1.createMolecularGraph()
        _mol2.createMolecularGraph()
        match = np.array_equal(_mol1.graph, _mol2.graph)
    return match


def best_fit_plane(coordinates):
    """Finds the best fitting plane to a set of atoms at the specified coordinates.

    Parameters
    ----------
        corerefcoords : np.array
            Coordinates of atoms for which the best fitting plane is to be found. Shape is 3 x N.

    Returns
    -------
        normal_vector_plane : np.array
            The vector perpendicular to the best fitting plane.

    """
    # Solution from stack exchange

    # subtract out the centroid and take the SVD
    svd = np.linalg.svd(coordinates - np.mean(coordinates, axis=1, keepdims=True))

    # Extract the left singular vectors
    left = svd[0]

    # the corresponding left singular vector is the normal vector of the best-fitting plane
    normal_vector_plane = left[:, -1]
    return normal_vector_plane
