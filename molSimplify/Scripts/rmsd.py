import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def rmsd(V, W):
    """Calculate Root-mean-square deviation from two sets of vectors V and W.

    Parameters
    ----------
        V : np.array
            (N,D) matrix, where N is points and D is dimension.
        W : np.array
            (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
        rmsd : float
            Root-mean-square deviation between the two vectors.
    """
    D = len(V[0])
    N = len(V)
    result = 0.0
    for v, w in zip(V, W):
        result += sum([(v[i] - w[i]) ** 2.0 for i in range(D)])
    return np.sqrt(result / N)


def kabsch_rmsd(P, Q, translate=False):
    """Rotate matrix P unto Q using Kabsch algorithm and calculate the RMSD.

    Parameters
    ----------
        P : np.array
            (N,D) matrix, where N is points and D is dimension.
        Q : np.array
            (N,D) matrix, where N is points and D is dimension.
        translate : bool, optional
            Use centroids to translate vector P and Q unto each other. Default is False.

    Returns
    -------
        rmsd : float
            root-mean squared deviation
    """
    if translate:
        Q = Q - centroid(Q)
        P = P - centroid(P)

    P = kabsch_rotate(P, Q)
    return rmsd(P, Q)


def kabsch_rotate(P, Q):
    """Rotate matrix P unto matrix Q using Kabsch algorithm.

    Parameters
    ----------
        P : np.array
            (N,D) matrix, where N is points and D is dimension.
        Q : np.array
            (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
        P : np.array
            (N,D) matrix, where N is points and D is dimension, rotated.
    """
    U = kabsch(P, Q)

    # Rotate P
    P = np.dot(P, U)
    return P


def kabsch(P, Q):
    """Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD matrix, where D is the
    dimension of the space. The algorithm works in three steps:
    1. a centroid translation of P and Q (assumed done before this functioncall)
    2. the computation of a covariance matrix C
    3. computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm

    Parameters
    ----------
        P : np.array
            (N,D) matrix, where N is points and D is dimension.
        Q : np.array
            (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
        U : np.array
            Rotation matrix (D,D)
    """
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

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


def quaternion_rmsd(P, Q) -> float:
    """ Rotate matrix P unto Q and calculate the RMSD
    based on doi:10.1016/1049-9660(91)90036-O

    Parameters
    ----------
        P : np.array
            (N,D) matrix, where N is points and D is dimension.
        Q : np.array
            (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
        rmsd : float
            RMSD between P and Q.
    """
    rot = quaternion_rotate(P, Q)
    P = np.dot(P, rot)
    return rmsd(P, Q)


def quaternion_transform(r):
    """Get optimal rotation.
    Note: translation will be zero when the centroids of each molecule are the same.

    Parameters
    ----------
        r : np.array
            Array of vectors to transform.

    """
    Wt_r = makeW(*r).T
    Q_r = makeQ(*r)
    rot = Wt_r.dot(Q_r)[:3, :3]
    return rot


def makeW(r1, r2, r3, r4=0):
    """Make W matrix involved in quaternion rotation

    Parameters
    ----------
        r1 : np.array
            Vector 1.
        r2 : np.array
            Vector 2.
        r3 : np.array
            Vector 3.
        r4 : np.array, optional
            Vector 4. Default is 0.

    Return
    ------
        W : np.array
            W matrix involved in quaternion rotation.

    """
    W = np.asarray([
        [r4, r3, -r2, r1],
        [-r3, r4, r1, r2],
        [r2, -r1, r4, r3],
        [-r1, -r2, -r3, r4]])
    return W


def makeQ(r1, r2, r3, r4=0):
    """Make Q matrix involved in quaternion rotation

    Parameters
    ----------
        r1 : np.array
            Vector 1.
        r2 : np.array
            Vector 2.
        r3 : np.array
            Vector 3.
        r4 : np.array, optional
            Vector 4. Default is 0.

    Return
    ------
        Q : np.array
            Q matrix involved in quaternion rotation.

    """
    Q = np.asarray([
        [r4, -r3, r2, r1],
        [r3, r4, -r1, r2],
        [-r2, r1, r4, r3],
        [-r1, -r2, -r3, r4]])
    return Q


def quaternion_rotate(X, Y):
    """Calculate the rotation

    Parameters
    ----------
        X : array
            (N,D) matrix, where N is points and D is dimension.
        Y: array
            (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
        rot : matrix
            Rotation matrix (D,D)
    """
    N = X.shape[0]
    W = np.asarray([makeW(*Y[k]) for k in range(N)])
    Q = np.asarray([makeQ(*X[k]) for k in range(N)])
    Qt_dot_W = np.asarray([np.dot(Q[k].T, W[k]) for k in range(N)])
    # W_minus_Q = np.asarray([W[k] - Q[k] for k in range(N)])
    A = np.sum(Qt_dot_W, axis=0)
    eigen = np.linalg.eigh(A)
    r = eigen[1][:, eigen[0].argmax()]
    rot = quaternion_transform(r)
    return rot


def centroid(X):
    """ Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X. https://en.wikipedia.org/wiki/Centroid
    C = sum(X)/len(X)

    Parameters
    ----------
        X : np.array
            (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
        C : float
            centroid
    """
    C = X.mean(axis=0)
    return C


def hungarian(A, B):
    """Hungarian reordering.
    Assume A and B are coordinates for atoms of SAME type only.

    Parameters
    ----------
        A : np.array
            (N,D) matrix, where N is points and D is dimension. coordinates.
        B : np.array
            (N,D) matrix, where N is points and D is dimension. coordinates.

    Returns
    -------
        indices_b : np.array
            Indices as a result of Hungarian analysis on distance matrix between atoms of 1st structure and trial structure

    """

    # should be kabasch here i think
    distances = cdist(A, B, 'euclidean')

    # Perform Hungarian analysis on distance matrix between atoms of 1st
    # structure and trial structure
    indices_a, indices_b = linear_sum_assignment(distances)

    return indices_b


def reorder_hungarian(p_atoms, q_atoms, p_coord, q_coord):
    """Re-orders the input atom list and xyz coordinates using the Hungarian
    method (using optimized column results)

    Parameters
    ----------
        p_atoms : np.array
            (N,1) matrix, where N is points holding the atoms' names
        p_atoms : np.array
            (N,1) matrix, where N is points holding the atoms' names
        p_coord : np.array
            (N,D) matrix, where N is points and D is dimension
        q_coord : np.array
            (N,D) matrix, where N is points and D is dimension

    Returns
    -------
        view_reorder : np.array
                 (N,1) matrix, reordered indexes of atom alignment based on the
                 coordinates of the atoms
    """

    # Find unique atoms
    unique_atoms = np.unique(p_atoms)

    # generate full view from q shape to fill in atom view on the fly
    view_reorder = np.zeros(np.array(q_atoms).shape, dtype=int)
    view_reorder -= 1

    for atom in unique_atoms:
        p_atom_idx = np.where(p_atoms == atom)[0]
        q_atom_idx = np.where(q_atoms == atom)[0]
        A_coord = np.array(p_coord)[p_atom_idx]
        B_coord = np.array(q_coord)[q_atom_idx]

        view = hungarian(A_coord, B_coord)
        view_reorder[p_atom_idx] = q_atom_idx[view]

    return view_reorder


def reorder_distance(p_atoms, q_atoms, p_coord, q_coord):
    """ Re-orders the input atom list and xyz coordinates by atom type and then by
    distance of each atom from the centroid.

    Parameters
    ----------
        atoms : np.array
            (N,1) matrix, where N is points holding the atoms' names
        coord : np.array
            (N,D) matrix, where N is points and D is dimension

    Returns
    -------
        atoms_reordered : np.array
            (N,1) matrix, where N is points holding the ordered atoms' names
        coords_reordered : np.array
            (N,D) matrix, where N is points and D is dimension (rows re-ordered)
    """

    # Find unique atoms
    unique_atoms = np.unique(p_atoms)

    # generate full view from q shape to fill in atom view on the fly
    view_reorder = np.zeros(np.array(q_atoms).shape, dtype=int)

    for atom in unique_atoms:
        p_atom_idx = np.where(p_atoms == atom)[0]
        q_atom_idx = np.where(q_atoms == atom)[0]
        A_coord = np.array(p_coord)[p_atom_idx]
        B_coord = np.array(q_coord)[q_atom_idx]

        # Calculate distance from each atom to centroid
        A_norms = np.linalg.norm(A_coord, axis=1)
        B_norms = np.linalg.norm(B_coord, axis=1)

        reorder_indices_A = np.argsort(A_norms)
        reorder_indices_B = np.argsort(B_norms)

        # Project the order of P onto Q
        translator = np.argsort(reorder_indices_A)
        view = reorder_indices_B[translator]
        view_reorder[p_atom_idx] = q_atom_idx[view]

    return view_reorder


def rmsd_reorder_rotate(p_atoms, q_atoms, p_coord, q_coord,
                        rotation="kabsch", reorder="hungarian"):
    """Reorder and rotate for RMSD.

    Parameters
    ----------
        p_atoms : np.array
            Atom symbol list.
        q_atoms : np.array
            Atom symbol list.
        p_coord : np.array
            List of coordinates for p_atoms.
        q_atoms : np.array
            List of coordinates for q_atoms.
        rotation : str, optional
            Rotation method. Default is kabsch.
        reorder : str, optional
            Reorder method. Default is hungarian.

    Returns
    -------
        result_rmsd : float
            Resulting RMSD from aligning and rotating.

    """
    if not p_atoms.shape[0] == q_atoms.shape[0]:
        print(("Warning: Number of atoms do not match!",
               p_atoms.shape[0], q_atoms[0]))
        return 1000
    elif not len(set(np.unique(p_atoms)) - set(np.unique(q_atoms))) == 0:
        print(("Warning: Atom types do not match!",
               np.unique(p_atoms), np.unique(q_atoms)))
        return 1000

    p_cent = centroid(p_coord)
    q_cent = centroid(q_coord)
    p_coord -= p_cent
    q_coord -= q_cent

    # set rotation method
    if rotation.lower() == "kabsch":
        rotation_method = kabsch_rmsd
    elif rotation.lower() == "quaternion":
        rotation_method = quaternion_rmsd
    elif rotation.lower() == "none":
        rotation_method = None
    else:
        raise ValueError("error: Unknown rotation method:", rotation)

    # set reorder method
    if reorder.lower() == "hungarian":
        reorder_method = reorder_hungarian
    elif reorder.lower() == "distance":
        reorder_method = reorder_distance
    elif reorder.lower() == "none":
        reorder_method = None
    else:
        raise ValueError("error: Unknown reorder method:", reorder)

    if not reorder.lower() == "none":
        q_review = reorder_method(p_atoms, q_atoms, p_coord, q_coord)
        q_coord = q_coord[q_review]
        # q_atoms = q_atoms[q_review]
        # print("q_review", q_review)

    if rotation_method is None:
        result_rmsd = rmsd(p_coord, q_coord)
    else:
        result_rmsd = rotation_method(p_coord, q_coord)
    return result_rmsd


def rigorous_rmsd(mol1, mol2, rotation: str = "kabsch",
                  reorder: str = "hungarian") -> float:
    """Rigorous RMSD measurement

    Parameters
    ----------
        mol1 : mol3D
            mol3D instance of initial molecule.
        mol2 : np.mol3D
            mol3D instance of final molecule.
        rotation : str, optional
            Rotation method. Default is kabsch.
        reorder : str, optional
            Reorder method. Default is hungarian.

    Returns
    -------
        result_rmsd : float
            Resulting RMSD from aligning and rotating.

    """
    mol1_atoms = mol1.symvect()
    mol1_coords = mol1.coordsvect()
    mol2_atoms = mol2.symvect()
    mol2_coords = mol2.coordsvect()
    result_rmsd = rmsd_reorder_rotate(mol1_atoms, mol2_atoms, mol1_coords, mol2_coords,
                                      rotation=rotation, reorder=reorder)
    return result_rmsd


def test_case():
    p_atoms = np.array(["N", "H", "H", "H"])
    q_atoms = np.array(["H", "N", "H", "H"])
    p_coord = np.array([[0.000000, 2.030000, 0.000000],
                        [-0.975035, 2.404393, -0.001212],
                        [0.486430, 2.404203, 0.845016],
                        [0.488605, 2.404166, -0.843804]
                        ])
    q_coord = np.array([[0.486430, 2.404203, 0.845016],
                        [0.000000, 2.030000, 0.000000],
                        [-0.975035, 2.404393, -0.001212],
                        [0.488605, 2.404166, -0.843804]
                        ])
    return p_atoms, q_atoms, p_coord, q_coord
