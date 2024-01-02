from molSimplify.Scripts.cellbuilder_tools import import_from_cif
from molSimplify.Informatics.MOF.PBC_functions import (
    compute_adj_matrix,
    compute_distance_matrix3,
    frac_coord,
    fractional2cart,
    readcif,
    XYZ_connected,
    write_cif,
    )
from molSimplify.Informatics.MOF.MOF_functionalizer import get_linkers
import numpy as np
import os

def rotate_around_axis(axis, r, p, t):
    """
    Function that rotates the point about the axis with given angle
    # 1) Translate space so that the reference point locate at the origin (T)
    # 2) Rotate space so that the rotation axis lies in the xz plane (R_x)
    # 3) Rotate space so that the rotation axis lies in the z axis (R_y)
    # 4) Rotate angle t about the z axis (R_z)
    # 5) Rotate and translate space back to original space (T_inv, R_xinv, R_yinv)

    Parameters
    ----------
    axis : The rotation axis vector
    r : The reference point of the axis vector
    p : The point to rotate
    t : Rotation angle

    Returns
    -------
    new : new coordinates
    """
    unit_axis = axis / np.linalg.norm(axis) # normalize axis vector
    a = unit_axis[0]
    b = unit_axis[1]
    c = unit_axis[2]
    d = np.sqrt(unit_axis[1]**2 + unit_axis[2]**2)

    T = [[1, 0, 0, -r[0]], [0, 1, 0, -r[1]], [0, 0, 1, -r[2]], [0, 0, 0, 1]]
    T_inv = np.linalg.inv(T)
    old = [p[0], p[1], p[2], 1]

    if d != 0:
        R_x = [[1, 0, 0, 0], [0, c/d, -b/d, 0], [0, b/d, c/d, 0], [0, 0, 0, 1]]
        R_xinv = np.linalg.inv(R_x)
        R_y = [[d, 0, -a, 0], [0, 1, 0, 0], [a, 0, d, 0], [0, 0, 0, 1]]
        R_yinv = np.linalg.inv(R_y)
        R_z = [[np.cos(t), np.sin(t), 0, 0], [-np.sin(t), np.cos(t), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        # new = T_inv * R_xinv * R_yinv * R_z * R_y * R_x * T * old
        new = T_inv.dot(R_xinv).dot(R_yinv).dot(R_z).dot(R_y).dot(R_x).dot(T).dot(old)
    else: # if d ==0, rotation axis is along the x axis -> no rotation along y/z axis
        R_x = [[1, 0, 0, 0], [0, np.cos(t), np.sin(t), 0], [0, -np.sin(t), np.cos(t), 0], [0, 0, 0, 1]]
        # new = T_inv * R_x * T * old
        new = T_inv.dot(R_x).dot(T).dot(old)
    return (new[0:3])


def linker_rotation(molcif, fcoords, linker, rot_angle):
    """
    Finds the rotation axis on the given linker and rotate the linker about the rotation axis.
    Linker must be carboxylic acid linker.
    Currently works for MOFs with Zr atom as metal atom

    Parameters
    ----------
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    fcoords : numpy.ndarray
        The fractional coordinates of the atoms.
    linker : list of numpy.int32
        The indices of the atoms in the linker.
    rot_angle : float
        Desired angle of rotation.

    Returns
    -------
    frac_new_linker : numpy.ndarray
        fractional coordinates of new linker atoms
    """

    tmp_cart_coords = fractional2cart(fcoords, cell_v)
    fcoords_connected = XYZ_connected(cell_v, tmp_cart_coords, np.array(molcif.graph))
    cart_coords = fractional2cart(fcoords_connected, cell_v)
    Zr_bonded_O = []
    ZrO_bonded_C = []
    C_axis = []
    atom_not_to_rotate = []
    new_linker = []
    linker_coord_original = cart_coords[linker]

    # identifying Zr coordinated O's
    # molcif.getBondedAtomsSmart(idx) returns bonded atom id, allatomtypes[val] has bonded atom type
    for idx in linker:
        adj_atoms = [allatomtypes[val] for val in molcif.getBondedAtomsSmart(idx)]
        if 'Zr' in adj_atoms:
            atom_not_to_rotate.append(idx)
            Zr_bonded_O.append(idx)

    # Identifying stationary Carbon coordinated to Zr-O
    for idx in linker:
        for val in molcif.getBondedAtomsSmart(idx):
            if val in Zr_bonded_O and idx not in atom_not_to_rotate:
                atom_not_to_rotate.append(idx)
                ZrO_bonded_C.append(idx)
                break

    # Identifying stationary Carbon coordinated to Zr-O-C
    for idx in linker:
        for val in molcif.getBondedAtomsSmart(idx):
            if val in ZrO_bonded_C and idx not in atom_not_to_rotate:
                atom_not_to_rotate.append(idx)
                C_axis.append(cart_coords[idx]) # designate these Carbons as rotation axis
                break

    # Rotation axis defined by vectors between two stationary Carbons
    rot_axis = np.array(C_axis[1] - C_axis[0])

    # Obtain new linker coordinates
    for idx in linker:
        if idx in atom_not_to_rotate:
            new_linker.append(cart_coords[idx])
        else:
            new_linker.append(rotate_around_axis(rot_axis, C_axis[0], np.ndarray.tolist(cart_coords[idx]), rot_angle))

    # Change back to fractional coordinates
    frac_new_linker = frac_coord(new_linker, cell_v)

    return frac_new_linker

### End of functions ###

# Functional groups to use
func_group = ['Br','CF3','CH3','CN','COOH','Cl','F','I','NH2','NO2','OH','SH']

for elem in func_group:
    cif_file=f'functionalized_UiO66_{elem}_1.cif' # Functionalized .CIF file name goes here
    path2write = str(elem)+'/'
    # (from pbc_functions) reads cif and returns cpar (cell parametrs: 3 cell lengths, 3 cell angles), a list all atom elements in atom index order, and fractional coordinates
    cpar, allatomtypes, fcoords = readcif(path2write+cif_file)
    # obtains mol3D
    molcif,cell_vector, alpha, beta, gamma = import_from_cif(path2write+cif_file, True)
    cell_v = np.array(cell_vector)
    cart_coords = fractional2cart(fcoords,cell_v)
    distance_mat = compute_distance_matrix3(cell_v,cart_coords) # distance matrix of all atoms
    adj_matrix, _ = compute_adj_matrix(distance_mat,allatomtypes) # from distance matrix and heuristics for bond distances, obtains connectivity information in the form of adjacency matrix (graph)
    molcif.graph = adj_matrix.todense() # dense form of adjacency matrix / graph is saved to molcif object


    # list of linkers
    linker_list, linker_subgraphlist = get_linkers(molcif, adj_matrix, allatomtypes)
    # get BDC linkers
    linker_bdc_list = []
    for linker_num, linker in enumerate(linker_list):
        if len(linker) < 2:
            continue
        else:
            linker_bdc_list.append(linker)
    print(cpar)

    # get coordinates of BDC linkers
    linker_coords = [fcoords[val,:] for val in linker_bdc_list]

    coords_new = fcoords.copy()
    rot_angle_degree = np.linspace(0, 360, 25) # Define rotation angles
    rot_angle_list = rot_angle_degree/180*np.pi

    # Rotation of all of the linkers
    for i, rot_angle in enumerate(rot_angle_list):
        for linker_num, linker in enumerate(linker_bdc_list):
            new_linker=linker_rotation(molcif, fcoords, linker, rot_angle)
            coords_new[linker_bdc_list[linker_num],:] = new_linker # new_linker
        path_directory = str(path2write)+str(int(rot_angle_degree[i]))
        if not os.path.exists(path_directory):
               os.mkdir(path_directory)
        write_cif(f'{path_directory}/modified_{elem}_{int(rot_angle_degree[i])}.cif', cpar, coords_new, allatomtypes)
    print(str(elem) + " done")
