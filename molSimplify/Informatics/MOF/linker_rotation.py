from molSimplify.Scripts.cellbuilder_tools import import_from_cif
from molSimplify.Informatics.MOF.PBC_functions import (
    compute_adj_matrix,
    compute_distance_matrix,
    frac_coord,
    fractional2cart,
    mkcell,
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

def linker_rotation(molcif, fcoords, linker, rot_angle, cell_v, all_atom_types):
    """
    Finds the rotation axis on the given linker and rotate the linker about the rotation axis.
    Linker must be carboxylic acid linker.
    Currently works for MOFs with Zr as the metal element.

    Parameters
    ----------
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    fcoords : numpy.ndarray
        The fractional coordinates of the atoms.
    linker : list of numpy.int32
        The indices of the atoms in the linker.
    rot_angle : float
        Desired angle of rotation, in radians.
    cell_v : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    all_atom_types : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.

    Returns
    -------
    frac_new_linker : numpy.ndarray
        fractional coordinates of new linker atoms

    """
    original_cart_coords = fractional2cart(fcoords, cell_v)
    full_cart_coords = original_cart_coords.copy()
    linker_subgraph = np.array(molcif.graph)[np.ix_(np.array(linker), np.array(linker))]
    linker_fcoords_connected = XYZ_connected(cell_v, full_cart_coords[linker], linker_subgraph)
    #global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(linker)}
    linker_cart_coords = fractional2cart(linker_fcoords_connected, cell_v)
    full_cart_coords[linker] = linker_cart_coords

    metal_bonded_O = []
    metalO_bonded_C = []
    C_axis = []
    atom_not_to_rotate = []
    new_linker = []
    #linker_coord_original = cart_coords[linker]

    metal_list = molcif.findMetal(transition_metals_only=False)

    # identifying metal coordinated O's
    # molcif.getBondedAtomsSmart(idx) returns bonded atom id, all_atom_types[val] has bonded atom type
    for idx in linker:
        adj_atoms = [all_atom_types[val] for val in molcif.getBondedAtomsSmart(idx)]
        if any(all_atom_types[metal] in adj_atoms for metal in metal_list):
            atom_not_to_rotate.append(idx)
            metal_bonded_O.append(idx)

    # Identifying stationary Carbon coordinated to metal-O
    for idx in linker:
        for val in molcif.getBondedAtomsSmart(idx):
            if val in metal_bonded_O and idx not in atom_not_to_rotate:
                atom_not_to_rotate.append(idx)
                metalO_bonded_C.append(idx)
                break

    # Identifying stationary Carbon coordinated to metal-O-C
    for idx in linker:
        for val in molcif.getBondedAtomsSmart(idx):
            if val in metalO_bonded_C and idx not in atom_not_to_rotate:
                atom_not_to_rotate.append(idx)
                C_axis.append(full_cart_coords[idx]) # designate these Carbons as rotation axis
                break

    # Rotation axis defined by vectors between two stationary Carbons
    # If there are no stationary Carbons, return original coordinates
    try:
        rot_axis = np.array(C_axis[1] - C_axis[0])
    except IndexError:
        return fcoords[linker]

    # Obtain new linker coordinates
    for idx in linker:
        if idx in atom_not_to_rotate:
            new_linker.append(full_cart_coords[idx])
        else:
            new_linker.append(rotate_around_axis(rot_axis, C_axis[0], np.ndarray.tolist(full_cart_coords[idx]), rot_angle))

    # Change back to fractional coordinates
    frac_new_linker = frac_coord(new_linker, cell_v)

    return frac_new_linker

def rotate_and_write(input_cif, path2write, rot_angle, is_degree=True):
    """
    Rotates the linkers in the provided cif by the angle rot_angle.
    Writes the resulting cif to path2write.
    Currently works for MOFs with Zr as the metal element,
    and carboxylate linkers.

    Parameters
    ----------
    input_cif : str
        The path to the cif file to have linkers rotated.
    path2write : str
        The folder path where the cif of the linker-rotated MOF will be written.
    rot_angle : float
        Desired angle of rotation, in radians.
    is_degree : bool, optional
        If True, rot_angle is in degrees. If False, rot_angle is in radians.
        The default is True.
    Returns
    -------
    None

    """
    basename = os.path.basename(input_cif).replace('.cif', '')
    cpar, all_atom_types, fcoords = readcif(input_cif)

    molcif, _, _, _, _ = import_from_cif(input_cif, True)
    cell_v = mkcell(cpar)
    cart_coords = fractional2cart(fcoords, cell_v)
    distance_mat = compute_distance_matrix(cell_v, cart_coords)
    adj_matrix, _ = compute_adj_matrix(distance_mat, all_atom_types)
    molcif.graph = adj_matrix.todense()

    # Convert the rotation angle to radians if it is in degrees
    if is_degree:
        rot_angle_rad = rot_angle * np.pi / 180

    # list of linkers
    linker_list, _ = get_linkers(molcif, adj_matrix, all_atom_types)

    linkers_to_rotate_list = []
    for linker_num, linker in enumerate(linker_list):
        if len(linker) < 2:
            continue
        else:
            linkers_to_rotate_list.append(linker)

    # get coordinates of BDC linkers
    linker_coords = [fcoords[val,:] for val in linkers_to_rotate_list]
    coords_new = fcoords.copy()

    # Rotation of all of the linkers
    for linker_num, linker in enumerate(linkers_to_rotate_list):
        new_linker = linker_rotation(molcif, fcoords, linker, rot_angle_rad, cell_v, all_atom_types)
        coords_new[linkers_to_rotate_list[linker_num],:] = new_linker
    write_cif(f'{path2write}/{basename}_rot_{rot_angle}.cif', cpar, coords_new, all_atom_types)


### End of functions ###

def main():
    ### Example functionalizing UiO-66 below ###
    input_cif = 'UiO-66.cif'
    path2write = '.'
    rot_angle = 45

    rotate_and_write(
        input_cif=input_cif,
        path2write=path2write
        rot_angle=rot_angle,
        is_degree=True
    )

    ### End of example ###

if __name__ == "__main__":
    main()
