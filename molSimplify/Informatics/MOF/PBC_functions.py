import numpy as np
import itertools
import networkx as nx
from scipy.spatial import distance
from scipy import sparse
import copy
from molSimplify.Scripts.cellbuilder_tools import import_from_cif
from molSimplify.Informatics.MOF.atomic import (
    COVALENT_RADII,
    alkali,
    lanthanides,
    metals,
    )

# PBC: periodic boundary conditions

deg2rad = np.pi/180.0
def readcif(name):
    """
    Reads a cif file and returns information about its structure and composition.

    Parameters
    ----------
    name : str
        The path of the cif file to be read.

    Returns
    -------
    cpar : numpy.ndarray
        The parameters (i.e. lattice constants) of the MOF cell. Specifically, A, B, C, alpha, beta, and gamma. Shape is (6,).
    atomtypes : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.
    positions : numpy.ndarray
        The fractional positions of the atoms of the cif file. Shape is (number of atoms, 3).

    """
    with open(name , 'r', errors='ignore') as fi: # ignore takes care of unicode errors in some cifs
        EIF = fi.readlines()
        cond=False
        atom_props_count=0
        atomlines=[]
        counter=0
        cell_parameter_boundary=[0.0,0.0]
        for line in EIF:
            line_stripped=line.strip()
            if (not line) or line_stripped.startswith("#"):
                continue
            line_splitted=line.split()

            if line_stripped.startswith("_cell_length_a"):
                temp = line_splitted[1].replace(')','')
                temp = temp.replace('(','')
                cell_a=float(temp)
                cell_parameter_boundary[0]=counter+1
            elif line_stripped.startswith("_cell_length_b"):
                temp = line_splitted[1].replace(')','')
                temp = temp.replace('(','')
                cell_b=float(temp)
            elif line_stripped.startswith("_cell_length_c"):
                temp = line_splitted[1].replace(')','')
                temp = temp.replace('(','')
                cell_c=float(temp)
            elif line_stripped.startswith("_cell_angle_alpha"):
                temp = line_splitted[1].replace(')','')
                temp = temp.replace('(','')
                cell_alpha=float(temp)
            elif line_stripped.startswith("_cell_angle_beta"):
                temp = line_splitted[1].replace(')','')
                temp = temp.replace('(','')
                cell_beta=float(temp)
            elif line_stripped.startswith("_cell_angle_gamma"):
                temp = line_splitted[1].replace(')','')
                temp = temp.replace('(','')
                cell_gamma=float(temp)
                cell_parameter_boundary[1]=counter+1
            # if cond and line_stripped.startswith("loop_"):
            #     break
            # else:

            if line_stripped.startswith("_atom") :

                if line_stripped=="_atom_site_label" or line_stripped == '_atom_site_type_symbol':
                    cond = True # We have entered the block with the desired atom information.
                    # The reason for the or is that the order fo these lines can vary depending on cif
                if line_stripped == '_atom_site_type_symbol':
                    type_index=atom_props_count
                elif line_stripped=="_atom_site_fract_x":
                    fracx_index=atom_props_count
                elif line_stripped=="_atom_site_fract_y":
                    fracy_index=atom_props_count
                elif line_stripped=="_atom_site_fract_z":
                    fracz_index=atom_props_count
                # elif "charge" in line_stripped:
                #     charge_index=atom_props_count

                if cond:
                    atom_props_count+=1 # Another atom property in the block we are interested in.

            elif cond:

                if len(line_splitted)==atom_props_count:
                    atomlines.append(line)
                elif line == '\n':
                    continue # Allow for newlines between the _atom_ lines and the lines holding the atom information
                else:
                    break # Don't need to keep looking through the file, since we've seen all the desired information for all atoms. We left the block.

            counter+=1

        positions=[]
        atomtypes=[]
        for cn,at in enumerate(atomlines):
            ln=at.strip().split()
            positions.append([float(ln[fracx_index].replace('(','').replace(')','')),
                              float(ln[fracy_index].replace('(','').replace(')','')),
                              float(ln[fracz_index].replace('(','').replace(')',''))])
            ln[type_index] = ln[type_index].strip("_")
            at_type = ln[type_index]
            # for idx, char in enumerate(ln[type_index]): # Looking through the characters of the element symbol in order to remove any numbers
            #     if char.isdigit(): # This means one of the characters in the atom type is a number.
            #         at_type = ln[type_index][:idx] # Overwriting. Use the atom element symbol without numbers.
            #         break # Get the characters up to the number, then stop
            at_type = at_type.capitalize()
            atomtypes.append(at_type)

        cpar=np.array([cell_a,cell_b,cell_c,cell_alpha,cell_beta,cell_gamma])
        positions = np.array(positions)
        return cpar, atomtypes, positions

def compute_image_flag(cell, fcoord1, fcoord2):
    """
    Calculates how to shift fcoord2 to get it as close as possible to fcoord1. Shift by the crystal cell vectors.

    Parameters
    ----------
    cell : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    fcoord1 : numpy.ndarray
        Fractional coordinates of atom 1. Shape is (3,).
    fcoord2 : numpy.ndarray
        Fractional coordinates of atom 2. Shape is (3,).

    Returns
    -------
    supercells[image] : numpy.ndarray
        The nearest cell shift of fcoord2 to fcoord1. Shape is (3,). Values will be -1, 0, or 1.

    """
    supercells = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    fcoords = fcoord2 + supercells # 27 versions of fcoord2, shifted some cells over in different directions
    coords = np.array([np.dot(j, cell) for j in fcoords]) # Cartesian coordinates
    coord1 = np.dot(fcoord1, cell)
    dists = distance.cdist([coord1], coords) # Euclidean distance
    dists = dists[0].tolist()
    image = dists.index(min(dists)) # The image of the closest fcoord2, when considering cell shifts
    return supercells[image]


def linker_length(adjmat, anchors):
    """
    Computes the shortest and longest paths between anchors in a linker.

    Parameters
    ----------
    adjmat : numpy.matrix
        The atom connections in the linker subgraph.
    anchors : set of ints
        The indices of linker atoms that are bonded to SBUs.

    Returns
    -------
    (min_length,max_length) : tuple of ints
        min_length is the shortest path length between two anchors in a linker.
        max_length is the longest path length between two anchors in a linker.

    """
    rows, cols = np.where(adjmat == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)

    # Start max_length and min_length off with values that will most likely be overwritten.
    max_length = 0
    min_length = 1000

    for i,j in itertools.combinations(anchors, 2):
        max_length=max(len(nx.shortest_path(gr,i,j))-1,max_length)
        min_length=min(len(nx.shortest_path(gr,i,j))-1,min_length)
    return (min_length,max_length)

def slice_mat(mat, atoms):
    """
    Slice the matrix mat.

    Parameters
    ----------
    mat : numpy.matrix
        The adjacency matrix. Shape is (number of atoms, number of atoms).
    atoms : list of numpy.int32
        The indices of atoms that determine the matrix slice.

    Returns
    -------
    np.array(mat[np.ix_(list(atoms),list(atoms))]) : numpy.ndarray
        The matrix slice. Shape is (len(atoms), len(atoms)).

    """
    return np.array(mat[np.ix_(list(atoms),list(atoms))])


def ligand_detect(cell, cart_coords, adj_mat, anchorlist):
    """
    Calculates how to shift anchor atoms so that they are close to atoms bonded to them.
    I imagine this tackles the issue of two bonded atoms being on different sides of a crystal cell.

    Parameters
    ----------
    cell : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    cart_coords : numpy.ndarray
        Cartesian coordinates of the atoms in the linker or sbu. Shape is (number of atoms, 3).
    adj_mat : numpy.ndarray
        Adjacency matrix. 1 represents a bond, 0 represents no bond. Shape is (number of atoms, number of atoms).
    anchorlist : set of ints
        The indices of the anchor atoms in the linker or sbu.

    Returns
    -------
    np.array(periodic_images) : numpy.ndarray
        The cell shifts that get the anchor atoms closest to an atom (current_node) they are bonded with. Shape is (len(anchorlist), 3).

    """
    invcell=np.linalg.inv(cell)
    fcoords=np.dot(cart_coords,invcell) # fractional coordinates
    connected_components=[0] # This list will be grown to include all atoms that are part of the linker or sbu.
    checked=[] # Keeps tracked of the indices of atoms that have already been checked.
    periodic_images=[]
    if 0 in anchorlist:
        periodic_images.append(np.array([0,0,0]))
    counter=0
    while len(connected_components) < len(cart_coords):
        current_node = connected_components[counter]
        for j,v in enumerate(adj_mat[current_node]):
            if v==1 and (j not in checked) and (j not in connected_components): # If find a bonded atom that hasn't been checked yet
                image_flag = compute_image_flag(cell,fcoords[current_node],fcoords[j])
                fcoords[j] += image_flag # Shifting fractional coordinates by the number of cells specified by compute_image_flag
                connected_components.append(j)
                checked.append(j)
                if j in anchorlist:
                    periodic_images.append(image_flag)
        counter+=1

    return np.array(periodic_images)


def XYZ_connected(cell, cart_coords, adj_mat):
    """
    Calculate fractional coordinates of atoms for the specified connected component, shifted by cell vectors to make the coordinates close to each other.

    Parameters
    ----------
    cell : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    cart_coords : numpy.ndarray
        Cartesian coordinates of the atoms in this component. Shape is (number of atoms, 3).
    adj_mat : numpy.ndarray
        Adjacency matrix. 1 represents a bond, 0 represents no bond. Shape is (number of atoms, number of atoms).

    Returns
    -------
    fcoords : numpy.ndarray
        Fractional coordinates of the atoms in this component. Shape is (number of atoms, 3).

    """
    invcell=np.linalg.inv(cell)
    fcoords=np.dot(cart_coords,invcell) # fractional coordinates
    connected_components=[0] # This list will be grown to include all atoms that are part of the linker or sbu.
    checked=[] # Keeps tracked of the indices of atoms that have already been checked.
    counter=0
    from scipy import sparse
    n_components, labels_components = sparse.csgraph.connected_components(csgraph=adj_mat, directed=False, return_labels=True)
    # print(n_components,'comp',labels_components)
    tested_index = 0 # The label for the connected components. 0 indicates the first connected component, etc.
    index_counter = 0
    while len(connected_components) < len(cart_coords):
        try:
            current_node = connected_components[counter]
        except:
            indices = [i for i, x in enumerate(labels_components) if x == tested_index] # Indices corresponding to atoms in the component corresponding to tested_index
            current_node = indices[index_counter]
            # print(current_node,indices)

            if index_counter == (len(indices)-1):
                tested_index += 1
                index_counter = 0
            else:
                index_counter += 1
        for j,v in enumerate(adj_mat[current_node]):
            if v==1 and (j not in checked) and (j not in connected_components): # If find a bonded atom that hasn't been checked yet
                fcoords[j]+=compute_image_flag(cell,fcoords[current_node],fcoords[j]) # Shifting fractional coordinates by the number of cells specified by compute_image_flag
                connected_components.append(j)
                checked.append(j)
            # print(connected_components)
        counter+=1
    return fcoords

def writeXYZfcoords(filename, atoms, cell, fcoords):
    """
    Write an XYZ file using fractional coordinates.

    Parameters
    ----------
    filename : str
        The path to where the xyz of the MOF structure will be written.
    atoms : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.
    cell : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    fcoords : numpy.ndarray
        The fractional positions of the atoms of the cif file. Shape is (number of atoms, 3).

    Returns
    -------
    None

    """
    with open(filename,"w") as fo:
        fo.write("%i\n\n"%len(atoms))
        for i,fcoord in enumerate(fcoords):
            cart_coord=np.dot(fcoord,cell)
            s="%10.2f %10.2f %10.2f"%(cart_coord[0],cart_coord[1],cart_coord[2])
            fo.write("%s %s\n"%(atoms[i],s))

def writeXYZandGraph(filename, atoms, cell, fcoords, molgraph):
    """
    Write the xyz file for the MOF structure, as well as the net file containing the MOF's graph.

    Parameters
    ----------
    filename : str
        The path to where the xyz of the MOF structure will be written.
    atoms : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.
    cell : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    fcoords : numpy.ndarray
        The fractional positions of the atoms of the cif file. Shape is (number of atoms, 3).
    molgraph : numpy.matrix or numpy.ndarray
        The adjacency matrix, which indicates which atoms are connected to which atoms. Shape is (number of atoms, number of atoms).

    Returns
    -------
    None

    """

    with open(filename,"w") as fo:
        fo.write("%i\n\n"%len(atoms)) # The first line indicates the number of atoms in the cell of the structure.
        for i,fcoord in enumerate(fcoords):
            cart_coord=np.dot(fcoord,cell) # Go from fractional coordinates to Cartesian coordinates.
            s="%10.2f %10.2f %10.2f"%(cart_coord[0],cart_coord[1],cart_coord[2]) # X, Y, Z
            fo.write("%s %s\n"%(atoms[i],s)) # Writing the coordinates of each atom.
    tmpstr=",".join([at for at in atoms])
    np.savetxt(filename[:-4]+".net",molgraph,fmt="%i",delimiter=",",header=tmpstr) # Save a net file.


def returnXYZandGraph(filename, atoms, cell, fcoords, molgraph):
    """
    TODO

    Parameters
    ----------
    TODO : TODO
        TODO
    TODO : TODO
        TODO
    TODO : TODO
        TODO

    Returns
    -------
    TODO : TODO
        TODO
    TODO : TODO
        TODO
    TODO : TODO
        TODO

    """
    coord_list = []
    for i,fcoord in enumerate(fcoords):
        cart_coord=np.dot(fcoord,cell)
        coord_list.append([cart_coord[0],cart_coord[1],cart_coord[2]])
    tmpstr=",".join([at for at in atoms])
    if filename != None:
        np.savetxt(filename[:-4]+".net",molgraph,fmt="%i",delimiter=",",header=tmpstr)
    return coord_list, molgraph

def writeXYZcoords(filename, atoms, coords):
    """
    Write an XYZ file using Cartesian coordinates.

    Parameters
    ----------
    filename : str
        The path to where the xyz of the MOF structure will be written.
    atoms : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.
    coords : numpy.ndarray
        The Cartesian positions of the atoms of the cif file. Shape is (number of atoms, 3).

    Returns
    -------
    None

    """
    with open(filename,"w") as fo:
        fo.write("%i\n\n"%len(atoms))
        for i,cart_coord in enumerate(coords):
            s="%10.2f %10.2f %10.2f"%(cart_coord[0],cart_coord[1],cart_coord[2])
            fo.write("%s %s\n"%(atoms[i],s))
    return

def writeXYZcoords_withcomment(filename, atoms, coords, comment):
    """
    Write an XYZ file using Cartesian coordinates, with a comment included.

    Parameters
    ----------
    filename : str
        The path to where the xyz of the MOF structure will be written.
    atoms : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.
    coords : numpy.ndarray
        The Cartesian positions of the atoms of the cif file. Shape is (number of atoms, 3).
    comment : str
        The comment to include in the XYZ file.

    """
    with open(filename,"w") as fo:
        fo.write("%i\n"%len(atoms))
        fo.write("%s\n"%comment)
        for i,cart_coord in enumerate(coords):
            s="%10.2f %10.2f %10.2f"%(cart_coord[0],cart_coord[1],cart_coord[2])
            fo.write("%s %s\n"%(atoms[i],s))
    return

def write2file(pt, fn, st):
    """
    Writes the string st to a file.

    Parameters
    ----------
    pt : str
        Path of the folder to make a file in.
    fn : str
        Name of the file to write to.
    st : str
        What to write in the file.

    Returns
    -------
    None

    """
    with open(pt+fn, "a") as fo:
        fo.write(st)

def write_cif(fname, cellprm, fcoords, atom_labels):
    """
    Writes a cif file with the provided parameters.

    Parameters
    ----------
    fname : str
        The path to the cif file to be written.
    cellprm : numpy.ndarray
        The parameters (i.e. lattice constants) of the MOF cell. Specifically, A, B, C, alpha, beta, and gamma. Shape is (6,).
    fcoords : numpy.ndarray
        The fractional positions of the atoms of the cif file. Shape is (number of atoms, 3).
    atom_labels : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.

    Returns
    -------
    None

    """
    with open(fname,'w') as f_cif:
       f_cif.write("data_I\n")
       f_cif.write("_chemical_name_common  \'%s\'\n"%(fname.strip(".cif")))
       f_cif.write("_cell_length_a %8.05f\n"%(cellprm[0]))
       f_cif.write("_cell_length_b %8.05f\n"%(cellprm[1]))
       f_cif.write("_cell_length_c %8.05f\n"%(cellprm[2]))
       f_cif.write("_cell_angle_alpha %4.05f\n"%(cellprm[3]))
       f_cif.write("_cell_angle_beta  %4.05f\n"%(cellprm[4]))
       f_cif.write("_cell_angle_gamma %4.05f\n"%(cellprm[5]))
       f_cif.write("_space_group_name_H-M_alt      \'P 1\'\n\n\n")
       f_cif.write("loop_\n_space_group_symop_operation_xyz\n  'x, y, z' \n\n")
       f_cif.write("loop_\n")
       f_cif.write("_atom_site_label\n")
       f_cif.write("_atom_site_fract_x\n")
       f_cif.write("_atom_site_fract_y\n")
       f_cif.write("_atom_site_fract_z\n")
       f_cif.write("_atom_site_type_symbol\n")
       for i,atom in enumerate(atom_labels):
           f_cif.write("%-5s %8s %8s %8s %5s\n"%(atom,fcoords[i,0],fcoords[i,1],fcoords[i,2],"%s"%(atom)))

def cell_to_cellpar(cell, radians=False):
    """
    TODO

    Parameters
    ----------
    TODO : TODO
        TODO
    TODO : TODO
        TODO
    TODO : TODO
        TODO

    Returns
    -------
    TODO : TODO
        TODO
    TODO : TODO
        TODO
    TODO : TODO
        TODO

    """
    lengths = [np.linalg.norm(v) for v in cell]
    angles = []
    for i in range(3):
        j = i - 1
        k = i - 2
        ll = lengths[j] * lengths[k]
        if ll > 1e-16:
            x = np.dot(cell[j], cell[k]) / ll
            angle = 180.0 / np.pi * np.arccos(x)
        else:
            angle = 90.0
        angles.append(angle)
    if radians:
        angles = [angle * np.pi / 180 for angle in angles]
    return np.array(lengths + angles)

def findPaths(G, u, n):
    """
    Finds paths between atom u and atoms n bonds away.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        networkx graph for the linker of interest.
    u : int
        The index of the anchor atom's index in the linker list of indices.
    n : int
        How many bonds away one functionalized atom should be from another.

    Returns
    -------
    paths : list of list of int
        Inner lists will be length four, if n is three. All inner lists start with u.
        Note, may return [[u]] instead if n is zero. [[u]] is a list of list of int.

    """
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path] # recursive
        # if u not in path ensures no atom is used twice in a path.
        # Example of paths: [[12, 3, 7, 6], [12, 3, 7, 14], [12, 4, 0, 14], [12, 4, 0, 15], [12, 4, 9, 5], [12, 4, 9, 11]]
    return paths

def fractional2cart(fcoords, cell):
    """
    Convert from fractional coordinates to Cartesian coordinates.

    Parameters
    ----------
    fcoords : numpy.ndarray
        The fractional positions of the atoms of the cif file. Shape is (number of atoms, 3).
    cell : The three Cartesian vectors representing the edges of the crystal cell.
        Shape is (3,3).

    Returns
    -------
    np.dot(fcoords,cell) : numpy.ndarray
        The Cartesian coordinates of the crystal atoms. Shape is (number of atoms, 3).

    """
    return np.dot(fcoords,cell)

def frac_coord(coord, cell):
    """
    Convert from Cartesian coordinates to fractional coordinates.

    Parameters
    ----------
    coord : numpy.ndarray
        The Cartesian coordinates of the atoms of the cif file. Shape is (number of atoms, 3).
    cell : The three Cartesian vectors representing the edges of the crystal cell.
        Shape is (3,3).

    Returns
    -------
    np.dot(coord,invcell) : numpy.ndarray
        The fractional positions of the crystal atoms. Shape is (number of atoms, 3).

    """
    invcell=np.linalg.inv(cell)
    return np.dot(coord,invcell)

def compute_distance_matrix3(cell, cart_coords, num_cells=1):
    """
    Computes the pairwise distances between all atom pairs in the crystal cell.

    Parameters
    ----------
    cell : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    cart_coords : numpy.ndarray
        The Cartesian coordinates of the crystal atoms. Shape is (number of atoms, 3).
    num_cells : int
        The number of crystal cells to put together for the evaluation of distances.

    Returns
    -------
    distance_matrix : numpy.ndarray
        The distance of each atom to each other atom. Shape is (number of atoms, number of atoms).

    """
    pos = np.arange(-num_cells, num_cells+1, 1) # [-1, 0, 1] if num_cells is 1
    combos = np.array(np.meshgrid(pos, pos, pos)).T.reshape(-1,3) # The 27 combinations of -1, 0, 1 if num_cells is 1
    shifts = np.sum(np.expand_dims(cell, axis=0)*np.expand_dims(combos, axis=-1), axis=1) # The possible shifts by the crystal cell vectors.
    # NxNxCells distance array
    shifted = np.expand_dims(cart_coords, axis=1) + np.expand_dims(shifts, axis=0) # The shifted Cartesian coordinates. Shape is (number of atoms, number of combinations in combos, 3)

    # The distances between atoms, across different crystal cell shifts, for the three Cartesian dimensions.
    dist = np.expand_dims(np.expand_dims(cart_coords, axis=1), axis=1) - np.expand_dims(shifted, axis=0) # Shape is (number of atoms, number of atoms, number of combinations in combos, 3)
        # The shape of np.expand_dims(np.expand_dims(cart_coords, axis=1), axis=1) is (number of atoms, 1, 1, 3)
        # The shape of np.expand_dims(shifted, axis=0) is (1, number of atoms, number of combinations in combos, 3)
        # numpy subtraction expands out the axes of length one for the subtraction.

    # The standard distance formula of square root of x^2 + y^2 + z^2
    dist = np.sqrt(np.sum(np.square(dist), axis=-1)) # Shape is (number of atoms, number of atoms, number of combinations in combos)

    # But we want only the minimum
    distance_matrix = np.min(dist, axis=-1) # Consider the distance between two atoms at the crystal cell shift where they are closest.
    return distance_matrix

def position_nearest_atom(cell, cart_coords, index_of_interest, num_cells=1):
    """
    Computes the pairwise distances between all atoms in the crystal cell to the atom specified by index_of_interest; returns the position of the nearest atom.

    Parameters
    ----------
    cell : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    cart_coords : numpy.ndarray
        The Cartesian coordinates of the crystal atoms. Shape is (number of atoms, 3).
    index_of_interest : int
        The index of the atom to which we want to find the nearest atom's position.
    num_cells : int
        The number of crystal cells to put together for the evaluation of distances.

    Returns
    -------
    nearest_position : numpy.ndarray
        The Cartesian coordinates of the nearest atom. Shape is (3,).
    nearest_index : numpy.int64
        The index of the nearest atom.
    shift_for_nearest_atom : numpy.ndarray
        The crystal cell shifts that position the nearest atom closest to the atom of interest. Shape is (3,). Will look something like [-1  0 -1] or [0 0 0] or etc.

    """
    pos = np.arange(-num_cells, num_cells+1, 1) # [-1, 0, 1] if num_cells is 1
    combos = np.array(np.meshgrid(pos, pos, pos)).T.reshape(-1,3) # The 27 combinations of -1, 0, 1 if num_cells is 1
    shifts = np.sum(np.expand_dims(cell, axis=0)*np.expand_dims(combos, axis=-1), axis=1) # The possible shifts by the crystal cell vectors.
    # NxNxCells distance array
    shifted = np.expand_dims(cart_coords, axis=1) + np.expand_dims(shifts, axis=0) # The shifted Cartesian coordinates. Shape is (number of atoms, number of combinations in combos, 3)

    # The distances between atoms, across different crystal cell shifts, for the three Cartesian dimensions.
    dist = np.expand_dims(np.expand_dims(cart_coords[index_of_interest], axis=0), axis=0) - shifted # Shape is (number of atoms, number of combinations in combos, 3)
        # The shape of np.expand_dims(np.expand_dims(cart_coords[index_of_interest], axis=0), axis=0) is (1, 1, 3). These are the coordinates of the atom of interest.
        # numpy subtraction expands out the axes of length one for the subtraction.

    # The standard distance formula of square root of x^2 + y^2 + z^2
    dist = np.sqrt(np.sum(np.square(dist), axis=-1)) # Shape is (number of atoms, number of combinations in combos)

    # Want the atom that is closest to index_of_interest, given the ideal shift
    # Don't want to consider distance of atom of interest to itself, so I eliminate it from consideration this way.
    dist[index_of_interest,:] = np.array([np.Inf]*np.shape(dist)[1])
    # Find the index of the closest atom.
    index_nearest_atom = np.argmin(dist)
    index_nearest_atom = np.unravel_index(index_nearest_atom, np.shape(dist)) # This is (atom index, shift index)

    # Get the Cartesian coordinates of the nearest atom
    nearest_position = shifted[index_nearest_atom[0], index_nearest_atom[1], :]
    nearest_index = index_nearest_atom[0]
    shift_for_nearest_atom = combos[index_nearest_atom[1],:]

    return nearest_position, nearest_index, shift_for_nearest_atom

def make_graph_from_nodes_edges(nodes, edges, attribs):
    """
    TODO

    Parameters
    ----------
    TODO : TODO
        TODO
    TODO : TODO
        TODO
    TODO : TODO
        TODO

    Returns
    -------
    TODO : TODO
        TODO
    TODO : TODO
        TODO
    TODO : TODO
        TODO

    """
    gr = nx.Graph()
    [gr.add_node(n,atomicNum=at) for n,at in zip(nodes,attribs)]
    #gr.add_nodes_from(nodes)
    gr.add_edges_from(edges)
    return gr

def mkcell(cpar):
    """
    Update the cell representation to match the parameters.

    Parameters
    ----------
    cpar : numpy.ndarray
        The parameters (i.e. lattice constants) of the MOF cell. Specifically, A, B, C, alpha, beta, and gamma. Shape is (6,).

    Returns
    -------
    vectors : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).

    """

    a_mag, b_mag, c_mag = cpar[:3]
    alpha, beta, gamma = [x * deg2rad for x in cpar[3:]] # Converting the angles to radians from degrees.
    a_vec = np.array([a_mag, 0.0, 0.0]) # a_vec is taken to be along the x axis
    b_vec = np.array([b_mag * np.cos(gamma), b_mag * np.sin(gamma), 0.0]) # See this depiction of lattice parameters for reasoning behind these equations. https://www.doitpoms.ac.uk/tlplib/crystallography3/parameters.php. b_vec is taken to be in the X-Y plane.
    c_x = c_mag * np.cos(beta)
    c_y = c_mag * (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma) # You have to use a matrix to convert. This is derived in most textbooks on crystallography, such as McKie & McKie 'Essentials of Crystallography'. https://chemistry.stackexchange.com/questions/136836/converting-fractional-coordinates-into-cartesian-coordinates-for-crystallography
    c_vec = np.array([c_x, c_y, (c_mag**2 - c_x**2 - c_y**2)**0.5]) # c_x**2 + c_y**2 + c_z**2 = c_mag**2
    vectors = np.array([a_vec, b_vec, c_vec])
    return vectors

def make_supercell(cell, atoms, fcoords, exp_coeff):
    """
    TODO

    Parameters
    ----------
    TODO : TODO
        TODO
    TODO : TODO
        TODO
    TODO : TODO
        TODO

    Returns
    -------
    TODO : TODO
        TODO
    TODO : TODO
        TODO
    TODO : TODO
        TODO

    """
    supercell = np.multiply(cell.T, exp_coeff).T
    superatoms=[]
    superfcoords=[]
    for i in range(exp_coeff[0]):
        for j in range(exp_coeff[1]):
            for k in range(exp_coeff[2]):
                for na,atom in enumerate(atoms):
                    fc=fcoords[na]
                    fx = fc[0]/exp_coeff[0] + float(i)/exp_coeff[0]
                    fy = fc[1]/exp_coeff[1] + float(j)/exp_coeff[1]
                    fz = fc[2]/exp_coeff[2] + float(k)/exp_coeff[2]
                    superfcoords.append([fx,fy,fz])
                    superatoms.append(atom)
    superfcoords= np.array(superfcoords)
    return supercell,superatoms,superfcoords


def compute_adj_matrix(distance_mat, allatomtypes, wiggle_room=1, handle_overlap=False):
    """
    Calculates what atoms are bonded to each other.

    Bonding is trickier in MOFs than in TM complexes due to metal-metal bonding, motivating the existence of this function
    even though a similar one exists in mol3D.

    Parameters
    ----------
    distance_mat : numpy.ndarray
        The distance of each atom to each other atom. Shape is (number of atoms, number of atoms).
    allatomtypes : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.
    wiggle_room : float
        A multiplier that allows for more or less strict bond distance cutoffs.
    handle_overlap : bool
        Indicates whether to provide diagnostics for atoms that are overlapping, or to just raise an error if it finds overlaps

    Returns
    -------
    sparse.csr_matrix(adj_matrix) : scipy.sparse.csr.csr_matrix
        Adjacency matrix. 1 represents a bond, 0 represents no bond. Shape is (number of atoms, number of atoms).
    overlap_atoms : list
        Indices of atoms that overlap with any atom of a lower index.

    """

    overlap_atoms = []
    adj_matrix=np.zeros(distance_mat.shape)
    for i,e1 in enumerate(allatomtypes[:-1]): # Iterating through all pairs of atoms.
        for j,e2 in enumerate(allatomtypes[i+1:]):
            elements = set([e1, e2])

            # In the context of sets, < means that all the items in the set elements is in the set metals, for example.
            if (elements < metals): # FIXME no metal-metal bond allowed
                continue

            rad = (COVALENT_RADII[e1] + COVALENT_RADII[e2])
            dist = distance_mat[i,i+j+1]
            # check for atomic overlap:
            if dist < min(COVALENT_RADII[e1] , COVALENT_RADII[e2]):
                print(f"Atomic overlap involving atom {i} and {i+j+1}! Zero-indexed.")
                print(f"dist is {dist} and the cutoff is {min(COVALENT_RADII[e1] , COVALENT_RADII[e2])}")
                if handle_overlap:
                    # Check whether atom i is already in overlap_atoms and will be removed.
                    # If so, no need to remove an atom that overlaps with atom i.
                    if i not in overlap_atoms:
                        overlap_atoms.append(i+j+1) # The atom with index i+j+1 overlapped with another atom.
                else:
                    print('Overlapping atoms! Error')
                    raise NotImplementedError # Exit the function.
            tempsf = 0.9 # This is modified below under certain conditions, to account for looser or tigher bonding.
            # There is probably a better way to fix these kinds of issues.
            # In the context of sets, & is the intersection. If the intersection is null, the (&) expression is False.
            if (set("F") < elements) and  (elements & metals): # One of the members of elements is fluorine, and one is a metal.
                tempsf = 0.8
            if (set("C") < elements) and  (elements & metals):
                tempsf = 0.95
            if (set("H") < elements) and  (elements & metals) and (not elements & alkali):
                tempsf = 0.75

            if (set("O") < elements) and (elements & metals):
                tempsf = 0.85
            if (set("N") < elements) and (elements & metals):
                tempsf = 0.82
            # fix for water particle recognition.
            if(set(["O", "H"]) <= elements):
                tempsf = 0.8
            # very specific fix for Michelle's amine appended MOF
            if(set(["N","H"]) <= elements):
                tempsf = 0.67
            if(set(["Mg","N"]) <= elements):
                tempsf = 0.80
            if(set(["C","H"]) <= elements):
                tempsf = 0.80
            if(set(["K"]) <= elements):
                tempsf = 0.95
            if(lanthanides & elements):
                tempsf = 0.95
            if(elements ==set(["C"]) ):
                tempsf = 0.85
            if dist*tempsf < rad * wiggle_room: # and not (alkali & elements):
                # Entering this if statement means there is a bond between the two atoms.
                adj_matrix[i,i+j+1]=1
                adj_matrix[i+j+1,i]=1

        # Removing duplicates and sorting.
        overlap_atoms = [*set(overlap_atoms)]
    return sparse.csr_matrix(adj_matrix), overlap_atoms



def get_closed_subgraph(linkers, SBUlist, adj_matrix):
    ###############################################################################
    # This part separates the linkers into their respective subgraphs             #
    # First element is the things you want to find subgraphs of.                  #
    # If this is the linkers, you input that as the first.                        #
    # If you input the SBU as the first, then you get the subgraphs of the SBU.   #
    # The second element tells you what part of the matrix is NOT what you want.  #
    # If we want subgraphs of linkers, we want to exclude the SBU.                #
    ###############################################################################
    """

    Parameters
    ----------
    linkers : set of int
        Indices corresponding to atoms in the linkers (or SBUs; see the summary part of this docstring) of the MOF. The part of the matrix to analyze.
    SBUlist : set of numpy.int64
        Indices corresponding to atoms in the SBUs (or linkers) of the MOF. The part of the matrix to ignore.
    adj_matrix : scipy.sparse.csr.csr_matrix
        Adjacency matrix. 1 represents a bond, 0 represents no bond. Shape is (number of atoms, number of atoms).

    Returns
    -------
    linker_list : list of lists of ints
        Each inner list is its own separate linker (or SBU). The ints are the atom indices of that linker (or SBU). Length is # of linkers (or SBUs).
    linker_subgraphlist : list of scipy.sparse.csr.csr_matrix
        The atom connections in the linker (or SBU) subgraph. Length is # of linkers (or SBUs).

    """

    linkers_sub = linkers.copy()
    linker_list = []
    linker_subgraphlist = []
    counter = 0
    while len(linkers_sub)>0:
        # Every time this while loop is entered, an entire linker will be identified.
        counter += 1
        if counter > 5000:
            break
        start_idx = list(linkers_sub)[0] # index of an atom belonging to the linkers
        current_linker_list = set([start_idx]) # Linker atoms will be added to this set as they are discovered.
        checked_list = set() # Will contain all of the indices that have already been tried as start_idx.
        while len(checked_list) <= len(current_linker_list):
            loop_over = np.nonzero(adj_matrix[start_idx])[1] # indices of atoms with bonds to the atom with the index start_idx
            current_linker_list.update(loop_over)
            current_linker_list = current_linker_list-SBUlist
            checked_list.add(start_idx)
            for val in loop_over:
                if (not val in SBUlist):
                    current_linker_list.update(np.nonzero(adj_matrix[val])[1]) # np.nonzero(adj_matrix[val])[1] are the indices of atoms with bonds to the atom with index val
            left_to_check = current_linker_list-checked_list-SBUlist # Linker atoms whose connecting atoms still need to be checked.
            if len(left_to_check) == 0:
                break
            else:
                start_idx = list(left_to_check)[0] # update start_idx for the next pass through the while loop
        current_linker_list = current_linker_list - SBUlist
        linkers_sub = linkers_sub - current_linker_list
        ####### We want to return both the linker itself as well as the subgraph corresponding to it.
        linker_list.append(list(current_linker_list))
        linker_subgraphlist.append(adj_matrix[np.ix_(list(current_linker_list),list(current_linker_list))])

    return linker_list, linker_subgraphlist

def include_extra_shells(SBUlists, subgraphlists, molcif, adjmat):
    """
    Include extra atoms in the SBUs. One more shell.

    Parameters
    ----------
    SBUlists : list of lists of ints
        Each inner list is its own separate SBU. The ints are the atom indices of that SBU. Length is # of SBUs.
    subgraphlists : list of scipy.sparse.csr.csr_matrix
        The atom connections in the SBU subgraph. Length is # of SBUs.
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    adjmat : scipy.sparse.csr.csr_matrix
        1 represents a bond, 0 represents no bond. Shape is (number of atoms, number of atoms).

    Returns
    -------
    SBUs : list of lists of numpy.int64
        The expanded atom indices of each SBU.
    subgraphs : list of scipy.sparse.csr.csr_matrix
        The atom bonding information of the SBUs in the variable `SBUs`. Which atoms are bonded to which.

    """

    SBUs=[]
    subgraphs=[]
    for SBU in SBUlists:
        for zero_first_shell in copy.deepcopy(SBU):
            for val in molcif.getBondedAtomsSmart(zero_first_shell):
                SBU.append(val) # Include in the SBU every atom that is bonded to the SBU
        SBUset = set(SBU) # Removing duplicate atom indices.
        SBUs.append(list(SBUset))
        subgraphs.append(adjmat[np.ix_(list(SBUset),list(SBUset))])

    return SBUs, subgraphs

def disorder_detector(name):
    """
    Reads a cif file and returns information on which atoms have fractional occupancy.

    Parameters
    ----------
    name : str
        The path of the cif file to be read.

    Returns
    -------
    disordered_atom_indices : list of ints
        The indices of atoms with fractional occupancies.
    disordered_atom_types : list of str
        The elemental symbols of atoms with fractional occupancies.
    disordered_atom_occupancies : list of floats
        The fractional occupancies of the atoms with fractional occupancies.

    """
    with open(name , 'r', errors='ignore') as fi: # ignore takes care of unicode errors in some cifs
        EIF = fi.readlines()
        cond=False
        occupancy_index=False
        atom_props_count=0
        atomlines=[]
        for line in EIF:
            line_stripped=line.strip()
            if (not line) or line_stripped.startswith("#"):
                continue
            line_splitted=line.split()

            if line_stripped.startswith("_atom") :

                if line_stripped == "_atom_site_label" or line_stripped == '_atom_site_type_symbol':
                    cond=True # We have entered the block with the desired atom information.
                    # The reason for the or is that the order fo these lines can vary depending on cif
                if line_stripped == '_atom_site_type_symbol':
                    type_index=atom_props_count
                elif line_stripped=="_atom_site_occupancy":
                    occupancy_index=atom_props_count

                if cond:
                    atom_props_count+=1 # Another atom property in the block we are interested in.

            elif cond:
                if len(line_splitted)==atom_props_count:
                    atomlines.append(line)
                else:
                    break # Don't need to keep looking through the file, since we've seen all the desired information for all atoms. We left the block.


        disordered_atom_indices = []
        disordered_atom_types = []
        disordered_atom_occupancies = []

        if occupancy_index != False: # This means that occupancy information is available
            for idx, at in enumerate(atomlines): # Go through the lines of the cif with atom specific information. Atom by atom.
                ln=at.strip().split()

                current_atom_occupancy = ln[occupancy_index].split('(')[0] # Excluding parentheses in order to convert to float.
                current_atom_occupancy = float(current_atom_occupancy)

                if current_atom_occupancy != 1: # Disordered atom

                    disordered_atom_indices.append(idx)

                    ln[type_index] = ln[type_index].strip("_")
                    at_type = ln[type_index]
                    disordered_atom_types.append(at_type)

                    disordered_atom_occupancies.append(current_atom_occupancy)

        return disordered_atom_indices, disordered_atom_types, disordered_atom_occupancies

def remove_duplicate_atoms(allatomtypes, fcoords):
    """
    Removes any atoms that have the exact same coordinate as a lower index atom.
    This pops up after removing symmetry with Vesta. Symmetry removal helps the molSimplify code get connectivity right.

    Parameters
    ----------
    allatomtypes : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.
    fcoords : numpy.ndarray
        The fractional positions of the atoms of the cif file. Shape is (number of atoms, 3).

    Returns
    -------
    allatomtypes_trim : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.
        All duplicate atoms removed.
    fcoords_trim : numpy.ndarray
        The fractional positions of the atoms of the cif file. Shape is (number of atoms, 3).
        All duplicate atoms removed.

    """

    # Get the unique fractional coordinate 3-tuples.
    fcoords_trim, indices = np.unique(fcoords, axis=0, return_index=True)
    # Get the atom types of the unique fractional coordinates.
    allatomtypes_trim = [allatomtypes[_i] for _i in indices]

    return allatomtypes_trim, fcoords_trim

def remove_undesired_atoms(undesired_indices, allatomtypes, fcoords):
    """
    Takes a list of indices, and removes those elements from allatomtypes and fcoords.

    Parameters
    ----------
    undesired_indices : list
        The indices of the atoms to remove.
    allatomtypes : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.
    fcoords : numpy.ndarray
        The fractional positions of the atoms of the cif file. Shape is (number of atoms, 3).

    Returns
    -------
    allatomtypes_trim : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.
        All undesired atoms removed.
    fcoords_trim : numpy.ndarray
        The fractional positions of the atoms of the cif file. Shape is (number of atoms, 3).
        All undesired atoms removed.

    """
    number_of_atoms = len(allatomtypes)
    desired_indices = [_i for _i in list(range(number_of_atoms)) if (_i not in undesired_indices)] # The indices we want to keep.
    allatomtypes_trim = [value for (_i, value) in enumerate(allatomtypes) if (_i in desired_indices)]
    fcoords_trim = fcoords[desired_indices]

    return allatomtypes_trim, fcoords_trim


def overlap_removal(cif_path, new_cif_path):
    """
    Reads a cif file, removes overlapping atoms, and writes the cif to the provided path.
    For a new CIF, recommended to remove symmetry (either with Vesta or with get_primitive), then run overlap_removal, then run solvent_removal.

    Parameters
    ----------
    cif_path : str
        The path of the cif file to be read.
    new_cif_path : str
        The path to which the modified cif file will be written.

    Returns
    -------
    None

    """

    # Much of this code parallels that in the beginning of the MOF_descriptors.get_MOF_descriptors function

    # Loading the cif and getting information about the crystal cell.
    cpar, allatomtypes, fcoords = readcif(cif_path)
    allatomtypes, fcoords = remove_duplicate_atoms(allatomtypes, fcoords)
    cell_v = mkcell(cpar)
    cart_coords = fractional2cart(fcoords, cell_v)
    # if len(cart_coords) > 2000: # Don't deal with large cifs because of computational resources required for their treatment.
    #     raise Exception("Too large of a cif file")

    # Assuming that the cif does not have graph information of the structure.
    distance_mat = compute_distance_matrix3(cell_v,cart_coords)
    adj_matrix, overlap_atoms = compute_adj_matrix(distance_mat, allatomtypes, handle_overlap=True)

    # Dealing with the case of overlapping atoms.
    if len(overlap_atoms) != 0:
        print('Dealing with overlap')
        allatomtypes, fcoords = remove_undesired_atoms(overlap_atoms, allatomtypes, fcoords)

    # Writing the cif files
    write_cif(new_cif_path,cpar,fcoords,allatomtypes)

def solvent_removal(cif_path, new_cif_path, wiggle_room=1):
    """
    Reads a cif file, removes floating solvent and overlapping atoms, and writes the cif to the provided path.
    Assumes cif has P1 symmetry.

    Parameters
    ----------
    cif_path : str
        The path of the cif file to be read.
    new_cif_path : str
        The path to which the modified cif file will be written.
    wiggle_room : float
        A multiplier that allows for more or less strict bond distance cutoffs.
        Useful for some trouble CIFs with long bonds.

    Returns
    -------
    None

    """

    # Much of this code parallels that in the beginning of the MOF_descriptors.get_MOF_descriptors function

    # Loading the cif and getting information about the crystal cell.
    cpar, allatomtypes, fcoords = readcif(cif_path)
    cell_v = mkcell(cpar)
    cart_coords = fractional2cart(fcoords, cell_v)
    # if len(cart_coords) > 2000: # Don't deal with large cifs because of computational resources required for their treatment.
    #     raise Exception("Too large of a cif file")

    # Assuming that the cif does not have graph information of the structure.
    distance_mat = compute_distance_matrix3(cell_v,cart_coords)
    try:
        adj_matrix, _ = compute_adj_matrix(distance_mat, allatomtypes, wiggle_room=wiggle_room, handle_overlap=False)
    except NotImplementedError:
        raise Exception("Failed due to atomic overlap")

    # Getting the adjacency matrix (bond information).
    adj_matrix = sparse.csr_matrix(adj_matrix)
    molcif,_,_,_,_ = import_from_cif(cif_path, True) # molcif is a mol3D class of a single unit cell (or the cell of the cif file)
    molcif.graph = adj_matrix.todense()

    # Finding the connected components
    n_components, labels_components = sparse.csgraph.connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    print(f'n_components: {n_components}')
    print(f'labels_components: {labels_components}')
    print(f'len is {len(labels_components)}')
    metal_list = set([at for at in molcif.findMetal(transition_metals_only=False)]) # the atom indices of the metals
    if not len(metal_list) > 0:
        raise Exception("No metal in the structure.")

    solvent_indices = [] # This list will be filled in with the indices of solvent atoms.

    for comp in range(n_components):
        inds_in_comp = [i for i in range(len(labels_components)) if labels_components[i]==comp]
        if not set(inds_in_comp) & metal_list: # In the context of sets, & is the intersection. If the intersection is null, the (&) expression is False; the `not` would then make it True.
            # If this if statement is entered, there is an entire connected component that has no metals in it. No connections to any metal. I.e. solvent.
            solvent_indices.extend(inds_in_comp)

    # Removing the atoms corresponding to the solvent.
    allatomtypes, fcoords = remove_undesired_atoms(solvent_indices, allatomtypes, fcoords)

    # print(f'The solvent indices are {solvent_indices}')

    # Writing the cif files
    write_cif(new_cif_path,cpar,fcoords,allatomtypes)






##### Deprecated #####

# The functions compute_distance_matrix, compute_distance_matrix2, and compute_distance_matrix3 all do the same thing.
# However, compute_distance_matrix3 is significantly faster than compute_distance_matrix2, which in turn is faster than compute_distance_matrix.
# This is due to the use of for loops in compute_distance_matrix and compute_distance_matrix2, versus the vectorized (pre-compiled C code) numpy functions in compute_distance_matrix3.

def compute_distance_matrix(cell, cart_coords):
    """
    Computes the pairwise distances between all atom pairs in the crystal cell. First version of this function.

    Parameters
    ----------
    cell : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    cart_coords : numpy.ndarray
        The Cartesian coordinates of the crystal atoms. Shape is (number of atoms, 3).

    Returns
    -------
    distance_matrix : numpy.ndarray
        The distance of each atom to each other atom. Shape is (number of atoms, number of atoms).

    """
    distance_matrix=np.zeros([len(cart_coords),len(cart_coords)]) # This array will be filled in.
    for i in range(len(cart_coords)): # Looping through all combinations of atoms.
        for j in range(i+1,len(cart_coords)):
            d=min_img_distance(cart_coords[i],cart_coords[j],cell)
            distance_matrix[i,j]=d # Filling in the distance numpy array.
            distance_matrix[j,i]=d

    return distance_matrix

def min_img_distance(coords1, coords2, cell):
    """
    Calculates the distance between two atoms specified by coords1 and coords2.
    The minimum image distance is taken, meaning the shortest distance between the two atoms with consideration of the repeating periodic structure of the MOF.

    Parameters
    ----------
    coords1 : numpy.ndarray
        The Cartesian coordinates of the first atom under consideration. Shape is (3,).
    coords2 : numpy.ndarray
        The Cartesian coordinates of the second atom under consideration. Shape is (3,).
    cell : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).

    Returns
    -------
    np.linalg.norm(four) : numpy.float64
        The distance between the two atoms.

    """
    invcell=np.linalg.inv(cell) # The inverse cell parameters
    one = np.dot(coords1,invcell) % 1 # Fractional coordinates. % is modulo.
    two = np.dot(coords2,invcell) % 1 # Fractional coordinates.
    three = np.around(one - two) # numpy array of three entries. Possible values of entries are -1, 0, and 1. Corresponds to the crystal cell shift that gets the two atoms the closest.
    four = np.dot(one - two - three, cell) # Converting back to Cartesian coordinates from fractional.
    return np.linalg.norm(four)

def compute_distance_matrix2(cell, cart_coords):
    """
    Computes the pairwise distances between all atom pairs in the crystal cell. Second version of this function.

    Parameters
    ----------
    cell : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    cart_coords : numpy.ndarray
        The Cartesian coordinates of the crystal atoms. Shape is (number of atoms, 3).

    Returns
    -------
    distance_matrix : numpy.ndarray
        The distance of each atom to each other atom. Shape is (number of atoms, number of atoms).

    """
    distance_matrix=np.zeros([len(cart_coords),len(cart_coords)]) # This array will be filled in.
    for i in range(len(cart_coords)): # Looping through all combinations of atoms.
        for j in range(i+1,len(cart_coords)):
            d=min_img_distance2(cart_coords[i],cart_coords[j],cell)
            distance_matrix[i,j]=d # Filling in the distance numpy array.
            distance_matrix[j,i]=d

    return distance_matrix

def min_img_distance2(coords1, coords2, cell):
    """
    Calculates the distance between two atoms specified by coords1 and coords2.
    The minimum image distance is taken, meaning the shortest distance between the two atoms with consideration of the repeating periodic structure of the MOF.

    Parameters
    ----------
    coords1 : numpy.ndarray
        The Cartesian coordinates of the first atom under consideration. Shape is (3,).
    coords2 : numpy.ndarray
        The Cartesian coordinates of the second atom under consideration. Shape is (3,).
    cell : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).

    Returns
    -------
    np.amin(dists) : numpy.float64
        The distance between the two atoms.

    """
    invcell=np.linalg.inv(cell) # The inverse cell parameters
    supercells = np.array(list(itertools.product((-1, 0, 1), repeat=3))) # 27 possible crystal cell shifts.
    fcoords = np.dot(coords2,invcell) + supercells # Many different versions of coords2, shifted different linear combinations of the crystal cell vectors.
    coords = np.array([np.dot(j,cell) for j in fcoords]) # Converting to Cartesian coordinates.
    dists = distance.cdist([coords1], coords) # Euclidean distance
    return np.amin(dists) # Take the minimum, corresponding to the distance between the two atoms at their closest, when considering the periodic structure of a MOF.
