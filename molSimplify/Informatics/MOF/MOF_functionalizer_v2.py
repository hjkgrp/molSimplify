from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.atom3D import atom3D
from molSimplify.Scripts.cellbuilder_tools import import_from_cif
from molSimplify.Informatics.MOF.MOF_descriptors import get_primitive
from molSimplify.Informatics.MOF.monofunctionalized_BDC.index_information import INDEX_INFO
from molSimplify.Scripts.geometry import checkplanar, PointRotateAxis, distance, rotate_around_axis
from molSimplify.Informatics.MOF.PBC_functions import (
        compute_adj_matrix,
        compute_distance_matrix,
        compute_image_flag,
        findPaths,
        frac_coord,
        fractional2cart,
        get_closed_subgraph,
        readcif,
        XYZ_connected,
        write_cif,
        overlap_removal,
        )
from pkg_resources import resource_filename, Requirement
import numpy as np
import scipy
import networkx as nx
import os
import shutil
import pickle
### Beginning of functions ###

##### THE INPUT REQUIRES A P1 CELL ######
# If not P1, will functionalize wrong.  #
#########################################

# This script can only functionalize C-H bonds.

def functionalize_MOF(cif_file, path2write, functional_group = 'F', functionalization_limit = 1, path_between_functionalizations = 3, additional_atom_offset = 0):
    """
    Functionalizes the provided MOF and writes the functionalized version to a cif file.
    Loops through the atoms of a MOF and functionalizes at suitable carbon atoms.

    Parameters
    ----------
    cif_file : str
        The path to the cif file to be functionalized.
    path2write : str
        The folder path where the cif of the functionalized MOF will be written.
    functional_group : str
        The functional group to use for MOF functionalization.
    functionalization_limit : int
        The number of functionalizations per linker.
    path_between_functionalizations : int
        How many bonds away one functionalized atom should be from another, if functionalized_limit is greater than one.
    additional_atom_offset : float
        Extent to which to rotate the placement of depth 2 functional group atoms. Give in degrees.
        Useful for preventing atomic overlap / unintended bonds.

    Returns
    -------
    functionalized_atoms : list of int
        Which indices of the original cif file were functionalized.

    """

    dict_approach = check_support(functional_group)
    if not dict_approach:
        raise Exception('That functional group is not supported through the dictionary approach.')

    base_mof_name = os.path.basename(cif_file)
    if base_mof_name.endswith('.cif'):
        base_mof_name = base_mof_name[:-4]
    ######################################################
    # Takes the CIF file as input of the bare structure. #
    # Functionalization limit is how many times a single #
    # linker is allowed to be functionalized. Default    #
    # functionalization is fluoride.                     #
    ######################################################

    # Read the cif file and make the cell for fractional coordinates
    cpar, allatomtypes, fcoords = readcif(cif_file)
    molcif, cell_vector, alpha, beta, gamma = import_from_cif(cif_file, True)
    cell_v = np.array(cell_vector)
    cart_coords = fractional2cart(fcoords, cell_v)
    distance_mat = compute_distance_matrix(cell_v, cart_coords)
    adj_matrix, _ = compute_adj_matrix(distance_mat, allatomtypes)
    molcif.graph = adj_matrix.todense()

   # Change
    ###### At this point, we have most things we need to functionalize.
    # Thus the first step is to break down into linkers. This uses what we developed for MOF featurization
    linker_list, linker_subgraphlist = get_linkers(molcif, adj_matrix, allatomtypes)


    ###### We need to then figure out which atoms to functionalize.
    checkedlist = set() # Keeps track of the atoms that have already been checked for functionalization.
    # Make a copy of the atom type list to loop over later.
    original_allatomtypes = allatomtypes.copy() # Storing all the chemical symbols that there were originally.
    delete_list = [] # Collect all of the H that need to be deleted later.
    extra_atom_coords = []
    extra_atom_types = []
    functionalized_atoms = []

    if functional_group != 'H': # We don't do anything for -H functionalization.

        for linker_to_analyze_index, linker_to_analyze in enumerate(linker_list):
            # delete_list = [] # Collect all of the H that need to be deleted later.
            # extra_atom_coords = []
            # extra_atom_types = []

            linker_atom_types, linker_graph, linker_cart_coords = analyze_linker(cart_coords, linker_to_analyze,
                                                                                 original_allatomtypes, linker_subgraphlist, linker_to_analyze_index, cell_v)
            functionalized_wo_bond = False

            for i in linker_to_analyze:
                atom = original_allatomtypes[i]
                print(f'i is {i}')
                if i in checkedlist:
                    continue # Move on to the next atom.
                if atom != 'C': # Assumes that functionalization is performed on a C atom.
                    checkedlist.add(i)
                    continue

            # Atoms that are connected to atom i.
                connected_atom_list, connected_atom_types = connected_atoms_from_adjmat(adj_matrix, i, original_allatomtypes)

                if ('H' not in connected_atom_types) or (connected_atom_types.count('H')>1 or len(connected_atom_types) != 3): ### must functionalize where an H was. Needs sp2 C.
                    # Note: if a carbon has more than one hydrogen bonded to it, it is not considered for functionalization.
                    # So, the carbons treated by this code will carbons in a benzene-style ring for the most part, I assume.
                        # Since apply_functionalization assumes two neighbors to the carbon excluding hydrogens.
                        # TODO expand in the future?
                    # Note: can only replace a hydrogen in the functionalization, at the moment. Can't replace a methyl, hydroxyl, etc.
                    checkedlist.add(i)
                    continue

                else: # Found a suitable location for functionalization.
                    functionalized_wo_bond = False
                    functionalization_counter = functionalization_limit
                    unintend_bond = True

                    """""""""
                    Linker functionalization of the current linker.
                    """""""""
                    # The following code will functionalize this linker functionalization_limit times, or as close to this many times as possible.
                    for k, connected_atom in enumerate(connected_atom_types): # Look through the atoms bonded to atom i.
                        if connected_atom == 'H':

                            """""""""
                            The first linker functionalization.
                            """""""""
                            molcif, functionalization_counter, functionalized, delete_list, extra_atom_coords, extra_atom_types, functionalized_atoms, unintend_bond = first_functionalization(cell_v, molcif,
                                allatomtypes,
                                i,
                                connected_atom_list,
                                k,
                                functional_group,
                                linker_cart_coords,
                                linker_to_analyze,
                                linker_atom_types,
                                linker_graph,
                                functionalization_counter,
                                delete_list,
                                extra_atom_coords,
                                extra_atom_types,
                                functionalized_atoms,
                                additional_atom_offset=additional_atom_offset
                                )
                            if not unintend_bond:
                                functionalized_wo_bond = True
                                break

                if functionalized_wo_bond:
                    break
                # if not unintend_bond:
                #     functionalized_wo_bond = True
                    ## Updating the atoms list
                    # new_coord_list, final_atom_types = atom_deletion(cart_coords, allatomtypes, delete_list)
                    # # Adding atoms (the atoms in the functional groups)
                    # allatomtypes, fcoords = atom_addition([extra_atom_types], final_atom_types, new_coord_list, extra_atom_coords, cell_v)

                   #break # Don't search the rest of the connected atoms if replaced a hydrogen and functionalized already at the atom with index i.
                    """""""""
            Any additional linker functionalizations.
            """""""""
            # If there is more than one functionalization, this is where that happens.
            # Will check other atoms on the linker to potentially functionalize them.

            if functionalized_wo_bond:
                while functionalization_counter > 0:
                    # delete_list = [] # Collect all of the H that need to be deleted later.
                    # extra_atom_coords = []
                    # extra_atom_types = []

                    molcif, functionalization_counter, delete_list, extra_atom_coords, extra_atom_types, functionalized_atoms, unintend_bond = additional_functionalization(cell_v, i,
                        linker_to_analyze,
                        linker_subgraphlist,
                        linker_to_analyze_index,
                        path_between_functionalizations,
                        functionalized,
                        adj_matrix,
                        allatomtypes,
                        molcif,
                        functional_group,
                        linker_cart_coords,
                        linker_atom_types,
                        linker_graph,
                        functionalization_counter,
                        delete_list,
                        extra_atom_coords,
                        extra_atom_types,
                        functionalized_atoms,
                        additional_atom_offset=additional_atom_offset
                        )

                    if unintend_bond:
                        break
                    # if not unintend_bond:
                    #     functionalized_wo_bond = True
                        ## Updating the atoms list
                        # new_coord_list, final_atom_types = atom_deletion(cart_coords, allatomtypes, delete_list)
                        # # Adding atoms (the atoms in the functional groups)
                        # allatomtypes, fcoords = atom_addition([extra_atom_types], final_atom_types, new_coord_list, extra_atom_coords, cell_v)


    """""""""
    Apply delete_list and extra_atom_types to make final_atom_types and new_coord_list.
    """""""""
    print(delete_list)
    print(extra_atom_coords)
    # Deleting atoms (hydrogens that are replaced by functional groups)
    new_coord_list, final_atom_types = atom_deletion(cart_coords, allatomtypes, delete_list)

    # Adding atoms (the atoms in the functional groups)
    allatomtypes, fcoords = atom_addition(extra_atom_types, final_atom_types, new_coord_list, extra_atom_coords, cell_v)
    bond_overlap = post_functionalization_overlap_and_bonding_check(cell_v, allatomtypes, fcoords, extra_atom_types)

    if not bond_overlap:
        ## post functionalization
        """""""""
        Write the cif.
        """""""""
        cif_folder = f'{path2write}/cif'
        mkdir_if_absent(cif_folder)
        write_cif(f'{cif_folder}/functionalized_{base_mof_name}_{functional_group}_{functionalization_limit}.cif', cpar, fcoords, allatomtypes)

        """""""""
        Check on how the functionalization affected the symmetry.
        """""""""
        print('------- UNFUNCTIONALIZED CASE --------')
        #symmetry_check(original_allatomtypes, original_fcoords, cell_v)

        # Analysis for the case where the cell is functionalized.
        # Difference with the block above: allatomtypes and fcoords, instead of original_allatomtypes and original_fcoords
        print('------- FUNCTIONALIZED CASE --------')
            #symmetry_check(allatomtypes, fcoords, cell_v)
        ## Post functionalization bond order check
    else:
    # # bond_overlap = post_functionalization_overlap_and_bonding_check(cell_v, allatomtypes, fcoords, extra_atom_types)
    # # if bond_overlap:
        mof_name = os.path.basename(cif_file).strip('.cif')
        with open(f'atomic_overlap_{functionalization_limit}_func.txt', 'a') as file:
            line = f'{mof_name} {functional_group}\n'
            file.write(line)
            file.close()


def first_functionalization(cell_v, molcif,
    allatomtypes,
    i,
    connected_atom_list,
    k,
    functional_group,
    linker_cart_coords,
    linker_to_functionalize,
    linker_atom_types,
    linker_graph,
    functionalization_counter,
    delete_list,
    extra_atom_coords,
    extra_atom_types,
    functionalized_atoms,
    additional_atom_offset=0
    ):
    """
    Functionalizes a linker for the first time, at atom `i` with functional group `functional_group`.

    Parameters
    ----------
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file to be functionalized.
    allatomtypes : list of str
        The atom types of the MOF, indicated by chemical symbols like 'O' and 'Cu'. Length is the number of atoms.
    i : int
        The global index of the atom of to be functionalized.
    connected_atom_list : numpy.ndarray of numpy.int32
        The indices of the atoms connected to the atom of interest i.
    k : int
        The index of the atom in connected_atom_types that is a hydrogen. Will be replaced with the functional group.
    functional_group : str
        The functional group to use for MOF functionalization.
    linker_cart_coords : numpy.ndarray of numpy.float64
        The Cartesian coordinates of the atoms in the linker. Shape is (number of atoms in linker, 3).
    linker_to_functionalize : list of numpy.int32
        A list of the global atom indices of the atoms in the identified linker.
        The identified linker is the one that has atom i.
    linker_atom_types : list of str
        The chemical symbols of the atoms in the linker. Length is the number of atoms in the linker.
    linker_graph : numpy.ndarray of numpy.float64
        The adjacency matrix of the linker. Shape is (number of atoms in linker, number of atoms in linker).
    functionalization_counter : int
        The number of functionalizations left to be done on the linker.
    delete_list : list of numpy.int32
        The indices of atoms that are deleted because they are replaced by functional groups.
    extra_atom_coords : list of numpy.ndarray of numpy.float64
        The Cartesian coordinates of the atoms added during functionalization.
        Each item of the list is a new functional group.
        Shape of each numpy.ndarray is (number of atoms in functional group, 3).
    extra_atom_types : list of list of str
        The chemical symbols of the atoms added through functional groups. Each inner list is a functional group.
    functionalized_atoms : list of int
        The global indices of atoms that have been functionalized.
    additional_atom_offset : float
        Extent to which to rotate the placement of depth 2 functional group atoms. Give in degrees.
        Useful for preventing atomic overlap / unintended bonds.

    Returns
    -------
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the functionalized MOF.
    functionalization_counter : int
        The number of functionalizations left to be done on the linker. This variable is decreased by one when this function is run successfully.
    functionalized : bool
        Indicates whether the number of linker functionalizations requested by the user have been made. True is so, False otherwise.
    delete_list : list of numpy.int32
        The updated indices of atoms that are deleted because they are replaced by functional groups.
    extra_atom_coords : list of numpy.ndarray of numpy.float64
        The updated Cartesian coordinates of the atoms added during functionalization.
    extra_atom_types : list of list of str
        The updated chemical symbols of the atoms added through functional groups. Each inner list is a functional group.
    functionalized_atoms : list of int
        The updated global indices of atoms that have been functionalized.


    """
    # Apply the functionalization to the MOF.
    molcif, atom_types_to_add, additions_to_cart, functionalization_counter, functionalized, unintend_bond = apply_functionalization(cell_v, molcif,
                                        allatomtypes, i, connected_atom_list[k], connected_atom_list, functional_group,
                                        linker_cart_coords, linker_to_functionalize, linker_atom_types, linker_graph, functionalization_counter, additional_atom_offset=additional_atom_offset)

    # Add the atom that's been functionalized to the deleted atom list so that it isn't kept in the final structure.
    delete_list.append(connected_atom_list[k]) # E.g. if a hydrogen was replaced by a fluorine, delete the hydrogen.

    # There are atoms added to the structure now. Add those.
    extra_atom_coords.append(additions_to_cart)
    extra_atom_types.append(atom_types_to_add)

    # Keep track of what atoms have been functionalized.
    functionalized_atoms.append(i)

    return molcif, functionalization_counter, functionalized, delete_list, extra_atom_coords, extra_atom_types, functionalized_atoms, unintend_bond

def additional_functionalization(cell_v, i,
    linker_to_functionalize,
    linker_subgraphlist,
    linker_to_functionalize_index,
    path_between_functionalizations,
    functionalized,
    adj_matrix,
    allatomtypes,
    molcif,
    functional_group,
    linker_cart_coords,
    linker_atom_types,
    linker_graph,
    functionalization_counter,
    delete_list,
    extra_atom_coords,
    extra_atom_types,
    functionalized_atoms,
    additional_atom_offset=0):
    """
    Executes additional functionalization on the specified linker,
    at positions (path_between_functionalizations) bonds away from the atom with index i.

    Parameters
    ----------
    i : int
        The global index of the atom from which atoms that are path_between_functionalizations bonds away will be considered.
    linker_to_functionalize : list of numpy.int32
        A list of the global atom indices of the atoms in the identified linker.
        The identified linker is the one that has atom i.
    linker_subgraphlist : list of scipy.sparse.csr.csr_matrix
        The atom connections in the linker subgraph. Length is # of linkers.
    linker_to_functionalize_index : int
        The number identifier of the linker that contains the atom of interest.
    path_between_functionalizations : int
        How many bonds away one functionalized atom should be from another, if functionalized_limit is greater than one.
    functionalized : bool
        Indicates whether the number of linker functionalizations requested by the user have been made. True is so, False otherwise.
    adj_matrix : scipy.sparse.csr.csr_matrix
        1 represents a bond, 0 represents no bond. Shape is (number of atoms, number of atoms).
    allatomtypes : list of str
        The atom types of the MOF, indicated by chemical symbols like 'O' and 'Cu'. Length is the number of atoms.
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file to be functionalized.
    functional_group : str
        The functional group to use for MOF functionalization.
    linker_cart_coords : numpy.ndarray of numpy.float64
        The Cartesian coordinates of the atoms in the linker. Shape is (number of atoms in linker, 3).
    linker_atom_types : list of str
        The chemical symbols of the atoms in the linker. Length is the number of atoms in the linker.
    linker_graph : numpy.ndarray of numpy.float64
        The adjacency matrix of the linker. Shape is (number of atoms in linker, number of atoms in linker).
    functionalization_counter : int
        The number of functionalizations left to be done on the linker.
    delete_list : list of numpy.int32
        The indices of atoms that are deleted because they are replaced by functional groups.
    extra_atom_coords : list of numpy.ndarray of numpy.float64
        The Cartesian coordinates of the atoms added during functionalization.
        Each item of the list is a new functional group.
        Shape of each numpy.ndarray is (number of atoms in functional group, 3).
    extra_atom_types : list of list of str
        The chemical symbols of the atoms added through functional groups. Each inner list is a functional group.
    functionalized_atoms : list of int
        The global indices of atoms that have been functionalized.
    additional_atom_offset : float
        Extent to which to rotate the placement of depth 2 functional group atoms. Give in degrees.
        Useful for preventing atomic overlap / unintended bonds.

    Returns
    -------
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the functionalized MOF.
    functionalization_counter : int
        The number of functionalizations left to be done on the linker. This variable is decreased by one when this function is run successfully.
    delete_list : list of numpy.int32
        The updated indices of atoms that are deleted because they are replaced by functional groups.
    extra_atom_coords : list of numpy.ndarray of numpy.float64
        The updated Cartesian coordinates of the atoms added during functionalization.
    extra_atom_types : list of list of str
        The chemical symbols of the atoms added through functional groups. Each inner list is a functional group.
    functionalized_atoms : list of int
        The updated global indices of atoms that have been functionalized.

    """
    original_functionalization_counter = functionalization_counter

    anchor_idx = linker_to_functionalize.index(i) # As a reminder, linker_to_functionalize is a list of numpy.int32, the numpy.int32s being indices for the atoms in the linker
    anchor_coordinate = linker_cart_coords[anchor_idx]
    G = make_networkx_graph(linker_subgraphlist[linker_to_functionalize_index]) # Getting the graph for the linker of interest.
    # Use network X to find functionalization paths that are N atoms away from the original spot
    n_path_lengths_away = findPaths(G,anchor_idx,path_between_functionalizations)
    already_functionalized = False

    path_distance_list = []
    for path in n_path_lengths_away:
        path_coord = linker_cart_coords[path[-1]]
        path_distance_list.append(np.linalg.norm(path_coord - anchor_coordinate))

    np_distances = np.array(path_distance_list)
    sorted_indices_desc = np.argsort(-np_distances)

    n_path_lengths_away_sorted = [n_path_lengths_away[idx] for idx in sorted_indices_desc]

    unintend_bond = False

    for path in n_path_lengths_away_sorted: # Looking at the possible paths between the anchor_idx and atoms that are N (path_between_functionalizations) atom away.
        if already_functionalized: # An atom was already functionalized.
            break

        potential_functionalization = path[-1] # Gets the last point on the graph at distance "path_between_functionalizations" away.
        functionalization_index = linker_to_functionalize[potential_functionalization] # Gets the global index of the atom to functionalize.

        if allatomtypes[functionalization_index] == 'C':
            # Get the neighbors of the atom that we are considering for functionalization.
            secondary_connected_atom_list, secondary_connected_atom_types = connected_atoms_from_adjmat(adj_matrix, functionalization_index, allatomtypes)

            if 'H' not in secondary_connected_atom_types or len(secondary_connected_atom_types) != 3:
                continue # Must functionalize where an H was. If not, skip.
            elif functionalization_index in functionalized_atoms:
                continue # This atom has already been functionalized.
            else:
                for l, secondary_connected_atom in enumerate(secondary_connected_atom_types):
                    if (secondary_connected_atom == 'H') and (not functionalized):
                        molcif, atom_types_to_add, additions_to_cart, functionalization_counter, functionalized, unintend_bond = apply_functionalization(cell_v, molcif,
                                        allatomtypes, functionalization_index, secondary_connected_atom_list[l], secondary_connected_atom_list,
                                        functional_group, linker_cart_coords, linker_to_functionalize, linker_atom_types, linker_graph,
                                        functionalization_counter, additional_atom_offset=additional_atom_offset)
                        if not unintend_bond:
                            delete_list.append(secondary_connected_atom_list[l])
                            extra_atom_coords.append(additions_to_cart)
                            extra_atom_types.append(atom_types_to_add)
                            functionalized_atoms.append(functionalization_index)
                            already_functionalized = True # Want to break out of all the for loops
                            break # break the for l, secondary... loop since a functionalization was made.

    if functionalization_counter == original_functionalization_counter: # Equivalently, if already_functionalized == False
        # This means there are no more locations on the linker that can be functionalized.
        functionalization_counter = 0 # No more functionalizations to be done.

    return molcif, functionalization_counter, delete_list, extra_atom_coords, extra_atom_types, functionalized_atoms, unintend_bond

def apply_functionalization(cell_v, molcif, allatomtypes, position_to_functionalize, atom_to_replace, position_to_functionalize_neighbors,
                                        functional_group, linker_cart_coords, linker_to_analyze, linker_atom_types, linker_graph, functionalization_counter, additional_atom_offset=0):
    #######################################################################################################
    # Note: position_to_functionalize is distinct from atom_to_replace. When functionalizing a C-H bond,  #
    # position_to_functionalize is the C, and atom_to_replace is the H. Currently, only select            #
    # functionalizations can be handled: NH2, CH3, NO2, CF3, CN, OH, SH.                                  #
    # mol3D object deletes the H atom and adds all other groups. Position_to_functionalize_neighbors      #
    # allows determination of the plane of functionalization (important for symmetry preservation).       #
    #######################################################################################################
    """
    Functionalizes at the specified position. Supports some multi-atom functional groups.
    Functionalization will take place at the index position_to_functionalize.
    The atom with that index is in the linker described by linker_cart_coords, linker_to_analyze, linker_atom_types, and linker_graph.

    Parameters
    ----------
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file to be functionalized.
    allatomtypes : list of str
        The atom types of the cif file, indicated by chemical symbols like 'O' and 'Cu'. Length is the number of atoms.
    position_to_functionalize : int
        The global index of the atom to functionalize.
        For the current state of the code, this is a carbon.
    atom_to_replace : numpy.int32
        The index of the atom to replace with the specified functional group.
        For the current state of the code, this is a hydrogen.
    position_to_functionalize_neighbors : numpy.ndarray of numpy.int32
        The indices of atoms bonded to the atom with index position_to_functionalize.
    functional_group : str
        The functional group to use for MOF functionalization.
    linker_cart_coords : numpy.ndarray of numpy.float64
        The Cartesian coordinates of the atoms in the linker. Shape is (number of atoms in linker, 3).
    linker_to_analyze : list of numpy.int32
        The indices of the atoms in the linker.
    linker_atom_types : list of str
        The chemical symbols of the atoms in the linker.
    linker_graph : numpy.ndarray of numpy.float64
        The adjacency matrix of the linker. 1 indicates a bond. 0 indicates the absence of a bond.
    functionalization_counter : int
        The number of functionalizations left to be done on the linker.
    additional_atom_offset : float
        Extent to which to rotate the placement of depth 2 functional group atoms. Give in degrees.
        Useful for preventing atomic overlap / unintended bonds.

    Returns
    -------
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the functionalized MOF.
    atom_types_to_add : list of str
        The chemical symbols of the atoms added. These are the atoms in the functional group.
    additions_to_cart : numpy.ndarray of numpy.float64
        The Cartesian coordinates of the atoms added during functionalization. Shape is (number of atoms in functional group, 3).
    functionalization_counter : int
        The number of functionalizations left to be done on the linker. This variable is decreased by one when this function is run successfully.
    functionalized : bool
        Indicates whether the number of linker functionalizations requested by the user have been made. True is so, False otherwise.

    """
    # connection_length_dict, connection_atom_dict, bond_length_dict, bond_angle_dict, bond_rotation_dict = geo_dict_loader()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(script_dir, 'functional_group_dictionary.pkl')
    with open(pkl_path, 'rb') as pickle_file:
        geo_dict_loader_gram_schmidt = pickle.load(pickle_file)

    ##### Construct the plane to work on.
    neighbors_not_to_replace = list(set(position_to_functionalize_neighbors)-set([atom_to_replace])) # The not-to-be-replaced atoms bonded to the atom to functionalized.
    # print everything
    if len(neighbors_not_to_replace) > 2: # The atom to functionalize has too many atoms connected to it. More than 3.
        # with open('MOFs_with_sp2_issue_1inor_1edge_1func.txt', 'a') as file:
            #file.write(current_mof+'\n')
            #file.close()
        raise ValueError('currently can only work with sp2 carbons. This is > sp2.')

    functionalization_position_on_linker = linker_to_analyze.index(position_to_functionalize)  # Local index of the C atom in the linker where functionalization will happen.

    functionalization_hydrogen_on_linker = linker_to_analyze.index(atom_to_replace)   # Local index of the H atom in the linker where functionalization will happen.

    neighbor_index_on_linker = linker_to_analyze.index(neighbors_not_to_replace[0])
    neighbour_counter = 0

    for neighbor_index in neighbors_not_to_replace:
        if allatomtypes[neighbor_index] == 'C':
            neighbor_index_on_linker = linker_to_analyze.index(neighbor_index)
            break
        else:
            neighbour_counter += 1

    unintend_bond = True   ## Flag that checks when the unintended bond is there
    angle = 0 ## Angle for rotation

        ## Implementing roation of functional group atoms to avoid overlap
    while unintend_bond:

        molcif, atom_types_to_add, additions_to_cart, can_rotate = functionalization_gram_schmidt(molcif, geo_dict_loader_gram_schmidt,
                                                                        functional_group, linker_cart_coords,
                                                                        functionalization_position_on_linker,
                                                                        functionalization_hydrogen_on_linker,
                                                                        neighbor_index_on_linker, rotation_angle=angle, update_molcif=False)

        fcoords_updated = frac_coord(np.concatenate((molcif.coordsvect(), additions_to_cart)), cell_v)
        atom3d_atomtypes_new = molcif.getAtoms()
        allatomtypes_new = [atom.symbol() for atom in atom3d_atomtypes_new]
        allatomtypes_new.extend(atom_types_to_add)

        new_coord_list, final_atom_types = atom_deletion(np.array(fcoords_updated), allatomtypes_new, [atom_to_replace])

        bond_overlap = post_functionalization_overlap_and_bonding_check(cell_v, final_atom_types, new_coord_list, [atom_types_to_add])
        #print(bond_overlap)

        if not bond_overlap or angle >= 360 or not can_rotate:
            molcif, atom_types_to_add, additions_to_cart, can_rotate = functionalization_gram_schmidt(molcif, geo_dict_loader_gram_schmidt,
                                                                        functional_group, linker_cart_coords,
                                                                        functionalization_position_on_linker,
                                                                        functionalization_hydrogen_on_linker,
                                                                        neighbor_index_on_linker, rotation_angle=angle, update_molcif=True)
            unintend_bond = False

        else:
            angle += 30

    functionalization_counter -= 1
    if functionalization_counter == 0:
        # If the number of linker functionalizations requested by the user have been made, set this variable to true.
        functionalized = True
    else:
        functionalized = False

    return molcif, atom_types_to_add, additions_to_cart, functionalization_counter, functionalized, unintend_bond

def functionalization_gram_schmidt(molcif, geo_dict_loader_gram_schmidt, functional_group,
                                   linker_cart_coords, functionalization_position_on_linker, functionalization_hydrogen_on_linker,
                                   neighbor_index_on_linker, rotation_angle, update_molcif=True):

    can_rotate = True
    no_func_group_atoms = len(geo_dict_loader_gram_schmidt[functional_group]['atoms'][0])
    if no_func_group_atoms == 1:
        can_rotate = False
     # basis vectors for the functional group
    v1 = np.array(linker_cart_coords[functionalization_hydrogen_on_linker] - linker_cart_coords[functionalization_position_on_linker])
    v2 = np.array(linker_cart_coords[neighbor_index_on_linker] - linker_cart_coords[functionalization_position_on_linker])
    u1 = v1
    e1 = u1/np.linalg.norm(u1)
    u2 = v2 - (np.dot(v2, u1)/np.dot(u1, u1))*u1
    e2 = u2/np.linalg.norm(u2)
    e3 = np.cross(e1, e2)

    destination_basis = np.zeros(shape=(3, 3))
    destination_basis[:, 0] = e1
    destination_basis[:, 1] = e2
    destination_basis[:, 2] = e3

    ## Original basis of the functional group
    benzene_coords = geo_dict_loader_gram_schmidt[functional_group]['benzene_carbon_coord'][0]
    functional_group_coords_original = geo_dict_loader_gram_schmidt[functional_group]['coords'][0].reshape(no_func_group_atoms, 3)
    v1s = functional_group_coords_original[0, :] - benzene_coords[0, :]
    v2s = benzene_coords[1, :] - benzene_coords[0, :]
    u1s = v1s
    e1s = u1s/np.linalg.norm(u1s)
    u2s = v2s - (np.dot(v2s, u1s)/np.dot(u1s, u1s))*u1s
    e2s = u2s/np.linalg.norm(u2s)
    e3s = np.cross(e1s, e2s)

    source_basis = np.zeros(shape=(3, 3))
    source_basis[:, 0] = e1s
    source_basis[:, 1] = e2s
    source_basis[:, 2] = e3s

    rotation_matrix = destination_basis @ source_basis.T
    functional_group_coords_original_relative = functional_group_coords_original - benzene_coords[0, :]

    new_coords = np.zeros(shape=(no_func_group_atoms, 3))

    for i in range(no_func_group_atoms):
        new_coords[i, :] = (rotation_matrix @ functional_group_coords_original_relative[i, :].T).T

    ## Rotaion of functional group
    rotated_coords = functional_group_rotation(new_coords, e1, rotation_angle)
    new_coords = np.array(linker_cart_coords[functionalization_position_on_linker]) + rotated_coords

    additions_to_cart = np.zeros([no_func_group_atoms, 3])

    atom_types_to_add = []

    for i in range(no_func_group_atoms):
        connecting_atom = atom3D(geo_dict_loader_gram_schmidt[functional_group]['atoms'][0][i], new_coords[i, :])
        if update_molcif:
            molcif.addAtom(connecting_atom)
        additions_to_cart[i,:] = np.array(new_coords[i, :])

        atom_types_to_add.append(geo_dict_loader_gram_schmidt[functional_group]['atoms'][0][i])

    return molcif, atom_types_to_add, new_coords, can_rotate

def functional_group_rotation(functional_group_coords, axis, rotation_angle):

    skew_symmetric_matrix = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    rotation_matrix = np.identity(3) + np.sin(rotation_angle) * skew_symmetric_matrix + (1 - np.cos(rotation_angle)) * np.dot(skew_symmetric_matrix, skew_symmetric_matrix)
    updated_functional_group_coords = np.dot(functional_group_coords, rotation_matrix)

    return updated_functional_group_coords

def analyze_linker(cart_coords,
    linker_to_analyze,
    allatomtypes,
    linker_subgraphlist,
    linker_to_analyze_index,
    cell_v):
    """
    Returns information on the specified linker.

    Parameters
    ----------
    cart_coords : numpy.ndarray of numpy.float64
        The Cartesian coordinates of the crystal atoms. Shape is (number of atoms, 3).
    linker_to_analyze : list of numpy.int32
        A list of the global atom indices of the atoms in the identified linker.
        The identified linker is the one that has atom i.
    allatomtypes : list of str
        The atom types of the MOF, indicated by chemical symbols like 'O' and 'Cu'. Length is the number of atoms.
    linker_subgraphlist : list of scipy.sparse.csr.csr_matrix
        The atom connections in the linker subgraph. Length is # of linkers.
    linker_to_analyze_index : int
        The number identifier of the linker that contains the atom of interest.
    cell_v : numpy.ndarray of numpy.float64
        Each row corresponds to one of the cell vectors. Shape is (3, 3).

    Returns
    -------
    linker_atom_types : list of str
        The chemical symbols of the atoms in the linker. Length is the number of atoms in the linker.
    linker_graph : numpy.ndarray of numpy.float64
        The adjacency matrix of the linker. Shape is (number of atoms in linker, number of atoms in linker).
    linker_cart_coords : numpy.ndarray of numpy.float64
        The Cartesian coordinates of the atoms in the linker. Shape is (number of atoms in linker, 3).

    """
    # Get the cartesian coordinates of the linker from the linker atoms.
    linker_coords = [cart_coords[val,:] for val in linker_to_analyze] ### contains the atom numbers in the linker.
    # Get the linker atom types of the linker from the linker atoms.
    linker_atom_types = [allatomtypes[val] for val in linker_to_analyze]
    # Get the linker graph that's useful for determining what's connected to what.
    linker_graph = linker_subgraphlist[linker_to_analyze_index].todense()
    linker_graph = np.asarray(linker_graph)
    # Get the connected atoms that will shift positions in fractional coordinates.
    linker_f_coords = XYZ_connected(cell_v, linker_coords, linker_graph)
    # Use the cell vector to translate those coordinates back to Cartesian.
    linker_cart_coords = fractional2cart(linker_f_coords,cell_v)

    return linker_atom_types, linker_graph, linker_cart_coords

def symmetry_check(allatomtypes, fcoords, cell_v, precision=1):
    """
    Checks the spacegroup and the space group number of the provided MOF information.
    Before and after finding the Niggli cell (maximally reduced cell).

    Parameters
    ----------
    allatomtypes : list of str
        The atom types of the MOF, indicated by chemical symbols like 'O' and 'Cu'. Length is the number of atoms.
    fcoords : numpy.ndarray of numpy.float64
        The fractional positions of the crystal atoms. Shape is (number of atoms, 3).
    cell_v : numpy.ndarray of numpy.float64
        Each row corresponds to one of the cell vectors. Shape is (3, 3).
    precision : float
        Cartesian distance tolerance and angle tolerance.
        https://spglib.github.io/spglib/variable.html#variables-symprec

    Returns
    -------
    None

    """
    import spglib
    numbers = [] # Will keep track of which atom in unique_types is in which position in allatomtypes.
    unique_types=list(set(allatomtypes))
    for label in allatomtypes:
        numbers.append(int(unique_types.index(label)+1))

    full_cell_for_spg = (cell_v, fcoords, numbers)
    spcg = spglib.get_spacegroup(full_cell_for_spg, symprec=precision)
    dataset = spglib.get_symmetry_dataset(full_cell_for_spg)
    space_group_number = int(dataset['number'])
    print('spacegroup before:', spcg)
    print('space group number before:', space_group_number)

    try:
        lattice_new, scaled_positions_new, numbers_new = spglib.standardize_cell(full_cell_for_spg, to_primitive=False,
            no_idealize=False, symprec=precision, angle_tolerance=precision)
        niggli_lattice = spglib.niggli_reduce(lattice_new, eps=1e-5) # Niggli reduction
        spcg = spglib.get_spacegroup((niggli_lattice, scaled_positions_new, numbers_new), symprec=precision)
        dataset = spglib.get_symmetry_dataset((niggli_lattice, scaled_positions_new, numbers_new))
        space_group_number = int(dataset['number'])
        print('spacegroup after standardization:', spcg)
        print('space group number after standardization:', space_group_number)

    except TypeError:
        print("Error: spglib.standardize_cell returned None.")

def linker_identification(linker_list, i, checkedlist):
    """
    Identifies which linker the atom i is in.

    Parameters
    ----------
    linker_list : list of lists of ints
        Each inner list is its own separate linker. The ints are the global atom indices of that linker. Length is # of linkers.
    i : int
        The global index of the atom of interest.
    checkedlist : set of int
        The indices of atoms that have already been checked for functionalization.

    Returns
    -------
    linker_to_analyze : list of numpy.int32
        A list of the global atom indices of the atoms in the identified linker.
        The identified linker is the one that has atom i.
    linker_to_analyze_index : int
        The number identifier of the linker that contains the atom of interest.
    checkedlist : set of int
        The indices of atoms that have already been checked for functionalization.
        Updated to include the atoms in the identified linker.

    """
    linker_to_analyze, linker_to_analyze_index = None, None # These are updated in the following line.
    for linker_num, linker in enumerate(linker_list): # Iterate over the linkers of the MOF.
        if i in linker: # The atom i (which is to be functionalized) is in the atoms of the current linker.
            linker_to_analyze_index, linker_to_analyze = linker_num, linker
            # Once a linker has been functionalized, we want to be done with that linker and not functionalize it again.
            [checkedlist.add(val) for val in linker]
            break # Don't need to keep looking through the linkers in this case, since the atom i which is to be functionalized was in the current linker.

    if linker_to_analyze is None: # linker_to_analyze was never overwritten.
        raise Exception(f"Atom {i} was not in any linker - something has gone wrong.")

    return linker_to_analyze, linker_to_analyze_index, checkedlist

def geo_dict_loader():
    """
    Returns geometry information on the supported functional groups.
    Currently, these are F, Cl, Br, I, CH3, CN, NH2, NO2, CF3, OH, SH, COOH.

    Parameters
    ----------
    None

    Returns
    -------
    connection_length_dict : dict
        For each functional group, indicates to bond length of the connecting atom on the functional group to a carbon atom, in angstroms.
    connection_atom_dict : dict
        For each functional group, indicates the element in the functional group that connects to the carbon atom; then indicates remaining elements for multiatomic functional groups.
    bond_length_dict : dict
        For each functional group (as applicable), indicates the bond length of the connecting atom on the functional group to the other atoms in the functional group, in angstroms.
        Pertinent to multiatomic functional groups.
    bond_angle_dict : dict
        For each functional group (as applicable), indicates how far off a straight line the non-connecting functional group atoms are,
        relative to a directional unit vector that goes through the connecting carbon and the connecting atom on the functional group.
        For example, for CF3, this is the C-C-F angle, where the first C is the carbon being functionalized and the second C is part of the CF3 functional group.
        Pertinent to multiatomic functional groups.
    bond_rotation_dict : dict
        For each functional group (as applicable), indicates the angle rotation and the number of rotations for non-connecting atoms. E.g. for CH3, the hydrogens are 120 degrees rotated apart.
        Pertinent to multiatomic functional groups.

    """
    ### connection_length_dict: How far to place the functional group connecting atom from the connecting carbon.
    # All lengths and angles are coming from DFT calculations on the solo linker (Br, CF4).
    connection_length_dict = {'F':1.37,'Cl':1.80,'Br':2.03,'I':2.23,'CH3':1.52,'CN':1.44,'NH2':1.38,'NO2':1.49,'CF3':1.51,'OH':1.36,'SH':1.82, 'COOH':1.49}

    # The first element of every item is the connecting atom. For example, CH3 will connect through the carbon.
    connection_atom_dict = {'F':['F'],'Cl':['Cl'],'Br':['Br'],'I':['I'],'CH3':['C','H'],'CN':['C','N'],'NH2':['N','H'],'NO2':['N','O'],'CF3':['C','F'],'OH':['O','H'],'SH':['S','H'], 'COOH':['C','O']}

    # The dictionary below constructs the bond angle of the primitive portion (i.e. CH, NH, CF, NO, etc.).
    bond_length_dict = {'CH3':1.09, 'CN':1.17, 'NH2':1.02, 'NO2':1.25, 'CF3':1.36, 'OH':1.03, 'SH':1.34, 'COOH':1.25}

    bond_angle_dict = {'CH3':110, 'CN':180, 'NH2':120, 'NO2':122.5, 'CF3':112, 'OH':100, 'SH':100, 'COOH': 122.5}

    # The bond rotation dictionary is a list of the angles to rotate by, followed by the number of
    # times the motif must be repeated, in addition to the initial bond placement (e.g. CH3 needs 2
    # more CH bonds in additional to the one originally placed).
    bond_rotation_dict = {'CH3':[120, 2], 'CN':0, 'NH2':[180, 1], 'NO2': [180, 1], 'CF3': [120, 2], 'OH': 0, 'SH': 0, 'COOH': [180, 1]}

    return connection_length_dict, connection_atom_dict, bond_length_dict, bond_angle_dict, bond_rotation_dict

def vector_preparation(connected_atom_types, neighbors_not_to_replace, linker_to_analyze, linker_cart_coords, functionalization_position_on_linker,
    connection_length_dict, functional_group):
    """
    Prepares placement information for monoatomic and multiatomic functionalization.

    Parameters
    ----------
    connected_atom_types : list of str
        The chemical symbols of the atoms bonded to one of the not-to-be-replaced atoms (which is in turn bonded to the atom to functionalize).
    neighbors_not_to_replace : list of numpy.int32
        The global indices of not-to-be-replaced atoms bonded to the atom to functionalized.
    linker_to_analyze : list of numpy.int32
        The indices of the atoms in the linker.
    linker_cart_coords : numpy.ndarray of numpy.float64
        The Cartesian coordinates of the atoms in the linker. Shape is (number of atoms in linker, 3).
    functionalization_position_on_linker : int
        The linker index of the atom to functionalize.
    connection_length_dict : dict
        How far to place the functional group from the connecting carbon.
    functional_group : str
        The functional group to use for MOF functionalization.

    Returns
    -------
    initial_placement : numpy.ndarray of numpy.float64
        The Cartesian coordinates of the connecting atom of the functional group.
        Pertinent to mono and multiatomic functionalization. Shape is (3,).
    directional_unit_vector : numpy.ndarray of numpy.float64
        Vector resulting from the addition of the two vectors from the two not-to-be-replaced neighbor atoms to the atom to be functionalized.
        Normalized.
        Pertinent to multiatomic functionalization later. Shape is (3,).
    norm_cp : numpy.ndarray of numpy.float64
        Cross product of the two vectors from the two not-to-be-replaced neighbor atoms to the atom to be functionalized.
        Normalized.
        Pertinent to multiatomic functionalization. Shape is (3,).

    """
    # NOTE: there is an assumption here that neighbors_not_to_replace is length 2, based on the code below.
        # These are the two not-to-be-replaced neighbor atoms.

    if 'H' in connected_atom_types: # The first of the not-to-be-replaced atoms has hydrogen bonded to it.
        print('First type of vector_a')
        vector_a = neighbors_not_to_replace[0] # vector is a misnomer here, but this variable is used to calculate a vector in a few lines.
        vector_b = neighbors_not_to_replace[1]
    else: # The first of the not-to-be-replaced atoms does not have hydrogen bonded to it.
        print('Second type of vector_a')
        vector_a = neighbors_not_to_replace[1]
        vector_b = neighbors_not_to_replace[0]

    neighbor1 = linker_to_analyze.index(vector_a) # The index of the index vector_a in the list of indices linker_to_analyze.
    neighbor2 = linker_to_analyze.index(vector_b)
    v1 = linker_cart_coords[functionalization_position_on_linker]-linker_cart_coords[neighbor1] # Vector from one of the not-to-be-replaced atoms to the atom to functionalize.
    v2 = linker_cart_coords[functionalization_position_on_linker]-linker_cart_coords[neighbor2] # Vector from the other of the not-to-be-replaced atoms to the atom to functionalize.

    cp = np.cross(v1, v2) # cross product to get a perpendicular vector to v1 and v2
    # a, b, c = cp
    directional_unit_vector = (v1+v2)/np.linalg.norm(v1+v2) # points in the direction of where the functional group should be placed. Draw it out to visualize it!
    norm_cp = cp/np.linalg.norm(cp) # Normalizing the cross product

    #### Find the vector for placement of the connecting atom
    direction_to_place_connecting = directional_unit_vector*connection_length_dict[functional_group] # This is a vector. Direction * magnitude

    # Apply the vector to the starting position, linker_cart_coords[functionalization_position_on_linker]
    initial_placement = linker_cart_coords[functionalization_position_on_linker]+direction_to_place_connecting

    #### Check if it is in the plane with the original 3 atoms (it should be)
    planartest = checkplanar(initial_placement, linker_cart_coords[functionalization_position_on_linker], linker_cart_coords[neighbor1],linker_cart_coords[neighbor2])
    if not planartest:
        raise ValueError('This atom is not planar to the original 3 atoms. Issue detected. Exiting.')

    return initial_placement, directional_unit_vector, norm_cp

def connecting_atom_functionalization(connection_atom_dict,
    functional_group,
    initial_placement,
    molcif):
    """
    Adds the connecting atom of the functional group.

    Parameters
    ----------
    connection_atom_dict : dict
        For each functional group, indicates the element in the functional group that connects to the carbon atom; then indicates remaining elements for multiatomic functional groups.
    functional_group : str
        The functional group to use for MOF functionalization.
    initial_placement : numpy.ndarray of numpy.float64
        The Cartesian coordinates of the connecting atom of the functional group.
        Shape is (3,).
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file to be functionalized.

    Returns
    -------
    molcif : molSimplify.Classes.mol3D.mol3D
        The modified cell of the cif file.
    atom_types_to_add : list of str
        The chemical symbols of the atoms added. When leaving this function, length is 1.
    additions_to_cart : numpy.ndarray of numpy.float64
        The Cartesian coordinates of the atoms added during functionalization. When leaving this function, shape is (1, 3).

    """
    connecting_atom = atom3D(connection_atom_dict[functional_group][0], initial_placement)
    molcif.addAtom(connecting_atom)
    additions_to_cart = np.array([initial_placement])
    atom_types_to_add = []
    atom_types_to_add.append(connection_atom_dict[functional_group][0])
    return molcif, atom_types_to_add, additions_to_cart

def multiatomic_functionalization(connection_atom_dict,
    bond_length_dict,
    bond_angle_dict,
    bond_rotation_dict,
    functional_group,
    directional_unit_vector,
    norm_cp,
    initial_placement,
    additions_to_cart,
    atom_types_to_add,
    molcif,
    update_molcif=True,
    additional_atom_offset=0):
    """
    Adds functional group atoms that are not the connecting atom.
    Adds these atoms to the connecting atom.

    Parameters
    ----------
    connection_atom_dict : dict
        For each functional group, indicates the element in the functional group that connects to the carbon atom; then indicates remaining elements for multiatomic functional groups.
    bond_length_dict : dict
        For each functional group (as applicable), indicates the bond length of the connecting atom on the functional group to the other atoms in the functional group, in angstroms.
    bond_angle_dict : dict
        For each functional group (as applicable), indicates how far off a straight line the non-connecting functional group atoms are,
        relative to a directional unit vector that goes through the connecting carbon and the connecting atom on the functional group.
        For example, for CF3, this is the C-C-F angle, where the first C is the carbon being functionalized and the second C is part of the CF3 functional group.
    bond_rotation_dict : dict
        For each functional group (as applicable), indicates the angle rotation and the number of rotations for non-connecting atoms. E.g. for CH3, the hydrogens are 120 degrees rotated apart.
    functional_group : str
        The functional group to use for MOF functionalization.
    directional_unit_vector : numpy.ndarray of numpy.float64
        Vector resulting from the addition of the two vectors from the two not-to-be-replaced neighbor atoms to the atom to be functionalized.
        Normalized.
        Shape is (3,).
    norm_cp : numpy.ndarray of numpy.float64
        Cross product of the two vectors from the two not-to-be-replaced neighbor atoms to the atom to be functionalized.
        Normalized.
        Shape is (3,).
    initial_placement : numpy.ndarray of numpy.float64
        The Cartesian coordinates of the connecting atom of the functional group.
        Shape is (3,).
    additions_to_cart : numpy.ndarray of numpy.float64
        The Cartesian coordinates of the atoms added during functionalization. When entering this function, shape is (1, 3).
    atom_types_to_add : list of str
        The chemical symbols of the atoms added. When entering this function, length is 1.
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the MOF to be functionalized.
    additional_atom_offset : float
        Extent to which to rotate the placement of depth 2 functional group atoms. Give in degrees.
        Useful for preventing atomic overlap / unintended bonds.

    Returns
    -------
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the functionalized MOF.
    atom_types_to_add : list of str
        The chemical symbols of the atoms added. These are the atoms in the functional group.
    additions_to_cart : numpy.ndarray of numpy.float64
        The Cartesian coordinates of the atoms added during functionalization. Shape is (number of atoms in functional group, 3).

    """
    #### Take direction of placing the first connecting atom. Rotate it within the plane made by the 4 points, by the angle from the dictionary.
    deg2rad = 2*np.pi/360
    bonded_placement = ((np.cos((180-bond_angle_dict[functional_group])*deg2rad))*directional_unit_vector*bond_length_dict[functional_group]+
        (np.sin((180-bond_angle_dict[functional_group])*deg2rad))*(np.cross(norm_cp, directional_unit_vector*bond_length_dict[functional_group])))

    # The shape of bonded_placement is (3,), as is the shape of initial_placement

    ##### Find where the first functionalization should be placed.
    final_placement = initial_placement + bonded_placement
    final_placement = np.array(PointRotateAxis(directional_unit_vector.tolist(),initial_placement.tolist(),final_placement.tolist(), additional_atom_offset*deg2rad)) # Apply rotation additional_atom_offset if requested.
    # This is where one of the hydrogens in CH3 is added, for example.
    bonded_atom = atom3D(connection_atom_dict[functional_group][1], final_placement)
    molcif.addAtom(bonded_atom)
    first_o_index = molcif.natoms-1
    # Change Akash
    additions_to_cart = np.concatenate((additions_to_cart, np.array([final_placement])))
    atom_types_to_add.append(connection_atom_dict[functional_group][1])

    if bond_rotation_dict[functional_group] != 0:
        rotated_atom3Ds = []
        num_rotations = bond_rotation_dict[functional_group][1]
        counter = 1
        while counter <= num_rotations: # This is where the two extra H's in CH3 are added, for example.
            rotate_by = bond_rotation_dict[functional_group][0]*counter*deg2rad
            # print('initial coords of CH bond',final_placement)
            rotated_coords = PointRotateAxis(directional_unit_vector.tolist(),initial_placement.tolist(),final_placement.tolist(),rotate_by)
            # print('rotated coords of CH bond, rotation number',rotated_coords, counter)
            rotated_atom3Ds.append(atom3D(connection_atom_dict[functional_group][1], rotated_coords))
            additions_to_cart = np.concatenate((additions_to_cart, np.array([rotated_coords])))
            atom_types_to_add.append(connection_atom_dict[functional_group][1])
            counter += 1

        if update_molcif:
            [molcif.addAtom(val) for val in rotated_atom3Ds]

        # Change Akash
    if functional_group == 'COOH':
        cooh_h_coords = molcif.getAtom(first_o_index).coords() + directional_unit_vector
        additions_to_cart = np.concatenate((additions_to_cart, np.array([cooh_h_coords])))
        atom_types_to_add.append('H')

    return molcif, atom_types_to_add, additions_to_cart


def make_networkx_graph(adj_matrix):
    """
    Makes a networkx graph of the bonds of the atoms in the linker specified by adj_matrix.

    Parameters
    ----------
    adj_matrix : scipy.sparse.csr.csr_matrix
        The atom connections of a linker subgraph. Indicates what atoms are bonded to what.

    Returns
    -------
    G : networkx.classes.graph.Graph
        The networkx graph of the bonds of the atoms in the linker.

    """
    if scipy.sparse.issparse(adj_matrix):
        adj_matrix = adj_matrix.todense()
    rows, cols = np.where(np.array(adj_matrix) == 1) # 1 indicates a bond. 0 indicates no bond.
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def get_linkers(molcif, adj_matrix, allatomtypes):
    """
    Returns information on the linkers in the provided MOF.
    Similar to the code in molSimplify.Informatics.MOF.MOF_descriptors.get_MOF_descriptors. Specifically, step 1: metallic part

    Parameters
    ----------
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    adj_matrix : scipy.sparse.csr.csr_matrix
        1 represents a bond, 0 represents no bond. Shape is (number of atoms, number of atoms).
    allatomtypes : list of str
        The atom types of the cif file, indicated by chemical symbols like 'O' and 'Cu'. Length is the number of atoms.

    Returns
    -------
    linker_list : list of lists of ints
        Each inner list is its own separate linker. The ints are the global atom indices of that linker. Length is # of linkers.
    linker_subgraphlist : list of scipy.sparse.csr.csr_matrix
        The atom connections in the linker subgraph. Length is # of linkers.

    """
    SBUlist = set() # Will contain the indices of atoms belonging to SBUs.
    [SBUlist.update(set([metal])) for metal in molcif.findMetal(transition_metals_only=False)] # Consider all metals as part of the SBUs.
    [SBUlist.update(set(molcif.getBondedAtomsSmart(metal))) for metal in molcif.findMetal(transition_metals_only=False)] # Also consider all atoms bonded to a metals part of the SBUs.

    removelist = set()
    [removelist.update(set([metal])) for metal in molcif.findMetal(transition_metals_only=False)] # Remove all metals as part of the SBU.
    # for metal in removelist:
    #     bonded_atoms = set(molcif.getBondedAtomsSmart(metal))
    #     bonded_atoms_types = set([str(allatomtypes[at]) for at in set(molcif.getBondedAtomsSmart(metal))]) # The types of elements bonded to metals. E.g. oxygen, carbon, etc.

    # Add to removelist any atoms that are only bonded to metals (not counting hydrogens).
        # The all() function returns True if all items in an iterable are true, otherwise it returns False.
    [removelist.update(set([atom])) for atom in SBUlist if all((molcif.getAtom(val).ismetal() or
        molcif.getAtom(val).symbol().upper() == 'H') for val in molcif.getBondedAtomsSmart(atom))]

    allatoms = set(range(0, adj_matrix.shape[0])) # A set that goes from 0 to the number of atoms - 1
    linkers = allatoms - removelist
    linker_list, linker_subgraphlist = get_closed_subgraph(linkers.copy(), removelist.copy(), adj_matrix)
    return linker_list, linker_subgraphlist

def connected_atoms_from_adjmat(adj_matrix, index, allatomtypes):
    """
    Finds the atoms connected to the atom with the index `index`.
    This function works with sparse matrices. Assumes you handed a sparse matrix.

    Parameters
    ----------
    adj_matrix : scipy.sparse.csr.csr_matrix
        1 represents a bond, 0 represents no bond. Shape is (number of atoms, number of atoms).
    index : int
        The index of the atom for which the connected atoms will be found.
    allatomtypes : list of str
        The atom types of the cif file, indicated by chemical symbols like 'O' and 'Cu'. Length is the number of atoms.

    Returns
    -------
    connected_atom_list : numpy.ndarray of numpy.int32
        The indices of the atoms connected to the atom of interest.
    connected_atom_types : list of str
        The chemical symbols of the atoms connected to the atom of interest. Order is the same as connected_atom_list.

    """
    connected_atom_list = np.nonzero(adj_matrix[index,:])[1] # indices of atoms with bonds to the atom with the index `index`
    connected_atom_types = [allatomtypes[j] for j in connected_atom_list]

    return connected_atom_list, connected_atom_types

def apply_monatomic_functionalization(molcif, allatomtypes, atom_to_replace, functional_group, functionalization_counter):
    """
    Deprecated way of executing monatomic functionalization.
    Does not take into account the different bond lengths of different functional groups, as is done in vector_preparation's calculation of initial_placement.

    """
    molcif.getAtom(atom_to_replace).mutate(functional_group) # Replaces one atom3D with another.
    allatomtypes[atom_to_replace] = functional_group
    functionalization_counter -= 1
    if functionalization_counter == 0:
        functionalized = True
    else:
        functionalized = False
    return molcif, allatomtypes, functionalization_counter, functionalized

def check_support(functional_group):
    """
    Raises a ValueError if the functional_group is not in the pre-defined list of supported functional groups.

    Parameters
    ----------
    functional_group : str
        Chemical formula for the functional group.

    Returns
    -------
    dict_approach : bool
        Indicates whether the functional group is added using the dictionary approach.
        If not, functional group is added using some template xyz about its structure, through a merge of mol3D objects.

    """
    supported_functional_groups = ['F', 'Cl', 'Br', 'I', 'CH3', 'CN', 'NH2', 'NO2', 'CF3', 'OH', 'SH', 'COOH', '1-3-5-7-octatetraene', '1-3-5-7-9-11-dodecahexaene', 'acetamide', 'penta-1-3-diene', 'styrene',
                                   'propanamine', 'propyl', 'pyrrole', 'ethyl', 'methyl-amine', '1-3-5-Heptatriene', 'acetylene', 'ethyl-amine', '1-3-5-7-9-decapentaene',
                                   'phenylacetylene', '1-3-5-7-9-undecapentaene', 'propyne', '3Z-hexa-1-3-5-triene', 'sulfonic']

    # These functional groups are not added via the dictionary approach since they go more than two atoms deep, or are not uniform in bond length at every atom depth.
    supported_functional_groups_by_mol3D_merge = ['OCF3', 'OCH3']
    if functional_group not in supported_functional_groups+supported_functional_groups_by_mol3D_merge:
        raise ValueError('Unsupported functional group requested.')
    else:
        dict_approach = functional_group in supported_functional_groups
        return dict_approach

def functionalize_MOF_at_indices(cif_file, path2write, functional_group, func_indices, gram_schmidt, additional_atom_offset=0):
    """
    Functionalizes the provided MOF and writes the functionalized version to a cif file.
    Functionalizes at the specified indices func_indices, provided the atoms at those indices are sp2 carbons with a hydrogen atom.

    Parameters
    ----------
    cif_file : str
        The path to the cif file to be functionalized.
    path2write : str
        The folder path where the cif of the functionalized MOF will be written.
    functional_group : str
        The functional group to use for MOF functionalization.
    func_indices : list of int
        The indices of the atoms at which to functionalize. Zero-indexed.
    additional_atom_offset : float or list of float
        Extent to which to rotate the placement of depth 2 functional group atoms. Give in degrees.
        Useful for preventing atomic overlap / unintended bonds.
        If list, must be the length of func_indices.

    Returns
    -------
    None

    """

    if not isinstance(additional_atom_offset, list):
        # Convert to a list.
        additional_atom_offset = [additional_atom_offset] * len(func_indices)

    dict_approach = check_support(functional_group)
    if not dict_approach:
        # The requested functional group is more than two atoms deep, or has differing bond lengths/atom identities at a given depth.
        # Use a different function for treating these.
        functionalize_MOF_at_indices_mol3D_merge(cif_file, path2write, functional_group, func_indices, gram_schmidt, additional_atom_offset)
        return

    ### Start of repeat code (in common with functionalize_MOF) ###
    base_mof_name = os.path.basename(cif_file)
    if base_mof_name.endswith('.cif'):
        base_mof_name = base_mof_name[:-4]

    # Read the cif file and make the cell for fractional coordinates
    cpar, allatomtypes, fcoords = readcif(cif_file)
    molcif, cell_vector, alpha, beta, gamma = import_from_cif(cif_file, True)
    cell_v = np.array(cell_vector)
    cart_coords = fractional2cart(fcoords, cell_v)
    distance_mat = compute_distance_matrix(cell_v, cart_coords)
    adj_matrix, _ = compute_adj_matrix(distance_mat, allatomtypes)
    molcif.graph = adj_matrix.todense()

    ### End of repeat code ###

    ###### At this point, we have most things we need to functionalize.
    # Thus the first step is to break down into linkers. This uses what we developed for MOF featurization
    linker_list, linker_subgraphlist = get_linkers(molcif, adj_matrix, allatomtypes)

    ###### We need to then figure out which atoms to functionalize.
    checkedlist = set() # Keeps track of the atoms that have already been checked for functionalization.
    delete_list = [] # Collect all of the H that need to be deleted later.
    extra_atom_coords = []
    extra_atom_types = []
    functionalized_atoms = []

    for _i, func_index in enumerate(func_indices):
        print(f'On {_i+1} out of {len(func_indices)}')
        atom_to_functionalize = allatomtypes[func_index]

        # Atoms that are connected to atom at index func_index.
        connected_atom_list, connected_atom_types = connected_atoms_from_adjmat(adj_matrix, func_index, allatomtypes)

        if atom_to_functionalize != 'C': # Assumes that functionalization is performed on a C atom.
            raise ValueError('Invalid atom to functionalize: not a carbon atom.')
        elif 'H' not in connected_atom_types: # Must functionalize where an H was.
            raise ValueError('Invalid atom to functionalize: no hydrogen neighbor to replace.')
        elif len(connected_atom_types) != 3: # Needs sp2 C.
            raise ValueError('Invalid atom to functionalize: not an sp2 carbon atom.')
        else: # atom_to_functionalize is a suitable location for functionalization.
            # Identifying the linker that has atom atom_to_functionalize.
            # Also adds all the atoms in the identified linker to checkedlist.
            linker_to_analyze, linker_to_analyze_index, _ = linker_identification(linker_list, func_index, checkedlist) # checkedlist is not important here.

            linker_atom_types, linker_graph, linker_cart_coords = analyze_linker(cart_coords,
                linker_to_analyze,
                allatomtypes,
                linker_subgraphlist,
                linker_to_analyze_index,
                cell_v,
                )

            """""""""
            Functionalization.
            """""""""

            for k, connected_atom in enumerate(connected_atom_types): # Look through the atoms bonded to atom i.
                if connected_atom == 'H':

                    functionalization_counter = 1
                    molcif, functionalization_counter, functionalized, delete_list, extra_atom_coords, extra_atom_types, functionalized_atoms, unintend_bond = first_functionalization(cell_v, molcif,
                        allatomtypes,
                        func_index,
                        connected_atom_list,
                        k,
                        functional_group,
                        linker_cart_coords,
                        linker_to_analyze,
                        linker_atom_types,
                        linker_graph,
                        functionalization_counter,
                        delete_list,
                        extra_atom_coords,
                        extra_atom_types,
                        functionalized_atoms,
                        additional_atom_offset=additional_atom_offset[_i]
                        ) # functionalized_atoms is not important here.

                    break # Don't search the rest of the connected atoms if replaced a hydrogen and functionalized already at the atom with index i.

    """""""""
    Apply delete_list and extra_atom_types to make final_atom_types and new_coord_list.
    """""""""
    # Deleting atoms (hydrogens that are replaced by functional groups)
    new_coord_list, final_atom_types = atom_deletion(cart_coords, allatomtypes, delete_list)

    # Adding atoms (the atoms in the functional groups)
    allatomtypes, fcoords = atom_addition(extra_atom_types, final_atom_types, new_coord_list, extra_atom_coords, cell_v)

    # Check to make sure none of the functional group atoms are too close to other atoms in the CIF
    # If so, code will interpret atoms to be bonded that should not be.
    post_functionalization_overlap_and_bonding_check(cell_v, allatomtypes, fcoords, extra_atom_types)

    """""""""
    Write the cif.
    """""""""
    cif_folder = f'{path2write}/cif'
    mkdir_if_absent(cif_folder)
    write_cif(f'{cif_folder}/functionalized_{base_mof_name}_{functional_group}_index.cif', cpar, fcoords, allatomtypes)


def functionalize_MOF_at_indices_mol3D_merge(cif_file, path2write, functional_group, func_indices, gram_schmidt, additional_atom_offset):
    """
    Functionalizes the provided MOF and writes the functionalized version to a cif file.
    Functionalizes at the specified indices func_indices, provided the atoms at those indices are sp2 carbons with a hydrogen atom.
    Differs from functionalize_MOF_at_indices in that this function handles more challenging functionalizations.
    Works with geometries stored in the folder monofunctionalized_BDC.

    Parameters
    ----------
    cif_file : str
        The path to the cif file to be functionalized.
    path2write : str
        The folder path where the cif of the functionalized MOF will be written.
    functional_group : str
        The functional group to use for MOF functionalization.
    func_indices : list of int
        The indices of the atoms at which to functionalize. Zero-indexed.
    additional_atom_offset : list of float
        Extent to which to rotate the placement of depth 2 functional group atoms. Give in degrees.
        Useful for preventing atomic overlap / unintended bonds.
        Must be the length of func_indices.

    Returns
    -------
    None

    """

    ### Start of repeat code (in common with functionalize_MOF_at_indices) ###
    base_mof_name = os.path.basename(cif_file)
    if base_mof_name.endswith('.cif'):
        base_mof_name = base_mof_name[:-4]

    # Read the cif file and make the cell for fractional coordinates
    cpar, allatomtypes, fcoords = readcif(cif_file)
    molcif, cell_vector, alpha, beta, gamma = import_from_cif(cif_file, True)
    cell_v = np.array(cell_vector)
    cart_coords = fractional2cart(fcoords, cell_v)
    distance_mat = compute_distance_matrix(cell_v, cart_coords)
    adj_matrix, _ = compute_adj_matrix(distance_mat, allatomtypes)
    molcif.graph = adj_matrix.todense()

    ### End of repeat code ###


    ### Section with the functional group template ###

    # Load in the mol3D from the folder molSimplify folder monofunctionalized_BDC.
    functional_group_template = mol3D()
    func_group_xyz_path = resource_filename(Requirement.parse(
                "molSimplify"), f"molSimplify/Informatics/MOF/monofunctionalized_BDC/{functional_group}.xyz")
    functional_group_template.readfromxyz(func_group_xyz_path) # This is a whole BDC linker with the requested functional group on it.

    # Read information about the important indices of the functional_group_template.
    fg_anchor_index, fg_fg_indices, fg_main_carbon_index, fg_carbon_neighbor_indices = INDEX_INFO[functional_group] # fg stands for functional group.

    ### Begin functionalization process ###

    # To keep track of the hydrogen atoms replaced with functional groups.
    H_indices_to_delete = []

    # To keep track of the functional groups to merge on the cif. These are mol3D objects.
    func_groups = []

    for _i, func_index in enumerate(func_indices): # Loop over all indices to be functionalized.
        print(f'On {_i+1} out of {len(func_indices)}')
        atom_to_functionalize = allatomtypes[func_index]

        # Atoms that are connected to atom at index func_index.
        connected_atom_list, connected_atom_types = connected_atoms_from_adjmat(adj_matrix, func_index, allatomtypes)

        if atom_to_functionalize != 'C': # Assumes that functionalization is performed on a C atom with two C neighbors and one H neighbor.
            raise ValueError('Invalid atom to functionalize: not a carbon atom.')
        elif 'H' not in connected_atom_types: # Must functionalize where an H was.
            raise ValueError('Invalid atom to functionalize: no hydrogen neighbor to replace.')
        elif len(connected_atom_types) != 3: # Needs sp2 C.
            raise ValueError('Invalid atom to functionalize: not an sp2 carbon atom.')
        else: # atom_to_functionalize is a suitable location for functionalization.

            """""""""
            Functionalization.
            """""""""

            carbon_neighbor_indices = []
            for k, connected_atom in enumerate(connected_atom_types): # Look through the atoms bonded to atom i.
                if connected_atom == 'H':
                    H_indices_to_delete.append(connected_atom_list[k])
                elif connected_atom == 'C':
                    carbon_neighbor_indices.append(connected_atom_list[k])
            # Checking to make sure the results of the above for loop make sense.
            if len(carbon_neighbor_indices) != 2:
                raise ValueError(f"Unexpected number of carbon neighbors {len(carbon_neighbor_indices)}.")

            """""""""
            Aligning a copy of the functional_group_template to where we want to functionalize.
            """""""""

            functional_group_clone = mol3D()
            functional_group_clone.copymol3D(functional_group_template)

            ### Doing some stuff with the MOF mol3D ###

            # How should we rotate functional_group_clone so it aligns with the carbon we want to functionalize?
            # Answer: align a copy of functional_group_template with the three carbons of interest in the cif.
            molcif_clone = mol3D()
            molcif_clone.copymol3D(molcif)

            # Shift the two neighbor carbons by cell vectors until they are closest to the carbon to be functionalized.
            shift_carbon_1 = compute_image_flag(cell_v, fcoords[func_index], fcoords[carbon_neighbor_indices[0]])
            shift_carbon_2 = compute_image_flag(cell_v, fcoords[func_index], fcoords[carbon_neighbor_indices[1]])
            # Changing the cartesian coordinates of the two carbon neighbors by the required cell vector shifts.
            carbon_1_cart = fractional2cart(fcoords[carbon_neighbor_indices[0]]+shift_carbon_1, cell_v)
            carbon_2_cart = fractional2cart(fcoords[carbon_neighbor_indices[1]]+shift_carbon_2, cell_v)
            # Setting new positions.
            molcif_clone.getAtoms()[carbon_neighbor_indices[0]].setcoords(carbon_1_cart)
            molcif_clone.getAtoms()[carbon_neighbor_indices[1]].setcoords(carbon_2_cart)
            molcif_clone.getAtoms()[func_index].setcoords(cart_coords[func_index])

            # Translate first. Just the difference between the two main carbon atoms.
            # So, will make the two main carbons overlap.
            translation_vector = np.array(molcif_clone.getAtom(func_index).coords()) - np.array(functional_group_template.getAtom(fg_main_carbon_index).coords())

            ### Now, answer the question of how much to rotate the functional group to align it to the carbon in the CIF. ###
            initial_guess = np.zeros(3)
            rotation_vector = scipy.optimize.fmin(alignment_objective, initial_guess, args=(molcif_clone, func_index,
                carbon_neighbor_indices, functional_group_template, fg_main_carbon_index, fg_carbon_neighbor_indices, translation_vector))

            # Unpacking
            x_rotation = rotation_vector[0]
            y_rotation = rotation_vector[1]
            z_rotation = rotation_vector[2]

            # Applying the translation and rotation to the functional_group_clone.
            functional_group_clone.translate(translation_vector)
            main_carbon_coordinate = functional_group_clone.getAtom(fg_main_carbon_index).coords()
            functional_group_clone = rotate_around_axis(functional_group_clone, main_carbon_coordinate, [1,0,0], x_rotation)
            functional_group_clone = rotate_around_axis(functional_group_clone, main_carbon_coordinate, [0,1,0], y_rotation)
            functional_group_clone = rotate_around_axis(functional_group_clone, main_carbon_coordinate, [0,0,1], z_rotation)

            # Account for additional_atom_offset
            # Vector between functionalized carbon and the anchor atom of the functional group (e.g. the C in -CH3 functional group).
            anchor_coordinate = functional_group_clone.getAtom(fg_anchor_index).coords()
            direction_vector = np.array(anchor_coordinate) - np.array(main_carbon_coordinate)
            functional_group_clone = rotate_around_axis(functional_group_clone, main_carbon_coordinate, direction_vector, additional_atom_offset[_i])

            # Delete unwanted functional_group_template atoms.
            num_atoms = functional_group_clone.getNumAtoms()
            clone_indices = range(num_atoms)
            clone_indices_to_remove = [idx for idx in clone_indices if idx not in fg_fg_indices]
            functional_group_clone.deleteatoms(clone_indices_to_remove)
            func_groups.append(functional_group_clone)

    # Combining the mol3D objects.
    for new_fg in func_groups:
        molcif = molcif.combine(new_fg, dirty=True) # Adds the functional group to the end of molcif (index-wise)

    # Delete hydrogen atoms on the functionalized carbons.
    molcif.deleteatoms(H_indices_to_delete)

    # Getting the fractional coordinates.
    cartesian_coordinates = molcif.coordsvect()
    fcoords = frac_coord(cartesian_coordinates, cell_v)
    # Getting the atom types.
    allatomtypes = molcif.symvect()

    # """""""""
    # Write the cif.
    # """""""""
    cif_folder = f'{path2write}/cif'
    mkdir_if_absent(cif_folder)
    write_cif(f'{cif_folder}/functionalized_{base_mof_name}_{functional_group}_index.cif', cpar, fcoords, allatomtypes)


def alignment_objective(rotation_vector, molcif_clone, MOF_main_carbon_index, MOF_carbon_neighbor_indices,
    functional_group_template, fg_main_carbon_index, fg_carbon_neighbor_indices, translation_vector):
    """
    The objective function to be minimized by finding the optimal x, y, and z rotation angles.

    Parameters
    ----------
    rotation_vector : numpy.ndarray
        The vector by which to rotate the functional group BDC template.
        Shape is (3,).
    molcif_clone : mol3D
        Structure information on the MOF.
    MOF_main_carbon_index : int
        The index of the carbon to be functionalized in molcif_clone.
    MOF_carbon_neighbor_indices : list of int
        The two indices of the two carbon atoms bonded to the main carbon in molcif_clone.
    functional_group_template : mol3D
        Structure information on the functional group BDC template.
    fg_main_carbon_index : int
        The index of the carbon that is functionalized in functional_group_template.
    fg_carbon_neighbor_indices : list of int
        The two indices of the two carbon atoms bonded to the main carbon atom in functional_group_template.
    translation_vector : numpy.ndarray
        The vector by which to translate the functional group BDC template.
        Shape is (3,).

    Returns
    -------
    objective_function : float
        The sum of the distances between each of the three carbons of interest.
        Compares the carbons in the template BDC with the functional group, and the carbons in the MOF.
        The main carbon is the carbon at which functionalization occurs. The neighbors are bonded to that main carbon.

    """

    # Unpacking
    x_rotation = rotation_vector[0]
    y_rotation = rotation_vector[1]
    z_rotation = rotation_vector[2]

    template_copy = mol3D()
    template_copy.copymol3D(functional_group_template)

    # Translate template_copy.
    template_copy.translate(translation_vector)

    # Rotate template_copy through an axis passing through the main carbon, i.e. the carbon to be functionalized.
    main_carbon_coordinate = template_copy.getAtom(fg_main_carbon_index).coords()
    template_copy = rotate_around_axis(template_copy, main_carbon_coordinate, [1,0,0], x_rotation)
    template_copy = rotate_around_axis(template_copy, main_carbon_coordinate, [0,1,0], y_rotation)
    template_copy = rotate_around_axis(template_copy, main_carbon_coordinate, [0,0,1], z_rotation)

    # NOTE: small degree of flexibility here. Can align carbon 1 of MOF_carbon_neighbor_indices to carbon 1 of fg_carbon_neighbor_indices
    # Or align carbon 1 of MOF_carbon_neighbor_indices to carbon 2 of fg_carbon_neighbor_indices.
    # I go with the former approach here.

    distance1 = distance(molcif_clone.getAtom(MOF_main_carbon_index).coords(), template_copy.getAtom(fg_main_carbon_index).coords())
    distance2 = distance(molcif_clone.getAtom(MOF_carbon_neighbor_indices[0]).coords(), template_copy.getAtom(fg_carbon_neighbor_indices[0]).coords())
    distance3 = distance(molcif_clone.getAtom(MOF_carbon_neighbor_indices[1]).coords(), template_copy.getAtom(fg_carbon_neighbor_indices[1]).coords())

    objective_function = distance1 + distance2 + distance3 # Want to minimize this.
    return objective_function


def post_functionalization_overlap_and_bonding_check(cell_v, all_atom_types, fcoords, extra_atom_types):
    """
    Prints information on whether the introduced functional group atoms are overlapping with other atoms.
    Also prints information on the interpreted bonds of the introduced functional group atoms. Useful to make sure the functional group atoms are not too close to other atoms.

    Parameters
    ----------
    cell_v : numpy.ndarray of numpy.float64
        Each row corresponds to one of the cell vectors. Shape is (3, 3).
    allatomtypes : list of str
        The atom types of the MOF, indicated by chemical symbols like 'O' and 'Cu'. Length is the number of atoms.
    fcoords : numpy.ndarray of numpy.float64
        The fractional positions of the crystal atoms. Shape is (number of atoms, 3).
    extra_atom_types : list of list of str
        The chemical symbols of the atoms added through functional groups. Each inner list is a functional group.

    Returns
    -------
    None

    """
    # cart_coords = fractional2cart(fcoords, cell_v)
    # distance_mat = compute_distance_matrix(cell_v, cart_coords)
    # try:
    #     adj_matrix, _ = compute_adj_matrix(distance_mat, allatomtypes, handle_overlap=False) # Will throw an error if atoms are overlapping after functionalization.
    #     return False
    # except Exception:
    #     return True
    cart_coords = fractional2cart(fcoords, cell_v)
    distance_mat = compute_distance_matrix(cell_v, cart_coords)
    try:
        adj_matrix, _ = compute_adj_matrix(distance_mat, all_atom_types, wiggle_room=0.9, handle_overlap=False) # Will throw an error if atoms are overlapping after functionalization.
    except:
        return True
    adj_matrix = adj_matrix.todense()
    adj_matrix = np.squeeze(np.asarray(adj_matrix)) # Converting from numpy.matrix to numpy.array

    all_MOF_atoms_index = [i for i, atom in enumerate(all_atom_types)]

    # Since functional group atoms are added to the end of the atom type list and fractional coordinate numpy array (see function atom_addition),
    # we just check the last few rows of the adjacency matrix.
    # These last few rows correspond to the functional group atoms that were added.
    bond_overlap = False

    if len(extra_atom_types) != 0:  ## No functional group added due to no sp2 carbon
        func_group_length = len(extra_atom_types[0]) # Number of atoms in each functional group.

        # Will be set to True if there is a bond overlap.

        starting_index = -1
        for i in range(len(extra_atom_types)):
            mof_index_list = [starting_index - j for j in range(func_group_length)]
            mof_actual_index_list = [index + len(all_atom_types) for index in mof_index_list] # The indices of the atoms in the functional group.
            all_MOF_atoms_index_dropped = [index for index in all_MOF_atoms_index if index not in mof_actual_index_list]
            number_bonding  = np.sum(adj_matrix[np.ix_(mof_actual_index_list, all_MOF_atoms_index_dropped)]) # Number of bonds to the functional group atoms.
            if number_bonding > 1:
                bond_overlap = True
                return bond_overlap

            starting_index -= func_group_length # Move to the next functional group atoms in the adjacency matrix.
    else:
        return True ## Returning True so that the cif without functionalization is not written again
    return bond_overlap

def atom_deletion(cart_coords, allatomtypes, delete_list):
    """
    Makes new coordinate and atom lists that disregard the undesired hydrogens.

    Parameters
    ----------
    cart_coords : numpy.ndarray of numpy.float64
        The Cartesian coordinates of the crystal atoms. Shape is (number of atoms, 3).
    allatomtypes : list of str
        The atom types of the MOF, indicated by chemical symbols like 'O' and 'Cu'. Length is the number of atoms.
    delete_list : list of numpy.int32
        The indices of atoms that are deleted because they are replaced by functional groups.

    Returns
    -------
    new_coord_list : numpy.ndarray
        The updated Cartesian coordinates of the crystal atoms. Shape is (number of atoms, 3).
    final_atom_types : list of str
        The updated atom types of the MOF, indicated by chemical symbols like 'O' and 'Cu'. Length is the number of atoms.

    """
    print('Initial shape', cart_coords.shape, len(allatomtypes))
    new_coord_list = None # Will be changed in the following lines.
    final_atom_types = []
    for cart_row in range(0, cart_coords.shape[0]): # Going from zero through number of atoms - 1
        if cart_row in delete_list:
            if allatomtypes[cart_row] != 'H':
                raise Exception('Error!') # As the code is implemented right now, only hydrogens should be being replaced.
            else:
                continue
        elif new_coord_list is None: # new_coord_list still needs to get its first entry
            new_coord_list = np.array([cart_coords[cart_row,:]])
        else:
            new_coord_list = np.concatenate((np.array(new_coord_list),np.array([cart_coords[cart_row,:]])),axis=0)
        final_atom_types.append(allatomtypes[cart_row])
    # Really have deletion by non-inclusion (see the continue statement).
    print('Shape after deletions', new_coord_list.shape, len(final_atom_types)) # (shape is (number of atoms, 3))

    return new_coord_list, final_atom_types

def atom_addition(extra_atom_types, final_atom_types, new_coord_list, extra_atom_coords, cell_v):
    """
    Adds the functional group atoms to the lists of atom types and coordinates.

    Parameters
    ----------
    extra_atom_types : list of list of str
        The chemical symbols of the atoms added through functional groups. Each inner list is a functional group.
    final_atom_types : list of str
        The atom types of the MOF, indicated by chemical symbols like 'O' and 'Cu'. Length is the number of atoms.
    new_coord_list : numpy.ndarray
        The Cartesian coordinates of the crystal atoms. Shape is (number of atoms, 3).
    extra_atom_coords : list of numpy.ndarray of numpy.float64
        The Cartesian coordinates of the atoms added during functionalization.
        Each item of the list is a new functional group.
        Shape of each numpy.ndarray is (number of atoms in functional group, 3).
    cell_v : numpy.ndarray of numpy.float64
        Each row corresponds to one of the cell vectors. Shape is (3, 3).

    Returns
    -------
    allatomtypes : list of str
        The atom types of the MOF, indicated by chemical symbols like 'O' and 'Cu'. Length is the number of atoms.
    fcoords : numpy.ndarray of numpy.float64
        The fractional positions of the crystal atoms. Shape is (number of atoms, 3).

    """
    # fg: functional group
    for fg_num, new_fg in enumerate(extra_atom_types): # Atoms added upon functionalization
        print('new_fg',new_fg)
        [final_atom_types.append(new_atom) for new_atom in new_fg] # Adding to the chemical symbols list.
        new_coord_list = np.concatenate((new_coord_list,np.array(extra_atom_coords[fg_num])),axis=0) # Adding to the coordinates array.
    print('Shape after deletions and inclusion of functional group atoms', new_coord_list.shape, len(final_atom_types))
    allatomtypes = final_atom_types
    fcoords = frac_coord(new_coord_list, cell_v)

    return allatomtypes, fcoords

def DS_remover(file_list):
    """
    Removes .DS_Store from the provided list of files.

    Parameters
    ----------
    file_list : list of str
        A list of files.

    Returns
    -------
    file_list : list of str
        An updated list of files.

    """
    file_list = [_i for _i in file_list if 'DS_Store' not in _i]
    return file_list

def mkdir_if_absent(folder_path):
    """
    Makes a folder at folder_path if it does not yet exist.

    Parameters
    ----------
    folder_path : str
        The folder path to check, and potentially at which to make a folder.

    Returns
    -------
    None

    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

### End of functions ###

def main():
    # 31 Functional groups currently supported
    func_group = ['F', 'Cl', 'Br', 'I', 'CH3', 'CN', 'NH2', 'NO2', 'CF3', 'OH', 'SH', 'COOH',
                  '1-3-5-7-octatetraene', '1-3-5-7-9-11-dodecahexaene', 'acetamide', 'penta-1-3-diene', 'styrene',
                 'propanamine', 'propyl', 'pyrrole', 'ethyl', 'methyl-amine', '1-3-5-Heptatriene', 'acetylene', 'ethyl-amine', '1-3-5-7-9-decapentaene',
                 'phenylacetylene', '1-3-5-7-9-undecapentaene', 'propyne', '3Z-hexa-1-3-5-triene', 'sulfonic']

    mof_path = './CIFs/'  ## Path to the MOF CIFs
    mof_name = os.listdir(mof_path)
    mofname = [mof for mof in mof_name if mof.endswith('.cif')]
    print(f'Total no of MOFs to be functionalized: {len(mofname)}')

    for mof in mofname:
        mof_wo_cif = mof.split('.')[0]

        # Defining folder names
        base_database_path = mof_path
        base_database_path_primitive = './primitive/'+mof_wo_cif+'/'
        base_write_path = './functionalized/'+mof_wo_cif+'/'
        base_database_path_without_overlap = './without_overlap/'+mof_wo_cif+'/'

        mkdir_if_absent(base_database_path_primitive)
        mkdir_if_absent(base_write_path)
        try:
            get_primitive(base_database_path+mof, base_database_path_primitive+mof) ## Getting the primitive CIF
        except ValueError:
            shutil.copyfile(base_database_path+mof, base_database_path_primitive+mof)

        primitive_cifs = DS_remover(os.listdir(base_database_path_primitive))

        mkdir_if_absent(base_database_path_without_overlap) ## Error handling for bad structures. Removing overlapping atoms in the CIF.

        for primitive_mof in primitive_cifs:
            overlap_removal(cif_path=base_database_path_primitive+primitive_mof, new_cif_path=base_database_path_without_overlap+primitive_mof)

        cifs_without_overlap = DS_remover(os.listdir(base_database_path_without_overlap))


        for MOF in cifs_without_overlap:
            MOF = MOF[0:-4]
            num_list = [1, 2] ## List containing the number of times functionalization will be done on each linker
            path = 3          ## Functionalization depth between successive functionalizations
            for num_func in num_list:
                for func in func_group:
                    super_func_folder = f'{base_write_path}{MOF}_{num_func}_functionalization/'
                    mkdir_if_absent(super_func_folder)

                    func_folder = f'{base_write_path}{MOF}_{num_func}_functionalization/{func}/'

                    if not os.path.exists(func_folder):
                        os.mkdir(func_folder)

                    # Functionalization if not already functionalized.

                    if not os.path.exists(f'{func_folder}/cif/functionalized_{MOF}_{func}_{num_func}.cif'):
                        functionalize_MOF(f'{base_database_path_without_overlap}{MOF}.cif', func_folder,
                        path_between_functionalizations=path, functionalization_limit=num_func, functional_group=func)


if __name__ == "__main__":
    main()
