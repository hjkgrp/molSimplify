import os
import re
import copy
import itertools
from scipy import sparse
import numpy as np
import pandas as pd
from molSimplify.Classes.atom3D import atom3D
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Scripts.cellbuilder_tools import import_from_cif
from molSimplify.Informatics.RACassemble import append_descriptors
from molSimplify.Informatics.autocorrelation import (
    generate_atomonly_autocorrelations,
    generate_atomonly_deltametrics,
    generate_full_complex_autocorrelations,
    generate_multimetal_autocorrelations,
    generate_multimetal_deltametrics,
    full_autocorrelation,
    )
from molSimplify.Informatics.MOF.atomic import (
    COVALENT_RADII,
    )
from molSimplify.Informatics.MOF.PBC_functions import (
    compute_adj_matrix,
    compute_distance_matrix3,
    compute_image_flag,
    frac_coord,
    fractional2cart,
    get_closed_subgraph,
    include_extra_shells,
    ligand_detect,
    linker_length,
    mkcell,
    readcif,
    slice_mat,
    write2file,
    writeXYZandGraph,
    XYZ_connected,
    )


#### NOTE: In addition to molSimplify's dependencies, this portion requires
#### pymatgen to be installed. The RACs are intended to be computed
#### on the primitive cell of the material. You can compute them
#### using the commented out snippet of code if necessary.

# Example usage is given at the bottom of the script.

'''<<<< CODE TO COMPUTE PRIMITIVE UNIT CELLS >>>>'''
#########################################################################################
# This MOF RAC generator assumes that pymatgen is installed.                            #
# Pymatgen is used to get the primitive cell.                                           #
#########################################################################################

### Defining pymatgen function for getting primitive, since molSimplify does not depend on pymatgen.
# This function is commented out and should be uncommented if used.
from pymatgen.io.cif import CifParser

def get_primitive(datapath, writepath, occupancy_tolerance=1):
    """
    Calculates and writes the primitive cell of the provided structure.

    Parameters
    ----------
    datapath : str
        The path to the cif file for which the primitive cell will be calculated.
    writepath : str
        The path to where the cif of the primitive cell will be written.
    occupancy_tolerance : float
        Scales down occupancies for a site to one, if they sum to below occupancy_tolerance.

    Returns
    -------
    None

    """

    s = CifParser(datapath, occupancy_tolerance=occupancy_tolerance).get_structures()[0]
    sprim = s.get_primitive_structure()
    sprim.to("cif", writepath) # Output structure to a file.

'''<<<< END OF CODE TO COMPUTE PRIMITIVE UNIT CELLS >>>>'''

def load_sbu_lc_descriptors(sbupath):
    """
    Loads the sbu and lc descriptors.

    Parameters
    ----------
    sbupath : str
        Path of the folder to make a csv file in.

    Returns
    -------
    sbu_descriptors : pandas.core.frame.DataFrame
        The existing SBU descriptors.
    lc_descriptors : pandas.core.frame.DataFrame
        The existing lc descriptors.

    """
    sbu_descriptor_path = os.path.dirname(sbupath)
    if os.path.getsize(sbu_descriptor_path+'/sbu_descriptors.csv')>0: # Checking if there is a file there.
        sbu_descriptors = pd.read_csv(sbu_descriptor_path+'/sbu_descriptors.csv')
    else:
        sbu_descriptors = pd.DataFrame()
    if os.path.getsize(sbu_descriptor_path+'/lc_descriptors.csv')>0: # Checking if there is a file there.
        lc_descriptors = pd.read_csv(sbu_descriptor_path+'/lc_descriptors.csv')
    else:
        lc_descriptors = pd.DataFrame()

    return sbu_descriptor_path, sbu_descriptors, lc_descriptors

def gen_and_append_desc(temp_mol, target_list, depth, descriptor_names, descriptors, Gval, feature_type):
    """
    Generate and append descriptors, both standard and delta.

    Parameters
    ----------
    temp_mol : molSimplify.Classes.mol3D.mol3D
        mol3D object of the linker of interest.
    target_list : list
        The indices of the atoms of interest in the linker.
    depth : int
        The depth of the RACs that are generated. See https://doi.org/10.1021/acs.jpca.7b08750 for more information.
    descriptor_names : list
        The RAC descriptor names. Will be appended to.
    descriptors : list
        The RAC descriptor values. Will be appended to.
    Gval : bool
        Whether to use G value as RAC.
    feature_type : str
        Either 'lc' or 'func'.

    Returns
    -------
    results_dictionary : dict
        deltametrics RACs
    descriptor_names : list
        The updated RAC descriptor names.
    descriptors : list
        The updated RAC descriptor values.
    """

    results_dictionary = generate_atomonly_autocorrelations(temp_mol, target_list, depth=depth, loud=False, oct=False, polarizability=True, Gval=Gval)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors, results_dictionary['colnames'], results_dictionary['results'], feature_type, 'all')

    results_dictionary = generate_atomonly_deltametrics(temp_mol, target_list, depth=depth, loud=False, oct=False, polarizability=True, Gval=Gval)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors, results_dictionary['colnames'], results_dictionary['results'], f'D_{feature_type}', 'all')

    return results_dictionary, descriptor_names, descriptors

#########################################################################################
# The RAC functions here average over the different SBUs or linkers present. This is    #
# because one MOF could have multiple different linkers or multiple SBUs, and we need   #
# the vector to be of constant dimension so we can correlate the output property.       #
#########################################################################################

def make_MOF_SBU_RACs(SBUlist, SBU_subgraph, molcif, depth, name, cell_v, anchoring_atoms, sbupath, linkeranchors_superlist,
    Gval, connections_list=False, connections_subgraphlist=False):
    """
    Generated RACs on the SBUs of the MOF, as well as on the lc atoms (SBU-connected atoms of linkers).

    Parameters
    ----------
    SBUlist : list of lists of numpy.int64
        Each inner list is its own separate SBU. The ints are the atom indices of that SBU. Length is # of SBUs.
    SBU_subgraph : list of scipy.sparse.csr.csr_matrix
        The atom connections in the SBU subgraph. Length is # of SBU.
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    depth : int
        The depth of the RACs that are generated. See https://doi.org/10.1021/acs.jpca.7b08750 for more information.
    name : str
        The name of the cif being analyzed.
    cell_v : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    anchoring_atoms : set
        Linker atoms that are bonded to a metal.
    sbupath : str
        Path of the folder to make a csv file in and TXT files containing connection index information.
    linkeranchors_superlist : list of set
        Coordinating atoms of linkers. Number of sets is the number of linkers.
    Gval : bool
        Whether to use G value as RAC.
    connections_list : list of lists of int
        Each inner list is its own separate linker. The ints are the atom indices of that linker. Length is # of linkers.
    connections_subgraphlist : list of scipy.sparse.csr.csr_matrix
        The atom connections in the linker subgraph. Length is # of linkers.

    Returns
    -------
    names : list of str
        The names of the RAC SBU features.
    averaged_SBU_descriptors : numpy.ndarray
        The values of the RAC SBU features, averaged over all SBUs.
    lc_names : list of str
        The names of the RAC lc (ligand coordinating I think) features.
    averaged_lc_descriptors : numpy.ndarray
        The values of the RAC lc features, averaged over all lc atoms.

    """
    descriptor_list = []
    lc_descriptor_list = []
    lc_names = []
    names = []
    descriptor_names = []
    descriptors = []

    sbu_descriptor_path, sbu_descriptors, lc_descriptors = load_sbu_lc_descriptors(sbupath)

    global_connection_indices = []
    for item in linkeranchors_superlist:
        global_connection_indices.extend(list(item))

    """""""""
    Loop over all SBUs as identified by subgraphs. Then create the mol3Ds for each SBU.
    """""""""
    for i, SBU in enumerate(SBUlist):
        descriptor_names = []
        descriptors = []
        SBU_mol = mol3D()
        for val in SBU:
            SBU_mol.addAtom(molcif.getAtom(val))
        SBU_mol.graph = SBU_subgraph[i].todense()

        """""""""
        For each linker connected to the SBU, find the lc atoms for the lc-RACs.
        lc atoms are those bonded to a metal.
        """""""""
        for j, linker in enumerate(connections_list): # Iterating over the different linkers
            descriptor_names = []
            descriptors = []
            if len(set(SBU).intersection(linker))>0:
                #### This means that the SBU and the current linker are connected.
                temp_mol = mol3D()
                link_list = [] # Will hold the lc atoms for the current linker.
                for jj, val2 in enumerate(linker): # linker is a list of global atom indices for the atoms in that linker
                    if val2 in anchoring_atoms:
                        link_list.append(jj)
                    # This builds a mol object for the linker --> even though it is in the SBU section.
                    temp_mol.addAtom(molcif.getAtom(val2))

                temp_mol.graph = connections_subgraphlist[j].todense()
                """""""""
                Generate all of the lc autocorrelations (from the connecting atoms)
                """""""""
                results_dictionary, descriptor_names, descriptors = gen_and_append_desc(temp_mol, link_list, depth, descriptor_names, descriptors, Gval, 'lc')
                """""""""
                If heteroatom functional groups exist (anything that is not C or H, so methyl is missed, also excludes anything lc, so carboxylic metal-coordinating oxygens skipped),
                compile the list of atoms
                """""""""
                functional_atoms = []
                for jj in range(len(temp_mol.graph)):
                    if not jj in link_list: # linker atom is not bonded to a metal
                        if not set({temp_mol.atoms[jj].sym}) & set({"C","H"}): # not a carbon nor a hydrogen. syms get symbols.
                            functional_atoms.append(jj)
                """""""""
                Generate all of the functional group autocorrelations
                """""""""
                if len(functional_atoms) > 0:
                    results_dictionary, descriptor_names, descriptors = gen_and_append_desc(temp_mol, functional_atoms, depth, descriptor_names, descriptors, Gval, 'func')
                else: # There are no functional atoms.
                    if Gval:
                        num_atomic_properties = 7
                    else:
                        num_atomic_properties = 6

                    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,results_dictionary['colnames'],list([np.zeros(int(num_atomic_properties*(depth + 1)))]),'func','all')
                    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,results_dictionary['colnames'],list([np.zeros(int(num_atomic_properties*(depth + 1)))]),'D_func','all')

                for val in descriptors:
                    if not (type(val) == float or isinstance(val, np.float64)):
                        print('Mixed typing. Please convert to python float, and avoid np float')
                        raise AssertionError('Mixed typing creates issues. Please convert your typing.')

                # Some formatting
                descriptor_names += ['name']
                descriptors += [name]
                desc_dict = {key2: descriptors[kk] for kk, key2 in enumerate(descriptor_names)}
                descriptors.remove(name)
                descriptor_names.remove('name')
                lc_descriptors = lc_descriptors.append(desc_dict, ignore_index=True)
                lc_descriptor_list.append(descriptors)
                if j == 0:
                    lc_names = descriptor_names

        averaged_lc_descriptors = np.mean(np.array(lc_descriptor_list), axis=0) # Average the lc RACs over all of the linkers in the MOF.
        # This CSV will be overwritten until the last SBU, but information on all linkers is being kept thanks to the append function
        lc_descriptors.to_csv(sbu_descriptor_path+'/lc_descriptors.csv',index=False)
        descriptors = []
        descriptor_names = []
        SBU_mol_cart_coords=np.array([atom.coords() for atom in SBU_mol.atoms])
        SBU_mol_atom_labels=[atom.sym for atom in SBU_mol.atoms]
        SBU_mol_adj_mat = np.array(SBU_mol.graph)

        ###### WRITE THE SBU MOL TO THE PLACE
        xyz_path = f'{sbupath}/{name}_sbu_{i}.xyz'
        if not os.path.exists(xyz_path):
            SBU_mol_fcoords_connected = XYZ_connected(cell_v, SBU_mol_cart_coords, SBU_mol_adj_mat)
            writeXYZandGraph(xyz_path, SBU_mol_atom_labels, cell_v, SBU_mol_fcoords_connected,SBU_mol_adj_mat)

        # Write TXT file indicating the connecting atoms
        sbu_index_connection_indices = []
        for item in global_connection_indices:
            if item in SBU:
                sbu_index_connection_indices.append(SBU.index(item))
        sbu_index_connection_indices = list(np.sort(sbu_index_connection_indices)) # Sort in ascending order
        sbu_index_connection_indices = [str(item) for item in sbu_index_connection_indices]
        with open(f'{sbupath}/{name}_connection_indices_sbu_{i}.txt', 'w') as f:
            f.write(' '.join(sbu_index_connection_indices))

        """""""""
        Generate all of the SBU based RACs (full scope, mc)
        """""""""
        results_dictionary = generate_full_complex_autocorrelations(SBU_mol,depth=depth,loud=False,flag_name=False,Gval=Gval)
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,results_dictionary['colnames'],results_dictionary['results'],'f','all')
        #### Now starts at every metal on the graph and autocorrelates
        results_dictionary = generate_multimetal_autocorrelations(molcif,depth=depth,loud=False,Gval=Gval)
        descriptor_names, descriptors =  append_descriptors(descriptor_names, descriptors, results_dictionary['colnames'],results_dictionary['results'],'mc','all')
        results_dictionary = generate_multimetal_deltametrics(molcif,depth=depth,loud=False,Gval=Gval)
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,results_dictionary['colnames'],results_dictionary['results'],'D_mc','all')

        # Some formatting
        descriptor_names += ['name']
        descriptors += [name]
        descriptors == list(descriptors)
        desc_dict = {key: descriptors[ii] for ii, key in enumerate(descriptor_names)}
        descriptors.remove(name)
        descriptor_names.remove('name')
        sbu_descriptors = sbu_descriptors.append(desc_dict, ignore_index=True)
        descriptor_list.append(descriptors)
        if i == 0:
            names = descriptor_names

    sbu_descriptors.to_csv(sbu_descriptor_path+'/sbu_descriptors.csv',index=False)
    averaged_SBU_descriptors = np.mean(np.array(descriptor_list), axis=0) # Average the SBU RACs over all of the SBUs in the MOF.
    return names, averaged_SBU_descriptors, lc_names, averaged_lc_descriptors

def make_MOF_linker_RACs(linkerlist, linker_subgraphlist, molcif, depth, name, cell_v, linkerpath, linkeranchors_superlist, Gval):
    """
    Generate RACs on the linkers of the MOF.

    Parameters
    ----------
    linkerlist : list of lists of int
        Each inner list is its own separate linker. The ints are the atom indices of that linker. Length is # of linkers.
    linker_subgraphlist : list of scipy.sparse.csr.csr_matrix
        The atom connections in the linker subgraph. Length is # of linkers.
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    depth : int
        The depth of the RACs that are generated. See https://doi.org/10.1021/acs.jpca.7b08750 for more information.
    name : str
        The name of the cif being analyzed.
    cell_v : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    linkerpath : str
        Path of the folder to make a csv file in and TXT files containing connection index information.
    linkeranchors_superlist : list of set
        Coordinating atoms of linkers. Number of sets is the number of linkers.
    Gval : bool
        Whether to use G value as RAC.

    Returns
    -------
    colnames : list of str
        The names of the RAC linker features.
    averaged_ligand_descriptors : numpy.ndarray
        The values of the RAC linker features, averaged over all linkers.

    """

    #### This function makes full scope linker RACs for MOFs ####
    descriptor_list = []
    descriptors = []
    if linkerpath:
        linker_descriptor_path = os.path.dirname(linkerpath)
        if os.path.getsize(linker_descriptor_path+'/linker_descriptors.csv')>0: # Checking if there is a file there.
            linker_descriptors = pd.read_csv(linker_descriptor_path+'/linker_descriptors.csv')
        else:
            linker_descriptors = pd.DataFrame()

    global_connection_indices = []
    for item in linkeranchors_superlist:
        global_connection_indices.extend(list(item))

    for i, linker in enumerate(linkerlist): # Iterating through the linkers.
        # Preparing a mol3D object for the current linker.
        linker_mol = mol3D()
        for val in linker:
            linker_mol.addAtom(molcif.getAtom(val))
        linker_mol.graph = linker_subgraphlist[i].todense()

        linker_mol_cart_coords=np.array([atom.coords() for atom in linker_mol.atoms])
        linker_mol_atom_labels=[atom.sym for atom in linker_mol.atoms]
        linker_mol_adj_mat = np.array(linker_mol.graph)

        ###### WRITE THE LINKER MOL TO THE PLACE
        xyz_path = f'{linkerpath}/{name}_linker_{i}.xyz'
        if not os.path.exists(xyz_path):
            linker_mol_fcoords_connected = XYZ_connected(cell_v, linker_mol_cart_coords, linker_mol_adj_mat)
            writeXYZandGraph(xyz_path, linker_mol_atom_labels, cell_v, linker_mol_fcoords_connected, linker_mol_adj_mat)

        # Write TXT file indicating the connecting atoms
        linker_index_connection_indices = []
        for item in global_connection_indices:
            if item in linker:
                linker_index_connection_indices.append(linker.index(item))
        linker_index_connection_indices = list(np.sort(linker_index_connection_indices)) # Sort in ascending order
        linker_index_connection_indices = [str(item) for item in linker_index_connection_indices]
        with open(f'{linkerpath}/{name}_connection_indices_linker_{i}.txt', 'w') as f:
            f.write(' '.join(linker_index_connection_indices))

        allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
        labels_strings = ['chi', 'Z', 'I', 'T', 'S']
        if Gval:
            allowed_strings.append('group_number')
            labels_strings.append('Gval')
        colnames = []
        lig_full = list()

        """""""""
        Generate all of the linker based RACs
        """""""""
        for ii, properties in enumerate(allowed_strings):
            if not list(descriptors): # This is the case when just starting and the list is empty.
                ligand_ac_full = full_autocorrelation(linker_mol, properties, depth) # RACs
            else:
                ligand_ac_full += full_autocorrelation(linker_mol, properties, depth)
            this_colnames = []
            for j in range(0,depth+1):
                this_colnames.append('f-lig-'+labels_strings[ii] + '-' + str(j))
            colnames.append(this_colnames)
            lig_full.append(ligand_ac_full)

        # Some formatting
        lig_full = [item for sublist in lig_full for item in sublist] # flatten lists
        colnames = [item for sublist in colnames for item in sublist]
        colnames += ['name']
        lig_full += [name]
        desc_dict = {key: lig_full[i] for i, key in enumerate(colnames)}
        linker_descriptors = linker_descriptors.append(desc_dict, ignore_index=True)
        lig_full.remove(name)
        colnames.remove('name')
        descriptor_list.append(lig_full)

    #### We dump the standard lc descriptors without averaging or summing so that the user
    #### can make the modifications that they want. By default we take the average ones.
    linker_descriptors.to_csv(linker_descriptor_path+'/linker_descriptors.csv', index=False)
    averaged_ligand_descriptors = np.mean(np.array(descriptor_list), axis=0)
    return colnames, averaged_ligand_descriptors

def mkdir_if_absent(folder_paths):
    """
    Makes a folder at each path in folder_path if it does not yet exist.

    Parameters
    ----------
    folder_path : list of str
        The folder paths to check, and potentially at which to make a folder.

    Returns
    -------
    None

    """

    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

def make_file_if_absent(path, filenames):
    """
    Makes the specified files if they do not yet exist.

    Parameters
    ----------
    path : str
        The base path.
    filenames : list of str
        The file names.

    Returns
    -------
    None

    """

    for filename in filenames:
        if not os.path.exists(f'{path}/{filename}'):
            with open(f'{path}/{filename}','w') as f:
                f.close()

def path_maker(path):
    """
    Makes the required folders and files.

    Parameters
    ----------
    path : str
        The path to which output will be written.

    Returns
    -------
    None

    """
    # Making the required folders
    required_folders = [f'{path}/ligands', f'{path}/linkers', f'{path}/sbus', f'{path}/xyz', f'{path}/logs']
    mkdir_if_absent(required_folders)

    # Making the required files
    required_files = ['sbu_descriptors.csv', 'linker_descriptors.csv', 'lc_descriptors.csv']
    make_file_if_absent(path, required_files)

def failure_response(path, failure_str):
    """
    Writes to the log file about the encountered failure.

    Parameters
    ----------
    path : str
        The path to which output will be written.

    Returns
    -------
    Two zero lists to indicate failure.

    """
    full_names = [0]
    full_descriptors = [0]
    write2file(path,"/FailedStructures.log",failure_str)
    return full_names, full_descriptors

def bond_information_write(linker_list, linkeranchors_superlist, adj_matrix, molcif, cell_v, path):
    """
    Attains and writes bond information about the bonds between SBUs and linkers.

    Parameters
    ----------
    linker_list : list of lists of ints
        Each inner list is its own separate linker. The ints are the atom indices of that linker. Length is # of linkers.
    linkeranchors_superlist : list of set
        Coordinating atoms of linkers. Number of sets is the number of linkers.
    adj_matrix : scipy.sparse.csr.csr_matrix
        1 represents a bond, 0 represents no bond. Shape is (number of atoms, number of atoms).
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    cell_v : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    path : str
        The parent path to which output will be written.

    Returns
    -------
    None

    """

    bond_length_list = []
    scaled_bond_length_list = []
    for linker_idx, linker_atoms_list in enumerate(linker_list): # Iterate over all linkers
        # Getting the connection points of the linker
        for anchor_super_idx in range(len(linkeranchors_superlist)):
            if list(linkeranchors_superlist[anchor_super_idx])[0] in linker_atoms_list: # Any anchor index in the current entry of linkeranchors_superlist is in the current linker's indices
                linker_connection_points = list(linkeranchors_superlist[anchor_super_idx]) # Indices of the connection points in the linker
        for con_point in linker_connection_points:
            connected_atoms = adj_matrix.todense()[con_point,:]
            connected_atoms = np.ravel(connected_atoms)

            connected_atoms = np.nonzero(connected_atoms)[0] # The indices of atoms connected to atom with index con_point.

            for con_atom in connected_atoms:
                con_atom3D = molcif.getAtom(con_atom) # atom3D of an atom connected to the connection point
                con_point3D = molcif.getAtom(con_point) # atom3D of the connection point on the linker
                # Check if the atom is a metal
                if con_atom3D.ismetal(transition_metals_only=False):
                    # Finding the optimal unit cell shift
                    molcif_cart_coords = np.array([atom.coords() for atom in molcif.atoms])
                    fcoords=frac_coord(molcif_cart_coords,cell_v) # fractional coordinates
                    fcoords[con_atom]+=compute_image_flag(cell_v,fcoords[con_point],fcoords[con_atom]) # Shifting the connected metal to get it close to the connection point
                    ccoords = fractional2cart(fcoords, cell_v)
                    shifted_con_atom3D = atom3D(Sym=con_atom3D.symbol(), xyz=list(ccoords[con_atom,:]))

                    bond_len = shifted_con_atom3D.distance(con_point3D) # Bond length between the connected metal and the connection point.
                    con_atom_radius = COVALENT_RADII[shifted_con_atom3D.symbol()]
                    con_point_radius = COVALENT_RADII[con_point3D.symbol()]
                    relative_bond_len = bond_len / (con_atom_radius + con_point_radius)

                    bond_length_list.append(bond_len)
                    scaled_bond_length_list.append(relative_bond_len)

    mean_bond_len = np.mean(bond_length_list) # Average over all SBU-linker atom to atom connections
    mean_scaled_bond_len = np.mean(scaled_bond_length_list) # Average over all SBU-linker atom to atom connections
    std_bond_len = np.std(bond_length_list)
    std_scaled_bond_len = np.std(scaled_bond_length_list)

    with open(f"{path}/sbu_linker_bondlengths.txt", "w") as f:
        f.write(f'Mean bond length: {mean_bond_len}\n')
        f.write(f'Mean scaled bond length: {mean_scaled_bond_len}\n')
        f.write(f'Stdev bond length: {std_bond_len}\n')
        f.write(f'Stdev scaled bond length: {std_scaled_bond_len}')

def surrounded_sbu_gen(SBU_list, linker_list, sbupath, molcif, adj_matrix, cell_v, allatomtypes, name):
    """
    Writes XYZ files for all SBUs provided, with each SBU surrounded by all linkers coordinated to it.

    Parameters
    ----------
    SBU_list : list of lists of ints
        Each inner list is its own separate SBU. The ints are the atom indices of that SBU. Length is # of SBUs.
    linker_list : list of lists of ints
        Each inner list is its own separate linker. The ints are the atom indices of that linker. Length is # of linkers.
    sbupath : str
        The path to which SBU information is written.
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    adj_matrix : scipy.sparse.csr.csr_matrix
        1 represents a bond, 0 represents no bond. Shape is (number of atoms, number of atoms).
    cell_v : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    allatomtypes : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.
    name : str
        The name of the cif being analyzed.

    Returns
    -------
    None

    """

    for SBU_idx, atoms_sbu in enumerate(SBU_list):
        # atoms_sbu are the indices of atoms in the SBU with index SBU_idx

        connection_atoms = [] # List of lists of the coordinating atoms of each of the connected linkers. Length is # of connected linkers.
        atoms_connected_linkers = [] # List of lists of the atoms of each of the connected linkers. Length is # of connected linkers.
        for atoms_linker in linker_list:
            atoms_in_common = list(set(atoms_sbu).intersection(set(atoms_linker)))
            if len(atoms_in_common) != 0:
                connection_atoms.append(atoms_in_common)
                atoms_connected_linkers.append(atoms_linker)

        # Generating an XYZ of the SBU surrounded by linkers.
        xyz_path = f'{sbupath}/{name}_sbu_{SBU_idx}_with_linkers.xyz'
        # For each atom index in an inner list in connection_atoms, build out the corresponding linker (inner list) in atoms_connected_linkers.

        ### Start with the atoms of the SBU
        surrounded_sbu = mol3D() # SBU surrounded by linkers
        starting_atom_idx = atoms_sbu[0]
        added_idx = [starting_atom_idx] # This list will contain the SBU indices that no longer need to be considered for branching.
        starting_atom3D = molcif.getAtom(starting_atom_idx)
        # The mol3D object starts out with a single atom. Atoms will be added branching out from this initial atom.
        surrounded_sbu.addAtom(starting_atom3D)
        atom3D_dict = {starting_atom_idx: starting_atom3D} # atom3D objects of the SBU

        dense_adj_mat = np.array(adj_matrix.todense())

        # Dictionary. Keys are ints (indices of atoms), values are lists of indices of atoms
        # The key is the atom relative to which the new atom must be positioned.
        atoms_connected_to_start = list(np.nonzero(dense_adj_mat[starting_atom_idx])[0])
        atoms_connected_to_start = [i for i in atoms_connected_to_start if i in atoms_sbu]
        sbu_atoms_to_branch_from = {starting_atom_idx: atoms_connected_to_start}
        sbu_atoms_to_branch_from_keys = [starting_atom_idx]


        while len(sbu_atoms_to_branch_from_keys) != 0:

            # Take from the first key, i.e. [0], of the dictionary
            my_key = sbu_atoms_to_branch_from_keys[0]
            neighbor_idx = sbu_atoms_to_branch_from[my_key][0] # Grabbing a neighbor to calculate its position.

            # Remove that index from further consideration
            sbu_atoms_to_branch_from[my_key].remove(neighbor_idx)

            # If the list associated with a key is now empty, remove the key.
            if len(sbu_atoms_to_branch_from[my_key]) == 0:
                sbu_atoms_to_branch_from_keys.remove(my_key) # sbu_atoms_to_branch_from_keys = [i for i in sbu_atoms_to_branch_from_keys if i != my_key]
                sbu_atoms_to_branch_from.pop(my_key)

            if neighbor_idx in added_idx:
                continue # Skip this index if it has already been added

            # Getting the optimal position of the neighbor, relative to my_key
            fcoords_my_key = frac_coord(atom3D_dict[my_key].coords(), cell_v)

            fcoords_neighbor_initial = frac_coord(molcif.getAtom(neighbor_idx).coords(), cell_v)
            fcoords_neighbor = fcoords_neighbor_initial + compute_image_flag(cell_v, fcoords_my_key, fcoords_neighbor_initial)
            coords_neighbor = fractional2cart(fcoords_neighbor, cell_v)
            symbol_neighbor = allatomtypes[neighbor_idx] # Element
            new_atom3D = atom3D(Sym=symbol_neighbor, xyz=coords_neighbor)
            surrounded_sbu.addAtom(new_atom3D)

            atom3D_dict[neighbor_idx] = new_atom3D
            added_idx.append(neighbor_idx)
            atoms_connected_to_neighbor = list(np.nonzero(dense_adj_mat[neighbor_idx])[0])
            atoms_connected_to_neighbor_to_check = [i for i in atoms_connected_to_neighbor if i not in added_idx and i in atoms_sbu]
            if len(atoms_connected_to_neighbor_to_check) > 0:
                sbu_atoms_to_branch_from[neighbor_idx] = atoms_connected_to_neighbor_to_check
                sbu_atoms_to_branch_from_keys.append(neighbor_idx)

        # At this point in the code, all SBU atoms have been added to the mol3D object surrounded_sbu.

        ### Next, add each of the linkers.
        # Using atom3D_dict, connection_atoms, and atoms_connected_linkers.
        for linker_idx in range(len(connection_atoms)):
            # For each linker, build out the linker in surrounded_sbu object.

            linker_indices = atoms_connected_linkers[linker_idx]

            for starting_atom_idx in connection_atoms[linker_idx]:
                atom3D_dict_copy = atom3D_dict.copy()
                added_idx = [starting_atom_idx]
                starting_atom3D = atom3D_dict_copy[starting_atom_idx] # Position of the starting atom in the SBU, which has been built by this point in the code.

                atoms_connected_to_start = list(np.nonzero(dense_adj_mat[starting_atom_idx])[0])
                atoms_connected_to_start = [i for i in atoms_connected_to_start if i in linker_indices]
                linker_atoms_to_branch_from = {starting_atom_idx: atoms_connected_to_start}
                linker_atoms_to_branch_from_keys = [starting_atom_idx]

                while len(linker_atoms_to_branch_from_keys) != 0:

                    # Take from the first key, i.e. [0], of the dictionary
                    my_key = linker_atoms_to_branch_from_keys[0]
                    neighbor_idx = linker_atoms_to_branch_from[my_key][0] # Grabbing a neighbor to calculate its position.

                    # Remove that index from further consideration
                    linker_atoms_to_branch_from[my_key].remove(neighbor_idx)

                    # If the list associated with a key is now empty, remove the key.
                    if len(linker_atoms_to_branch_from[my_key]) == 0:
                        linker_atoms_to_branch_from_keys.remove(my_key)
                        linker_atoms_to_branch_from.pop(my_key)

                    if neighbor_idx in added_idx:
                        continue # Skip this index if it has already been added

                    # Getting the optimal position of the neighbor, relative to my_key
                    fcoords_my_key = frac_coord(atom3D_dict_copy[my_key].coords(), cell_v)

                    fcoords_neighbor_initial = frac_coord(molcif.getAtom(neighbor_idx).coords(), cell_v)
                    fcoords_neighbor = fcoords_neighbor_initial + compute_image_flag(cell_v, fcoords_my_key, fcoords_neighbor_initial)
                    coords_neighbor = fractional2cart(fcoords_neighbor, cell_v)
                    symbol_neighbor = allatomtypes[neighbor_idx] # Element
                    new_atom3D = atom3D(Sym=symbol_neighbor, xyz=coords_neighbor)

                    # Only add the new atom if it does not overlap with an atom that is already in surrounded sbu.
                    # If there is overlap, then the atom was already added in the SBU.
                    min_dist = 100 # Starting from a big number that will be replaced in the subsequent lines.
                    num_atoms = surrounded_sbu.getNumAtoms()
                    for i in range(num_atoms):
                        pair_dist = new_atom3D.distance(surrounded_sbu.getAtom(i))
                        if pair_dist < min_dist:
                            min_dist = pair_dist
                    if min_dist > 0.1:
                        surrounded_sbu.addAtom(new_atom3D)

                    atom3D_dict_copy[neighbor_idx] = new_atom3D
                    added_idx.append(neighbor_idx)
                    atoms_connected_to_neighbor = list(np.nonzero(dense_adj_mat[neighbor_idx])[0])
                    atoms_connected_to_neighbor_to_check = [i for i in atoms_connected_to_neighbor if i not in added_idx and i in linker_indices]
                    if len(atoms_connected_to_neighbor_to_check) > 0:
                        linker_atoms_to_branch_from[neighbor_idx] = atoms_connected_to_neighbor_to_check
                        linker_atoms_to_branch_from_keys.append(neighbor_idx)


        surrounded_sbu.writexyz(xyz_path)

def dist_mat_comp(X):
    """
    Computes the pairwise distances between the rows of the coordinate information X.

    Parameters
    ----------
    X : numpy.ndarray
        Cartesian coordinate information for atoms. Shape is (number of atoms, 3).

    Returns
    -------
    dist_mat : numpy.ndarray
        Pairwise distances between all atoms. Shape is (number of atoms, number of atoms).

    """

    # Assumes X is an np array of shape (number of atoms, 3). The Cartesian coordinates of atoms.
    # Does not do any unit cell shifts
    # Makes use of numpy vectorization to speed things up versus a for loop approach.
    X1 = np.expand_dims(X, axis=1)
    X2 = np.expand_dims(X, axis=0)
    Z = X1 - X2
    dist_mat = np.sqrt(np.sum(np.square(Z), axis=-1)) # The pairwise distance matrix. Distances between all atoms.
    return dist_mat

def detect_1D_rod(SBU_list, molcif, allatomtypes, cell_v, logpath, name):
    """
    Writes to the log file if the MOF is likely to contain a 1D rod.

    Parameters
    ----------
    SBU_list : list of lists of ints
        Each inner list is its own separate SBU. The ints are the atom indices of that SBU. Length is # of SBUs.
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    allatomtypes : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.
    cell_v : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    logpath : str
        The path to which log files are written.
    name : str
        The name of the cif being analyzed.

    Returns
    -------
    None

    """

    sbu_atom_indices = []
    for i in SBU_list:
        # i is the indices in a given SBU
        sbu_atom_indices.extend(i)
    sbu_atom_indices.sort()

    # allatomtypes_sbus_initial = [i for idx, i in enumerate(allatomtypes) if idx in sbu_atom_indices]
    cart_coords_sbus_initial = [molcif.getAtom(i).coords() for i in sbu_atom_indices]
    allatomtypes_sbus_with_shifts = [] # Will contain the symbols of all SBU atoms, across the 8 unit cell shifts
    cart_coords_sbus_with_shifts = [] # Will contain the cartesian coordinates of all SBU atoms, across the 8 unit cell shifts

    # Applying all possible unit cell shifts in 0, 1, for all SBU atoms
    for idx, i in enumerate(sbu_atom_indices):
        supercells = np.array(list(itertools.product((0, 1), repeat=3)))
        fractional_coords = frac_coord(cart_coords_sbus_initial[idx], cell_v)
        fractional_coords_shifts = fractional_coords + supercells # 8 versions of fractional_coords, shifted some cells over in different directions
        for j in fractional_coords_shifts:
            allatomtypes_sbus_with_shifts.append(allatomtypes[i])
            cart_coords_sbus_with_shifts.append(fractional2cart(j, cell_v))

    cart_coords_sbus_with_shifts = np.array(cart_coords_sbus_with_shifts) # Converting nested list to a numpy array

    distance_mat = dist_mat_comp(cart_coords_sbus_with_shifts)
    adj_matrix, _ = compute_adj_matrix(distance_mat, allatomtypes_sbus_with_shifts, handle_overlap=True) # Ignoring overlap

    # For each connected component, see how long it is
    adj_matrix = sparse.csr_matrix(adj_matrix)
    n_components, labels_components = sparse.csgraph.connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

    # What is the shortest cell vector?
    min_vec_len = np.min(np.linalg.norm(cell_v, axis=1)) # Equivalent to min(cpar[:3])

    is_1d_rod = False
    for i in range(n_components):
        component_indices = np.where(labels_components == i)[0]
        component_cart_coords = cart_coords_sbus_with_shifts[component_indices]

        pairwise_atom_distances = dist_mat_comp(component_cart_coords)
        if np.max(pairwise_atom_distances) > min_vec_len:
            # In this case, an SBU likely stretches out longer than a unit cell
            is_1d_rod = True
            break

    if is_1d_rod:
        print(f'Likely 1D rod')
        tmpstr = "MOF SBU is likely a 1D rod"
        write2file(logpath,"/%s.log"%name,tmpstr)

def get_MOF_descriptors(data, depth, path=False, xyzpath=False, graph_provided=False, wiggle_room=1,
    max_num_atoms=2000, get_sbu_linker_bond_info=False, surrounded_sbu_file_generation=False, detect_1D_rod_sbu=False,
    Gval=False):
    """
    Generates RAC descriptors on a MOF, assuming it has P1 symmetry.
    Writes three files: sbu_descriptors.csv, linker_descriptors.csv, and lc_descriptors.csv
    These files contain the RAC descriptors of the MOF.

    Parameters
    ----------
    data : str
        The path to the cif file for which descriptors are generated.
    depth : int
        The depth of the RACs that are generated. See https://doi.org/10.1021/acs.jpca.7b08750 for more information.
    path : str
        The parent path to which output will be written.
        This output includes three csv files.
        This output also includes three folders called ligands, linkers, and sbus.
            These folders contain net and xyz files of the components of the MOF.
        This output also includes a folder called logs.
        This output also includes the xyz and net files for the cif being analyzed, written with the function writeXYZandGraph.
    xyzpath : str
        The path to where the xyz of the MOF structure will be written.
    graph_provided : bool
        Whether or not the cif file has graph information of the structure (i.e. what atoms are bonded to what atoms).
        If not, computes the N^2 pairwise distance matrix, which is expensive.
    wiggle_room : float
        A multiplier that allows for more or less strict bond distance cutoffs.
    max_num_atoms : int
        The maximum number of atoms in the unit cell for which analysis is conducted.
    get_sbu_linker_bond_info : bool
        Whether or not a TXT file is written with information on the bonds between SBUs and linkers.
    surrounded_sbu_file_generation : bool
        Whether or not an XYZ file for each SBU, surrounded by its connected linkers, will be generated.
    detect_1D_rod_sbu : bool
        Whether or not to check if SBU is a 1D rod.
    Gval : bool
        Whether to use G value as RAC, by default False.

    Returns
    -------
    full_names : list of str
        The names of the RAC features.
    full_descriptors : list of numpy.float64
        The values of the RAC features.

    """

    if not path: # Throw an error if the user did not supply a path to which to write the output.
        print('Need a directory to place all of the linker, SBU, and ligand objects. Exiting now.')
        raise ValueError('Base path must be specified in order to write descriptors.')
    else:
        if path.endswith('/'):
            path = path[:-1]

        # Making the required folders and files.
        path_maker(path)

    ligandpath = path+'/ligands'
    linkerpath = path+'/linkers'
    sbupath = path+'/sbus'
    logpath = path+"/logs"

    """""""""
    Input cif file and get the cell parameters and adjacency matrix. If too large or has overlap, do not featurize.
    Simultaneously prepare mol3D class for MOF for future RAC featurization (molcif).
    """""""""
    cpar, allatomtypes, fcoords = readcif(data)
    cell_v = mkcell(cpar)
    cart_coords = fractional2cart(fcoords, cell_v)
    name = os.path.basename(data).replace(".cif", "")
    if len(cart_coords) > max_num_atoms: # Don't deal with large cifs because of computational resources required for their treatment.
        print("cif file is too large, skipping it for now...")
        failure_str = f"Failed to featurize {name}: large primitive cell\n {len(cart_coords)} atoms"
        full_names, full_descriptors = failure_response(path, failure_str)
        return full_names, full_descriptors

    """""""""
    Getting the adjacency matrix.
    """""""""
    if not graph_provided: # Make the adjacency matrix.
        distance_mat = compute_distance_matrix3(cell_v, cart_coords)
        try:
            adj_matrix, _ = compute_adj_matrix(distance_mat, allatomtypes, wiggle_room)
        except NotImplementedError:
            failure_str = f"Failed to featurize {name}: atomic overlap\n"
            full_names, full_descriptors = failure_response(path, failure_str)
            return full_names, full_descriptors
    else: # Grab the adjacency matrix from the cif file.
        adj_matrix_list = []
        max_sofar = 0
        with open(data.replace('primitive','cif'),'r') as f:
            readdata = f.readlines()
            flag = False
            for i, row in enumerate(readdata):
                if '_ccdc_geom_bond_type' in row:
                    flag = True
                    continue
                if flag:
                    splitrow = row.split()
                    atom1 = int(re.findall(r'\d+',splitrow[0])[0])
                    atom2 = int(re.findall(r'\d+',splitrow[1])[0])
                    max_sofar = max(atom1, max_sofar)
                    max_sofar = max(atom2, max_sofar)
                    adj_matrix_list.append((atom1,atom2))
        adj_matrix = np.zeros((max_sofar+1,max_sofar+1)) # 0 indicates the absence of a bond.
        for i, row in enumerate(adj_matrix_list):
            adj_matrix[row[0],row[1]] = 1 # 1 is indicative of a bond.
            adj_matrix[row[1],row[0]] = 1
    adj_matrix = sparse.csr_matrix(adj_matrix)

    writeXYZandGraph(xyzpath, allatomtypes, cell_v, fcoords, adj_matrix.todense())
    molcif,_,_,_,_ = import_from_cif(data, True) # molcif is a mol3D class of a single unit cell (or the cell of the cif file)
    molcif.graph = adj_matrix.todense()

    """""""""
    check number of connected components.
    if more than 1: it checks if the structure is interpenetrated. Fails if no metal in one of the connected components (identified by the graph).
    This includes floating solvent molecules.
    """""""""

    n_components, labels_components = sparse.csgraph.connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    metal_list = set([at for at in molcif.findMetal(transition_metals_only=False)]) # the atom indices of the metals
    # print('##### METAL LIST', metal_list, [molcif.getAtom(val).symbol() for val in list(metal_list)])
    # print('##### METAL LIST', metal_list, [val.symbol() for val in molcif.atoms])
    if not len(metal_list) > 0:
        failure_str = f"Failed to featurize {name}: no metal found\n"
        full_names, full_descriptors = failure_response(path, failure_str)
        return full_names, full_descriptors
    for comp in range(n_components):
        inds_in_comp = [i for i in range(len(labels_components)) if labels_components[i]==comp]
        if not set(inds_in_comp) & metal_list: # In the context of sets, & is the intersection. If the intersection is null, the (&) expression is False; the `not` would then make it True.
            # If this if statement is entered, there is an entire connected component that has no metals in it. No connections to any metal.
            failure_str = f"Failed to featurize {name}: solvent molecules\n"
            full_names, full_descriptors = failure_response(path, failure_str)
            return full_names, full_descriptors

    if n_components > 1 : # There are multiple connected components that have a metal in them in this case.
        print("structure is interpenetrated")
        tmpstr = "%s found to be an interpenetrated structure\n"%(name)
        write2file(logpath,"/%s.log"%name,tmpstr)

    """""""""
    step 1: metallic part
        removelist = metals (1) + atoms only connected to metals (2) + H connected to (1+2)
            Actually, it looks like only (1) and (2). Not H connected to (1+2)
        SBUlist = removelist + 1st coordination shell of the metals
    removelist = set()
    Logs the atom types of the connecting atoms to the metal in logpath.
    """""""""
    SBUlist = set()
    metal_list = set([at for at in molcif.findMetal(transition_metals_only=False)]) # the atom indices of the metals
    # print('##### METAL LIST2', metal_list, [molcif.getAtom(val).symbol() for val in list(metal_list)])
    # print('##### all LIST2', metal_list, [val.symbol() for val in molcif.atoms])
    [SBUlist.update(set([metal])) for metal in molcif.findMetal(transition_metals_only=False)] # Consider all metals as part of the SBUs.
    [SBUlist.update(set(molcif.getBondedAtomsSmart(metal))) for metal in molcif.findMetal(transition_metals_only=False)] # atoms connected to metals.
    removelist = set()
    [removelist.update(set([metal])) for metal in molcif.findMetal(transition_metals_only=False)] # Remove all metals as part of the SBUs.
    for metal in removelist:
        bonded_atoms = set(molcif.getBondedAtomsSmart(metal))
        bonded_atoms_types = set([str(allatomtypes[at]) for at in set(molcif.getBondedAtomsSmart(metal))]) # The types of elements bonded to metals. E.g. oxygen, carbon, etc.
        cn = len(bonded_atoms)
        cn_atom = ",".join([at for at in bonded_atoms_types])
        tmpstr = "atom %i with type of %s found to have %i coordinates with atom types of %s\n"%(metal,allatomtypes[metal],cn,cn_atom)
        write2file(logpath,"/%s.log"%name,tmpstr)
    [removelist.update(set([atom])) for atom in SBUlist if all((molcif.getAtom(val).ismetal() or molcif.getAtom(val).symbol().upper() == 'H') for val in molcif.getBondedAtomsSmart(atom))]
    """""""""
    adding hydrogens connected to atoms which are only connected to metals. In particular interstitial OH, like in UiO SBU.
    """""""""
    for atom in SBUlist:
        for val in molcif.getBondedAtomsSmart(atom):
            if molcif.getAtom(val).symbol().upper() == 'H':
               removelist.update(set([val]))

    """""""""
    At this point:
    The remove list only removes metals and things ONLY connected to metals or hydrogens.
    Thus the coordinating atoms are double counted in the linker.

    step 2: organic part
        removelist = linkers are all atoms - the removelist (assuming no bond between
        organiclinkers)
    """""""""
    allatoms = set(range(0, adj_matrix.shape[0])) # The indices of all the atoms.
    linkers = allatoms - removelist
    linker_list, linker_subgraphlist = get_closed_subgraph(linkers.copy(), removelist.copy(), adj_matrix)
    connections_list = copy.deepcopy(linker_list)
    connections_subgraphlist = copy.deepcopy(linker_subgraphlist)
    linker_length_list = [len(linker_val) for linker_val in linker_list] # The number of atoms in each linker.
    """""""""
    find all anchoring atoms on linkers and ligands (lc identification)
        The atoms that are bonded to a metal.
    """""""""
    anc_atoms = set()
    for linker in linker_list: # Checking all of the linkers one by one.
        for atom_linker in linker: # Checking each atom in the current linker.
            bonded2atom = np.nonzero(adj_matrix[atom_linker,:])[1] # indices of atoms with bonds to the atom with the index atom_linker
            if set(bonded2atom) & metal_list: # This means one of the atoms bonded to the atom with the index atom_linker is a metal.
                anc_atoms.add(atom_linker)
    """""""""
    step 3: determine whether linker or ligand
    checking to find the anchors and #SBUs that are connected to an organic part
    anchor <= 1 -> ligand
    anchor > 1 and #SBU > 1 -> linker
    else: walk over the linker graph and count #crossing PBC
        if #crossing is odd -> linker
        else -> ligand
    """""""""
    initial_SBU_list, initial_SBU_subgraphlist = get_closed_subgraph(removelist.copy(), linkers.copy(), adj_matrix)
    templist = linker_list.copy()
    long_ligands = False
    max_min_linker_length , min_max_linker_length = (0,100) # The maximum value of the minimum linker length, and the minimum value of the maximum linker length. Updated later.
    linkeranchors_superlist = [] # Will contain the indices of the linker atoms that coordinate to metals
    for ii, atoms_list in reversed(list(enumerate(linker_list))): # Loop over all linker subgraphs
        linkeranchors_list = set()
        linkeranchors_atoms = set()
        sbuanchors_list = set()
        sbu_connect_list = set() # Will contain the indices of SBU subgraphs that have a connection to the current linker described by atoms_list
        """""""""
        Here, we are trying to identify what is actually a linker and what is a ligand.
        To do this, we check if something is connected to more than one SBU. Set to
        handle cases where primitive cell is small, ambiguous cases are recorded.
        """""""""
        for iii,atoms in enumerate(atoms_list): # loop over all atom indices in a linker
            connected_atoms = np.nonzero(adj_matrix[atoms,:])[1] # indices of atoms with bonds to the atom with the index atoms
            for kk, sbu_atoms_list in enumerate(initial_SBU_list): # loop over all SBU subgraphs
                for sbu_atoms in sbu_atoms_list: # Loop over SBU
                    if sbu_atoms in connected_atoms: # found an SBU atom bonded to an atom in the linker defined by atoms_list
                        linkeranchors_list.add(iii)
                        linkeranchors_atoms.add(atoms)
                        sbuanchors_list.add(sbu_atoms)
                        sbu_connect_list.add(kk) #Add if unique SBUs
        min_length,max_length = linker_length(linker_subgraphlist[ii].todense(), linkeranchors_list)
        linkeranchors_superlist.append(linkeranchors_atoms)

        if len(linkeranchors_list) >= 2 : # linker, and in one ambiguous case, could be a ligand.
            if len(sbu_connect_list) >= 2: # Something that connects two SBUs is certain to be a linker
                max_min_linker_length = max(min_length,max_min_linker_length)
                min_max_linker_length = min(max_length,min_max_linker_length)
                continue
            else:
                # check number of times we cross PBC :
                # TODO: we still can fail in multidentate ligands!
                linker_cart_coords = np.array([
                    at.coords() for at in [molcif.getAtom(val) for val in atoms_list]]) # Cartesian coordinates of the atoms in the linker
                linker_adjmat = np.array(linker_subgraphlist[ii].todense())
                pr_image_organic = ligand_detect(cell_v,linker_cart_coords,linker_adjmat,linkeranchors_list) # Periodic images for the organic component
                sbu_temp = linkeranchors_atoms.copy()
                sbu_temp.update({val for val in initial_SBU_list[list(sbu_connect_list)[0]]}) # Adding atoms. Not sure why the [0] is there? TODO
                sbu_temp = list(sbu_temp)
                sbu_cart_coords = np.array([
                    at.coords() for at in [molcif.getAtom(val) for val in sbu_temp]])
                sbu_adjmat = slice_mat(adj_matrix.todense(),sbu_temp)
                pr_image_sbu = ligand_detect(cell_v,sbu_cart_coords,sbu_adjmat,set(range(len(linkeranchors_list)))) # Periodic images for the SBU
                if not (len(np.unique(pr_image_sbu, axis=0))==1 and len(np.unique(pr_image_organic, axis=0))==1): # linker. More than one periodic image for sbu or organic component
                    max_min_linker_length = max(min_length,max_min_linker_length)
                    min_max_linker_length = min(max_length,min_max_linker_length)
                    tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
                            +','+' SBU connectlist: '+str(sbu_connect_list)+' set to be linker\n'
                    write2file(ligandpath,"/ambiguous.txt",tmpstr)
                    continue
                else: #  all anchoring atoms are in the same unitcell -> ligand
                    removelist.update(set(templist[ii])) # we also want to remove these ligands
                    SBUlist.update(set(templist[ii])) # we also want to remove these SBUs
                    linker_list.pop(ii)
                    linker_subgraphlist.pop(ii)
                    tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
                            +','+' SBU connectlist: '+str(sbu_connect_list)+' set to be ligand\n'
                    write2file(ligandpath,"/ambiguous.txt",tmpstr)
                    tmpstr = str(name)+str(ii)+','+' Anchors list: '+ \
                            str(sbuanchors_list)+','+' SBU connectlist: '+str(sbu_connect_list)+'\n'
                    write2file(ligandpath,"/ligand.txt",tmpstr)
        else: # definite ligand
            write2file(logpath,"/%s.log"%name,"found ligand\n")
            removelist.update(set(templist[ii])) # we also want to remove these ligands
            SBUlist.update(set(templist[ii])) # we also want to remove these ligands
            linker_list.pop(ii)
            linker_subgraphlist.pop(ii)
            tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
         +','+' SBU connectlist: '+str(sbu_connect_list)+'\n'
            write2file(ligandpath,"/ligand.txt",tmpstr)

    tmpstr = str(name) + ", (min_max_linker_length,max_min_linker_length): " + \
                str(min_max_linker_length) + " , " +str(max_min_linker_length) + "\n"
    write2file(logpath,"/%s.log"%name,tmpstr)
    if min_max_linker_length < 3:
        write2file(linkerpath,"/short_ligands.txt",tmpstr)
    if min_max_linker_length > 2:
        # for N-C-C-N ligand ligand
        if max_min_linker_length == min_max_linker_length:
            long_ligands = True
        elif min_max_linker_length > 3:
            long_ligands = True

    """""""""
    In the case of long linkers, add second coordination shell without further checks. In the case of short linkers, start from metal
    and grow outwards using the include_extra_shells function
    """""""""
    linker_length_list = [len(linker_val) for linker_val in linker_list]
    if len(set(linker_length_list)) != 1:
        write2file(linkerpath,"/uneven.txt",str(name)+'\n') # Linkers are different lengths.
    if not min_max_linker_length < 2: # treating the 2 atom ligands differently! Need caution
        if long_ligands:
            tmpstr = "\nStructure has LONG ligand\n\n"
            write2file(logpath,"/%s.log"%name,tmpstr)
            # Expanding the number of atoms considered to be part of the SBU
            [[SBUlist.add(val) for val in  molcif.getBondedAtomsSmart(zero_first_shell)] for zero_first_shell in SBUlist.copy()] #First account for all of the carboxylic acid type linkers, add in the carbons.
        truncated_linkers = allatoms - SBUlist # Taking the difference of sets
        SBU_list, SBU_subgraphlist = get_closed_subgraph(SBUlist, truncated_linkers, adj_matrix)
        if not long_ligands:
            tmpstr = "\nStructure has SHORT ligand\n\n"
            write2file(logpath,"/%s.log"%name,tmpstr)
            SBU_list, SBU_subgraphlist = include_extra_shells(SBU_list,SBU_subgraphlist,molcif,adj_matrix)
    else:
        tmpstr = "Structure %s has extremely short ligands, check the outputs\n"%name
        write2file(ligandpath,"/ambiguous.txt",tmpstr)
        tmpstr = "Structure has extremely short ligands\n"
        write2file(logpath,"/%s.log"%name,tmpstr)
        tmpstr = "Structure has extremely short ligands\n"
        write2file(logpath,"/%s.log"%name,tmpstr)
        truncated_linkers = allatoms - removelist
        SBU_list, SBU_subgraphlist = get_closed_subgraph(removelist, truncated_linkers, adj_matrix)
        SBU_list, SBU_subgraphlist = include_extra_shells(SBU_list, SBU_subgraphlist, molcif, adj_matrix)
        SBU_list, SBU_subgraphlist = include_extra_shells(SBU_list, SBU_subgraphlist, molcif, adj_matrix)

    """""""""
    For the cases that have a linker subgraph, do the featurization.
    """""""""
    if len(linker_subgraphlist)>=1: # Featurize cases that did not fail.
        # try:
            descriptor_names, descriptors, lc_descriptor_names, lc_descriptors = make_MOF_SBU_RACs(SBU_list, SBU_subgraphlist, molcif, depth, name, cell_v, anc_atoms, sbupath, linkeranchors_superlist, Gval, connections_list, connections_subgraphlist)
            lig_descriptor_names, lig_descriptors = make_MOF_linker_RACs(linker_list, linker_subgraphlist, molcif, depth, name, cell_v, linkerpath, linkeranchors_superlist, Gval)
            full_names = descriptor_names+lig_descriptor_names+lc_descriptor_names #+ ECFP_names
            full_descriptors = list(descriptors)+list(lig_descriptors)+list(lc_descriptors)
            print(len(full_names),len(full_descriptors))
        # except:
        #     full_names = [0]
        #     full_descriptors = [0]
    elif len(linker_subgraphlist) == 1: # Only one linker identified.
        print(f'Suspicious featurization for {name}: Only one linker identified.')
        full_names = [1]
        full_descriptors = [1]
    else: # Means len(linker_subgraphlist) is zero.
        failure_str = f'Failed to featurize {name}: No linkers were identified.\n'
        full_names, full_descriptors = failure_response(path, failure_str)
        return full_names, full_descriptors
    if (len(full_names) <= 1) and (len(full_descriptors) <= 1):
        print(f'full_names is {full_names} and full_descriptors is {full_descriptors}')
        failure_str = f'Failed to featurize {name}: Only zero or one total linkers identified.\n'
        full_names, full_descriptors = failure_response(path, failure_str)
        return full_names, full_descriptors

    # Getting bond information if requested, and writing it to a TXT file.
    if get_sbu_linker_bond_info:
        bond_information_write(linker_list, linkeranchors_superlist, adj_matrix, molcif, cell_v, path)

    # Generating XYZ files of SBUs surrounded by linkers.
    if surrounded_sbu_file_generation:
        try:
            surrounded_sbu_gen(SBU_list, linker_list, sbupath, molcif, adj_matrix, cell_v, allatomtypes, name)
        except:
            tmpstr = "Failed to generate surrounded SBU"
            write2file(logpath,"/%s.log"%name,tmpstr)

    if detect_1D_rod_sbu:
        detect_1D_rod(SBU_list, molcif, allatomtypes, cell_v, logpath, name)

    return full_names, full_descriptors


##### Example of usage over a set of cif files.
# featurization_list = []
# import sys
# featurization_directory = sys.argv[1]
# for cif_file in os.listdir(featurization_directory+'/cif/'):
#     #### This first part gets the primitive cells ####
#     get_primitive(featurization_directory+'/cif/'+cif_file, featurization_directory+'/primitive/'+cif_file)
#     full_names, full_descriptors = get_MOF_descriptors(featurization_directory+'/primitive/'+cif_file,3,path=featurization_directory+'/',
#         xyzpath=featurization_directory+'/xyz/'+cif_file.replace('cif','xyz'))
#     full_names.append('filename')
#     full_descriptors.append(cif_file)
#     featurization = dict(zip(full_names, full_descriptors))
#     featurization_list.append(featurization)
# df = pd.DataFrame(featurization_list)
# df.to_csv('./full_featurization_frame.csv',index=False)
